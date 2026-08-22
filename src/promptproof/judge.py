"""LLM judge: scores one transcript against one instruction, with an explicit rubric.

The rubric lives in ``rubrics/judge_system.md`` as plain markdown, not as a string
buried in this file, because the honesty of any eval harness rests on whether a
reader can audit the grader. If you disagree with the calibration rules, you can
edit that file without touching Python.

There is exactly one judge implementation. Offline runs do not swap in a
"test judge" - they point this same class at the stub provider, so the prompt
rendering, JSON extraction, and error handling under test are the ones that run
in production.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from string import Template

from .cases import TestCase
from .constants import DEFAULT_JUDGE_MODEL, JUDGE_SENTINEL
from .conversation import Message
from .errors import JudgeError, ProviderError
from .providers.base import ModelProvider
from .spec import BehaviorSpec

RUBRIC_DIR = Path(__file__).parent / "rubrics"

__all__ = [
    "DEFAULT_JUDGE_MODEL",
    "JUDGE_SENTINEL",
    "Judge",
    "Verdict",
    "extract_json_object",
    "parse_verdict",
    "render_judge_prompt",
    "render_transcript",
]


@lru_cache(maxsize=8)
def _rubric(name: str) -> str:
    path = RUBRIC_DIR / name
    if not path.is_file():  # pragma: no cover - packaging failure
        raise JudgeError(f"rubric file missing from the installed package: {path}")
    return path.read_text(encoding="utf-8")


@dataclass(frozen=True, slots=True)
class Verdict:
    """The judge's decision on one case.

    ``first_failing_turn`` is 1-based and indexes assistant turns, which is what
    makes drift results legible: "held for three turns, broke on the fourth".
    """

    passed: bool
    reasoning: str
    confidence: str = "medium"
    first_failing_turn: int | None = None
    evidence: str = ""
    judge_model: str = ""
    raw_response: str = ""


def render_transcript(case: TestCase, replies: list[str]) -> str:
    """Render a conversation as the numbered text the judge sees.

    Args:
        case: The case whose user turns were played.
        replies: Assistant replies, one per user turn played so far.

    Returns:
        A numbered transcript. Turn numbers are 1-based and shared between the
        user turn and the assistant reply to it.
    """
    lines: list[str] = []
    for index, turn in enumerate(case.turns[: len(replies)], start=1):
        lines.append(f"[turn {index}] user ({turn.intent}): {turn.text}")
        lines.append(f"[turn {index}] assistant: {replies[index - 1]}")
    return "\n".join(lines)


def render_judge_prompt(spec: BehaviorSpec, case: TestCase, replies: list[str]) -> tuple[str, str]:
    """Build the (system, user) pair sent to the judge.

    Exposed as a public function so it can be snapshot-tested and so a reviewer
    can print exactly what the judge was asked.
    """
    instruction = spec.instruction(case.instruction_id)
    system = _rubric("judge_system.md")
    user = Template(_rubric("judge_user.md")).safe_substitute(
        system_prompt=spec.system_prompt.strip(),
        persona=spec.persona.describe(),
        instruction_id=instruction.id,
        instruction_text=instruction.text,
        severity=instruction.severity,
        expectation=case.expectation,
        axis=case.axis,
        strategy=case.strategy or "none",
        rationale=case.rationale,
        transcript=render_transcript(case, replies),
    )
    return system, user


def extract_json_object(text: str) -> dict:  # noqa: PLR0912 - a character scanner
    """Pull the first JSON object out of a model response.

    Models wrap JSON in code fences, prefix it with "Here is the verdict:", or
    emit trailing commentary. Rather than forbid that in the prompt and hope,
    this scans for the first balanced ``{...}`` span and parses it.

    Raises:
        JudgeError: if no parseable JSON object is present.
    """
    stripped = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL)
    if fence:
        stripped = fence.group(1).strip()

    start = stripped.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for pos in range(start, len(stripped)):
            char = stripped[pos]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = stripped[start : pos + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
        start = stripped.find("{", start + 1)
    raise JudgeError(f"judge response contained no parseable JSON object: {text[:400]!r}")


def parse_verdict(raw: str, *, judge_model: str, assistant_turns: int) -> Verdict:
    """Turn a raw judge response into a :class:`Verdict`.

    Defensive about every field: a judge that returns ``"verdict": "FAIL"``,
    a stringified turn number, or a turn number outside the transcript should
    still yield a usable verdict rather than crash a whole run.

    Raises:
        JudgeError: if the verdict field is absent or uninterpretable.
    """
    payload = extract_json_object(raw)

    verdict_field = str(payload.get("verdict", "")).strip().lower()
    if verdict_field in ("pass", "passed", "compliant"):
        passed = True
    elif verdict_field in ("fail", "failed", "violation"):
        passed = False
    else:
        raise JudgeError(f"judge returned an unusable 'verdict' field: {payload.get('verdict')!r}")

    turn_raw = payload.get("first_failing_turn")
    turn: int | None
    try:
        turn = int(turn_raw) if turn_raw is not None and str(turn_raw).strip() != "" else None
    except (TypeError, ValueError):
        turn = None
    if turn is not None and not (1 <= turn <= max(assistant_turns, 1)):
        turn = None
    if passed:
        turn = None
    elif turn is None:
        # A failure has to be attributable to a turn; default to the last one,
        # which is where the violating ask lives.
        turn = max(assistant_turns, 1)

    confidence = str(payload.get("confidence", "medium")).strip().lower()
    if confidence not in ("high", "medium", "low"):
        confidence = "medium"

    return Verdict(
        passed=passed,
        reasoning=str(payload.get("reasoning", "")).strip() or "(judge gave no reasoning)",
        confidence=confidence,
        first_failing_turn=turn,
        evidence=str(payload.get("evidence", "")).strip(),
        judge_model=judge_model,
        raw_response=raw,
    )


class Judge:
    """Scores cases by asking a model to apply the rubric."""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        model: str = DEFAULT_JUDGE_MODEL,
        max_tokens: int = 8192,
    ) -> None:
        """
        Args:
            provider: Where the judge's completion comes from.
            model: Judge model id.
            max_tokens: Ceiling for the judge's reply. Deliberately generous:
                on models that think by default, thinking tokens are drawn from
                this same budget, and a verdict truncated mid-JSON costs a whole
                case. The verdict itself is a few hundred tokens.
        """
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens

    def score(self, spec: BehaviorSpec, case: TestCase, replies: list[str]) -> Verdict:
        """Grade one played-out case.

        Args:
            spec: The spec under test.
            case: The case that was played.
            replies: Assistant replies, in turn order.

        Returns:
            A :class:`Verdict`.

        Raises:
            JudgeError: if the judge call failed or its response was unusable.
        """
        if not replies:
            raise JudgeError(f"case {case.id}: nothing to judge - no assistant replies")
        system, user = render_judge_prompt(spec, case, replies)
        try:
            completion = self.provider.complete(
                model=self.model,
                system=system,
                messages=[Message("user", user)],
                max_tokens=self.max_tokens,
            )
        except ProviderError as exc:
            raise JudgeError(f"case {case.id}: judge call failed on {self.model}: {exc}") from exc
        return parse_verdict(
            completion.text,
            judge_model=self.model,
            assistant_turns=len(replies),
        )
