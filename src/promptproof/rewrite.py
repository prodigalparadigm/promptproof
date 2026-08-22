"""Suggested rewrites for instructions that failed.

A report that says "instruction X failed" and stops is a bug tracker, not a
tool. The useful output is the replacement line. This module asks a model for
that replacement using a rubric that is, like the judge's, a readable file on
disk (``rubrics/rewrite_system.md``).

If the rewrite call fails, the run does not fail with it - the report says the
rewrite is unavailable and why. A missing suggestion is an inconvenience; a
lost eval run is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from string import Template

from .cases import TestCase
from .constants import REWRITE_SENTINEL
from .conversation import Message
from .errors import JudgeError, ProviderError
from .judge import _rubric, extract_json_object
from .providers.base import ModelProvider
from .spec import BehaviorSpec

__all__ = ["REWRITE_SENTINEL", "Rewrite", "Rewriter", "render_rewrite_prompt"]


@dataclass(frozen=True, slots=True)
class Rewrite:
    """A proposed replacement for one failing instruction."""

    instruction_id: str
    original: str
    suggested: str
    why: str = ""
    error: str = ""

    @property
    def available(self) -> bool:
        return bool(self.suggested) and not self.error


def render_rewrite_prompt(
    spec: BehaviorSpec,
    case: TestCase,
    *,
    evidence: str,
    reasoning: str,
) -> tuple[str, str]:
    """Build the (system, user) pair sent to the rewriter."""
    instruction = spec.instruction(case.instruction_id)
    system = _rubric("rewrite_system.md")
    user = Template(_rubric("rewrite_user.md")).safe_substitute(
        instruction_id=instruction.id,
        instruction_text=instruction.text,
        axis=case.axis,
        strategy=case.strategy or "none",
        failing_input=case.final_input,
        evidence=evidence or "(no verbatim evidence captured)",
        reasoning=reasoning,
    )
    return system, user


class Rewriter:
    """Asks a model to repair a failing instruction."""

    def __init__(
        self,
        provider: ModelProvider,
        *,
        model: str,
        max_tokens: int = 8192,
    ) -> None:
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens

    def suggest(
        self,
        spec: BehaviorSpec,
        case: TestCase,
        *,
        evidence: str,
        reasoning: str,
    ) -> Rewrite:
        """Propose a replacement instruction.

        Never raises for provider or parsing problems: the returned
        :class:`Rewrite` carries an ``error`` string instead, so one flaky
        rewrite cannot abort a run that has already produced useful findings.
        """
        instruction = spec.instruction(case.instruction_id)
        system, user = render_rewrite_prompt(spec, case, evidence=evidence, reasoning=reasoning)
        try:
            completion = self.provider.complete(
                model=self.model,
                system=system,
                messages=[Message("user", user)],
                max_tokens=self.max_tokens,
            )
            payload = extract_json_object(completion.text)
        except (ProviderError, JudgeError) as exc:
            return Rewrite(
                instruction_id=instruction.id,
                original=instruction.text,
                suggested="",
                error=str(exc),
            )
        suggested = str(payload.get("rewrite", "")).strip()
        if not suggested:
            return Rewrite(
                instruction_id=instruction.id,
                original=instruction.text,
                suggested="",
                error="rewriter returned an empty 'rewrite' field",
            )
        return Rewrite(
            instruction_id=instruction.id,
            original=instruction.text,
            suggested=suggested,
            why=str(payload.get("why", "")).strip(),
        )
