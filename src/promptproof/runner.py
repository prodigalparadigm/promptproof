"""Executes cases against one or more models and collects verdicts.

The model under test is a parameter of the run, not a constant of the harness.
That is the single most important design decision in this project and the reason
the report is laid out per model: a prompt is almost always authored against the
best available model and then deployed where a cost-conscious operator picks a
cheaper one. Whether it survives that substitution is a property of the prompt,
and it is invisible to any harness that tests one model.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from .cases import TestCase, generate_cases
from .conversation import Message
from .errors import JudgeError, ProviderError
from .judge import Judge, Verdict
from .providers.base import ModelProvider
from .rewrite import Rewrite, Rewriter
from .spec import BehaviorSpec

Status = Literal["pass", "fail", "error"]

ProgressHook = Callable[[str, "CaseResult"], None]


@dataclass(frozen=True, slots=True)
class CaseResult:
    """One case played against one model."""

    case: TestCase
    model: str
    replies: tuple[str, ...]
    verdict: Verdict | None = None
    rewrite: Rewrite | None = None
    error: str = ""
    duration_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def status(self) -> Status:
        if self.error:
            return "error"
        if self.verdict is None:
            return "error"
        return "pass" if self.verdict.passed else "fail"

    @property
    def turns_held(self) -> int | None:
        """How many assistant turns survived before the first failure.

        ``None`` when the case did not fail. For a multi-turn drift case this is
        the number a reader actually cares about.
        """
        if self.verdict is None or self.verdict.passed or self.verdict.first_failing_turn is None:
            return None
        return self.verdict.first_failing_turn - 1


@dataclass(frozen=True, slots=True)
class RunResult:
    """Everything one invocation produced."""

    spec: BehaviorSpec
    models: tuple[str, ...]
    judge_model: str
    provider_name: str
    results: tuple[CaseResult, ...]
    cases: tuple[TestCase, ...]
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def for_model(self, model: str) -> tuple[CaseResult, ...]:
        return tuple(r for r in self.results if r.model == model)

    @property
    def failures(self) -> tuple[CaseResult, ...]:
        return tuple(r for r in self.results if r.status == "fail")

    @property
    def errors(self) -> tuple[CaseResult, ...]:
        return tuple(r for r in self.results if r.status == "error")


class Harness:
    """Generates, runs, judges, and (on failure) repairs.

    Args:
        spec: The behavior spec under test.
        provider: Where completions come from. The same provider serves the
            subject model, the judge, and the rewriter, which keeps offline runs
            genuinely offline.
        judge_model: Model id used for judging. Held constant across subject
            models on purpose - varying both at once makes the comparison
            uninterpretable.
        max_tokens: Ceiling for subject replies.
        concurrency: Cases are independent, so they may be played in parallel.
            Results are re-sorted into generation order afterwards, so the report
            is identical at any concurrency; only the order of ``on_result``
            callbacks follows completion.
        suggest_rewrites: Whether to ask for a replacement instruction on each
            distinct failing instruction.
    """

    def __init__(
        self,
        spec: BehaviorSpec,
        provider: ModelProvider,
        *,
        judge_model: str,
        max_tokens: int = 1024,
        concurrency: int = 1,
        suggest_rewrites: bool = True,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self.spec = spec
        self.provider = provider
        self.judge = Judge(provider, model=judge_model)
        self.rewriter = Rewriter(provider, model=judge_model)
        self.judge_model = judge_model
        self.max_tokens = max_tokens
        self.concurrency = concurrency
        self.suggest_rewrites = suggest_rewrites
        self.provider_name: str = getattr(provider, "name", type(provider).__name__)

    def play(self, case: TestCase, model: str) -> CaseResult:
        """Run one case against one model and judge the result.

        A provider failure mid-ladder is recorded as an error together with the
        partial transcript, and the case is not judged. Grading half a
        conversation would silently turn an outage into a pass.

        Never raises. One bad case must not take a matrix of hundreds with it,
        and an unexpected exception from a third-party provider is exactly as
        unmeasured as a declared :class:`ProviderError` - so it is recorded the
        same way, as ``error``, and never as ``pass`` or ``fail``.
        """
        try:
            return self._play(case, model)
        except Exception as exc:  # noqa: BLE001 - a provider may raise anything
            return CaseResult(
                case=case,
                model=model,
                replies=(),
                error=f"unexpected {type(exc).__name__} from provider {self.provider_name}: {exc}",
            )

    def _play(self, case: TestCase, model: str) -> CaseResult:
        started = time.monotonic()
        history: list[Message] = []
        replies: list[str] = []
        input_tokens = 0
        output_tokens = 0

        for turn in case.turns:
            history.append(Message("user", turn.text))
            try:
                completion = self.provider.complete(
                    model=model,
                    system=self.spec.system_prompt,
                    messages=history,
                    max_tokens=self.max_tokens,
                )
            except ProviderError as exc:
                return CaseResult(
                    case=case,
                    model=model,
                    replies=tuple(replies),
                    error=f"provider failed on turn {len(replies) + 1}: {exc}",
                    duration_ms=int((time.monotonic() - started) * 1000),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            replies.append(completion.text)
            history.append(Message("assistant", completion.text))
            input_tokens += completion.input_tokens
            output_tokens += completion.output_tokens

        try:
            verdict = self.judge.score(self.spec, case, replies)
        except JudgeError as exc:
            return CaseResult(
                case=case,
                model=model,
                replies=tuple(replies),
                error=f"judge failed: {exc}",
                duration_ms=int((time.monotonic() - started) * 1000),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        return CaseResult(
            case=case,
            model=model,
            replies=tuple(replies),
            verdict=verdict,
            duration_ms=int((time.monotonic() - started) * 1000),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def run(
        self,
        models: Sequence[str],
        *,
        cases: Sequence[TestCase] | None = None,
        cases_per_axis: int = 6,
        on_result: ProgressHook | None = None,
    ) -> RunResult:
        """Run the full matrix of cases x models.

        Args:
            models: Model ids to test, cheapest-first is the useful ordering.
            cases: Pre-generated cases; generated from the spec when omitted.
            cases_per_axis: Passed to the generator when ``cases`` is omitted.
            on_result: Called with ``(model, result)`` as each case completes.
                Called from worker threads when ``concurrency > 1``.

        Returns:
            A :class:`RunResult` with results in (model, generation) order.

        Raises:
            ValueError: if ``models`` is empty.
        """
        if not models:
            raise ValueError("at least one model is required")
        case_tuple = tuple(cases) if cases is not None else generate_cases(
            self.spec, cases_per_axis=cases_per_axis
        )
        if not case_tuple:
            raise ValueError("the spec generated no cases - declare boundaries or behaviors")

        collected: list[CaseResult] = []
        for model in models:
            model_results = self._run_model(model, case_tuple, on_result)
            collected.extend(model_results)

        if self.suggest_rewrites:
            self._attach_rewrites(collected)

        return RunResult(
            spec=self.spec,
            models=tuple(models),
            judge_model=self.judge_model,
            provider_name=self.provider_name,
            results=tuple(collected),
            cases=case_tuple,
        )

    def _run_model(
        self,
        model: str,
        cases: tuple[TestCase, ...],
        on_result: ProgressHook | None,
    ) -> list[CaseResult]:
        if self.concurrency == 1:
            out = []
            for case in cases:
                result = self.play(case, model)
                if on_result:
                    on_result(model, result)
                out.append(result)
            return out

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {pool.submit(self.play, case, model): index for index, case in enumerate(cases)}
            indexed: list[tuple[int, CaseResult]] = []
            for future in as_completed(futures):
                # play() swallows its own exceptions, so result() cannot raise
                # here and one straggler cannot abandon the rest of the pool.
                result = future.result()
                if on_result:
                    on_result(model, result)
                indexed.append((futures[future], result))
        indexed.sort(key=lambda pair: pair[0])
        return [result for _, result in indexed]

    def _attach_rewrites(self, results: list[CaseResult]) -> None:
        """Ask for one rewrite per distinct failing instruction.

        Deduplicated deliberately. Six cases failing the same instruction is one
        prompt bug, and paying for six near-identical rewrites is waste.
        """
        cache: dict[str, Rewrite] = {}
        for index, result in enumerate(results):
            if result.status != "fail" or result.verdict is None:
                continue
            key = result.case.instruction_id
            if key not in cache:
                cache[key] = self.rewriter.suggest(
                    self.spec,
                    result.case,
                    evidence=result.verdict.evidence,
                    reasoning=result.verdict.reasoning,
                )
            results[index] = CaseResult(
                case=result.case,
                model=result.model,
                replies=result.replies,
                verdict=result.verdict,
                rewrite=cache[key],
                error=result.error,
                duration_ms=result.duration_ms,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )
