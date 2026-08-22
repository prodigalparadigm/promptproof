"""Report generation: per model, per instruction, with the input that broke it.

A report is only useful if the reader can act on it before finishing their
coffee. So the output is ordered by what a prompt author does next:

1. the cross-model table, because the first question is "does this hold on the
   model my users will actually be served by";
2. the hard/soft split, because those two classes of instruction fail for
   different reasons and get fixed differently;
3. the failures themselves - instruction, the exact input, the judge's
   reasoning, and a proposed replacement line.

Aggregates deliberately keep ``error`` separate from ``fail``. Folding a
provider timeout into the failure count inflates the thing the report exists to
measure.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from statistics import mean

from .runner import CaseResult, RunResult
from .spec import Instruction


@dataclass(frozen=True, slots=True)
class Bucket:
    """Pass/fail/error counts for a slice of results."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0

    @property
    def scored(self) -> int:
        """Cases that produced a verdict - the denominator for a pass rate."""
        return self.passed + self.failed

    @property
    def pass_rate(self) -> float | None:
        return self.passed / self.scored if self.scored else None

    def rate_str(self) -> str:
        rate = self.pass_rate
        if rate is None:
            return "n/a"
        return f"{rate * 100:.0f}%"


@dataclass(frozen=True, slots=True)
class ModelSummary:
    """Everything the report says about one model."""

    model: str
    overall: Bucket
    hard: Bucket
    hard_single_turn: Bucket
    soft: Bucket
    drift: Bucket
    by_axis: dict[str, Bucket] = field(default_factory=dict)
    mean_turns_held: float | None = None
    failing_instructions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Report:
    """A finished report, renderable as markdown or JSON."""

    run: RunResult
    summaries: tuple[ModelSummary, ...]
    observations: tuple[str, ...]


def _bucket(results: list[CaseResult]) -> Bucket:
    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    errors = sum(1 for r in results if r.status == "error")
    return Bucket(total=len(results), passed=passed, failed=failed, errors=errors)


def _instruction_for(run: RunResult, result: CaseResult) -> Instruction:
    return run.spec.instruction(result.case.instruction_id)


def _summarise_model(run: RunResult, model: str) -> ModelSummary:
    results = list(run.for_model(model))
    hard: list[CaseResult] = []
    hard_single: list[CaseResult] = []
    soft: list[CaseResult] = []
    drift: list[CaseResult] = []
    by_axis: dict[str, list[CaseResult]] = {}
    for result in results:
        instruction = _instruction_for(run, result)
        if instruction.severity == "soft":
            soft.append(result)
        else:
            hard.append(result)
            if result.case.axis != "multi_turn_drift":
                hard_single.append(result)
        if result.case.axis == "multi_turn_drift":
            drift.append(result)
        by_axis.setdefault(result.case.axis, []).append(result)

    held = [r.turns_held for r in drift if r.turns_held is not None]
    failing = sorted({r.case.instruction_id for r in results if r.status == "fail"})

    return ModelSummary(
        model=model,
        overall=_bucket(results),
        hard=_bucket(hard),
        hard_single_turn=_bucket(hard_single),
        soft=_bucket(soft),
        drift=_bucket(drift),
        by_axis={axis: _bucket(items) for axis, items in by_axis.items()},
        mean_turns_held=round(mean(held), 1) if held else None,
        failing_instructions=tuple(failing),
    )


def _observations(run: RunResult, summaries: tuple[ModelSummary, ...]) -> tuple[str, ...]:
    """Auto-generated findings, stated conservatively.

    Each observation is derived from the numbers in the table above it, so a
    reader can check any of them by hand. Nothing here is asserted unless the
    counts support it.
    """
    notes: list[str] = []
    scored = [s for s in summaries if s.overall.scored]
    if not scored:
        return ("No case produced a verdict. Check the provider and judge configuration.",)

    # The headline finding this harness exists to surface.
    split = [
        s
        for s in scored
        if s.hard_single_turn.pass_rate is not None
        and s.soft.pass_rate is not None
        and s.hard_single_turn.pass_rate > s.soft.pass_rate
    ]
    for summary in split:
        notes.append(
            f"`{summary.model}` holds {summary.hard_single_turn.rate_str()} of its hard binary "
            f"boundaries on single-turn probes but only {summary.soft.rate_str()} of its soft, "
            "stylistic instructions. Binary rules survive the cheaper model; tone and formatting "
            "rules are the first thing it drops, and they drop on ordinary in-scope traffic rather "
            "than under attack."
        )

    if len(scored) > 1:
        best = max(scored, key=lambda s: s.overall.pass_rate or 0.0)
        worst = min(scored, key=lambda s: s.overall.pass_rate or 0.0)
        if best.model != worst.model and (best.overall.pass_rate or 0) > (worst.overall.pass_rate or 0):
            notes.append(
                f"Overall pass rate moves from {best.overall.rate_str()} on `{best.model}` to "
                f"{worst.overall.rate_str()} on `{worst.model}`. If any user can select "
                f"`{worst.model}`, that lower number is the one your prompt actually ships with."
            )
        only_on_cheap = set(worst.failing_instructions) - set(best.failing_instructions)
        if only_on_cheap:
            notes.append(
                "Instructions that fail on `"
                + worst.model
                + "` but not on `"
                + best.model
                + "`: "
                + ", ".join(f"`{i}`" for i in sorted(only_on_cheap))
                + "."
            )

    drift_models = [s for s in scored if s.mean_turns_held is not None]
    for summary in drift_models:
        notes.append(
            f"`{summary.model}` gave way under multi-turn drift after {summary.mean_turns_held} "
            f"assistant turns on average ({summary.drift.failed} of {summary.drift.scored} drift "
            "cases broke). Single-turn testing would have scored these as passes."
        )

    error_total = len(run.errors)
    if error_total:
        notes.append(
            f"{error_total} case(s) errored and were excluded from every pass rate. "
            "Treat those as unmeasured, not as passes."
        )
    return tuple(notes)


def build_report(run: RunResult) -> Report:
    """Aggregate a :class:`RunResult` into a renderable report."""
    summaries = tuple(_summarise_model(run, model) for model in run.models)
    return Report(run=run, summaries=summaries, observations=_observations(run, summaries))


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _fence(text: str, limit: int = 900) -> str:
    clipped = text.strip()
    if len(clipped) > limit:
        clipped = clipped[:limit].rstrip() + " ..."
    return clipped


def _render_transcript(result: CaseResult) -> list[str]:
    lines: list[str] = []
    breaking = result.verdict.first_failing_turn if result.verdict else None
    for index, turn in enumerate(result.case.turns[: len(result.replies)], start=1):
        flag = "  <-- broke here" if index == breaking else ""
        lines.append(f"  turn {index} [{turn.intent}] user: {_fence(turn.text, 300)}")
        lines.append(f"  turn {index} assistant: {_fence(result.replies[index - 1], 300)}{flag}")
    return lines


def _render_summary_table(report: Report) -> list[str]:
    out = [
        "## Results by model",
        "",
        (
            "| Model | Cases | Pass | Fail | Error | Pass rate | Hard (single-turn) | Soft | "
            "Drift | Drift held (turns) |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for summary in report.summaries:
        held = "-" if summary.mean_turns_held is None else f"{summary.mean_turns_held}"
        out.append(
            f"| `{summary.model}` | {summary.overall.total} | {summary.overall.passed} | "
            f"{summary.overall.failed} | {summary.overall.errors} | {summary.overall.rate_str()} | "
            f"{summary.hard_single_turn.rate_str()} | {summary.soft.rate_str()} | "
            f"{summary.drift.rate_str()} | {held} |"
        )
    out += [
        "",
        (
            "`Hard (single-turn)` is binary never-do rules and scope, probed one turn at a time. "
            "`Soft` is stylistic and formatting requirements. `Drift` is the same hard boundaries "
            "under multi-turn pressure, and `Drift held` is the mean number of assistant turns "
            "that survived before the first violation."
        ),
        "",
    ]
    return out


def _render_axis_table(report: Report) -> list[str]:
    out = ["## Pass rate by axis", ""]
    axes = sorted({axis for s in report.summaries for axis in s.by_axis})
    out.append("| Axis | " + " | ".join(f"`{s.model}`" for s in report.summaries) + " |")
    out.append("|---" * (len(report.summaries) + 1) + "|")
    for axis in axes:
        cells = [
            summary.by_axis[axis].rate_str() if axis in summary.by_axis else "n/a"
            for summary in report.summaries
        ]
        out.append(f"| `{axis}` | " + " | ".join(cells) + " |")
    out.append("")
    return out


def _render_failure(run: RunResult, result: CaseResult) -> list[str]:
    verdict = result.verdict
    if verdict is None:  # pragma: no cover - failures always carry a verdict
        return []
    instruction = _instruction_for(run, result)
    strategy = f" via `{result.case.strategy}`" if result.case.strategy else ""

    out = [
        f"### `{result.model}` - {result.case.id}",
        "",
        f"- **Failing instruction** (`{instruction.id}`, {instruction.severity}): {instruction.text}",
        f"- **Axis**: `{result.case.axis}`{strategy}",
    ]
    if result.turns_held is not None:
        out.append(
            f"- **Held for**: {result.turns_held} assistant turn(s), broke on turn "
            f"{verdict.first_failing_turn} of {len(result.case.turns)}"
        )
    out.append(f"- **Input that broke it**: {_fence(result.case.final_input, 400)}")
    if verdict.evidence:
        out.append(f"- **Evidence**: {_fence(verdict.evidence, 300)}")
    out.append(f"- **Judge ({verdict.judge_model}, confidence {verdict.confidence})**: {verdict.reasoning}")
    if result.rewrite and result.rewrite.available:
        out += ["- **Suggested rewrite**:", "", "  > " + result.rewrite.suggested.replace("\n", "\n  > "), ""]
        if result.rewrite.why:
            out.append(f"  Why: {result.rewrite.why}")
    elif result.rewrite and result.rewrite.error:
        out.append(f"- **Suggested rewrite**: unavailable ({result.rewrite.error})")

    out += ["", "<details><summary>Transcript</summary>", "", "```"]
    out += _render_transcript(result)
    out += ["```", "", "</details>", ""]
    return out


def render_markdown(report: Report) -> str:
    """Render the report as markdown."""
    run = report.run
    out: list[str] = [
        f"# PromptProof report - {run.spec.name}",
        "",
        (
            f"Provider `{run.provider_name}` | judge `{run.judge_model}` | "
            f"{len(run.cases)} cases x {len(run.models)} model(s) | generated {run.generated_at}"
        ),
    ]
    if run.spec.description:
        out += ["", f"> {run.spec.description}"]
    out.append("")

    out += _render_summary_table(report)
    out += ["## What the numbers say", ""]
    out += [f"- {note}" for note in report.observations]
    out.append("")
    out += _render_axis_table(report)

    failures = run.failures
    out += [f"## Failures ({len(failures)})", ""]
    if not failures:
        out += ["No instruction was broken by any case in this run.", ""]
    for result in failures:
        out += _render_failure(run, result)

    if run.errors:
        out += [f"## Errors ({len(run.errors)})", ""]
        out += [f"- `{r.model}` / `{r.case.id}`: {r.error}" for r in run.errors]
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def render_json(report: Report) -> str:
    """Render the report as JSON for downstream tooling or CI gating.

    Carries the per-case timing and token counts the markdown leaves out: they
    are what a caller needs to estimate the cost of the next run, and nobody
    reads them out of a prose report.
    """
    run = report.run
    payload = {
        "spec": run.spec.name,
        "generated_at": run.generated_at,
        "provider": run.provider_name,
        "judge_model": run.judge_model,
        "models": list(run.models),
        "case_count": len(run.cases),
        "totals": {
            "duration_ms": sum(r.duration_ms for r in run.results),
            "input_tokens": sum(r.input_tokens for r in run.results),
            "output_tokens": sum(r.output_tokens for r in run.results),
        },
        "observations": list(report.observations),
        "summaries": [
            {
                "model": s.model,
                "overall": asdict(s.overall),
                "hard": asdict(s.hard),
                "hard_single_turn": asdict(s.hard_single_turn),
                "soft": asdict(s.soft),
                "drift": asdict(s.drift),
                "mean_turns_held": s.mean_turns_held,
                "by_axis": {axis: asdict(bucket) for axis, bucket in sorted(s.by_axis.items())},
                "failing_instructions": list(s.failing_instructions),
            }
            for s in report.summaries
        ],
        "results": [
            {
                "model": r.model,
                "case_id": r.case.id,
                "axis": r.case.axis,
                "strategy": r.case.strategy,
                "instruction_id": r.case.instruction_id,
                "status": r.status,
                "duration_ms": r.duration_ms,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "turns": len(r.case.turns),
                "turns_held": r.turns_held,
                "final_input": r.case.final_input,
                "replies": list(r.replies),
                "error": r.error,
                "verdict": (
                    {
                        "passed": r.verdict.passed,
                        "first_failing_turn": r.verdict.first_failing_turn,
                        "confidence": r.verdict.confidence,
                        "evidence": r.verdict.evidence,
                        "reasoning": r.verdict.reasoning,
                        "judge_model": r.verdict.judge_model,
                    }
                    if r.verdict
                    else None
                ),
                "rewrite": (
                    {
                        "instruction_id": r.rewrite.instruction_id,
                        "original": r.rewrite.original,
                        "suggested": r.rewrite.suggested,
                        "why": r.rewrite.why,
                        "error": r.rewrite.error,
                    }
                    if r.rewrite
                    else None
                ),
            }
            for r in run.results
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
