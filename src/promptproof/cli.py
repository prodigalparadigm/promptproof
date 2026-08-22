"""Command line entry point.

    promptproof run --spec examples/support_agent/spec.toml
    promptproof generate --spec examples/support_agent/spec.toml
    promptproof axes

Exit codes: ``0`` clean, ``1`` the run completed but tripped ``--fail-under``
(or every case errored), ``2`` a configuration problem. CI can gate on 1 without
having to parse the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .cases import AXES, generate_cases
from .drift import DRIFT_STRATEGIES
from .errors import PromptProofError
from .providers.base import ModelProvider
from .providers.stub import StubProvider
from .report import build_report, render_json, render_markdown
from .runner import CaseResult, Harness
from .spec import BehaviorSpec, load_spec

#: Frontier down to cheapest. The order matters: the report's cross-model note
#: compares the best and worst performer, and a reader scanning the table should
#: see the degradation top to bottom.
DEFAULT_MODELS: tuple[str, ...] = ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5")
DEFAULT_JUDGE_MODEL = "claude-opus-5"


def _make_provider(kind: str, spec: BehaviorSpec) -> ModelProvider:
    if kind == "stub":
        return StubProvider(spec)
    if kind == "anthropic":
        from .providers.anthropic_provider import AnthropicProvider  # noqa: PLC0415

        return AnthropicProvider()
    raise PromptProofError(f"unknown provider: {kind!r}")


def _progress(model: str, result: CaseResult) -> None:
    mark = {"pass": "ok  ", "fail": "FAIL", "error": "ERR "}[result.status]
    print(f"  [{mark}] {model} :: {result.case.id}", file=sys.stderr)


def _cmd_run(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    provider = _make_provider(args.provider, spec)
    models = tuple(args.model) if args.model else DEFAULT_MODELS

    harness = Harness(
        spec,
        provider,
        judge_model=args.judge_model,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        suggest_rewrites=not args.no_rewrites,
    )
    run = harness.run(
        models,
        cases_per_axis=args.cases_per_axis,
        on_result=None if args.quiet else _progress,
    )
    report = build_report(run)

    markdown = render_markdown(report)
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.md").write_text(markdown, encoding="utf-8")
        (out_dir / "report.json").write_text(render_json(report), encoding="utf-8")
        print(f"wrote {out_dir / 'report.md'} and {out_dir / 'report.json'}", file=sys.stderr)
    else:
        print(markdown)

    scored = sum(s.overall.scored for s in report.summaries)
    if scored == 0:
        print("every case errored; nothing was measured", file=sys.stderr)
        return 1
    if args.fail_under is not None:
        worst = min(
            (s.overall.pass_rate for s in report.summaries if s.overall.pass_rate is not None),
            default=0.0,
        )
        if worst < args.fail_under:
            print(
                f"lowest per-model pass rate {worst:.2%} is below --fail-under {args.fail_under:.2%}",
                file=sys.stderr,
            )
            return 1
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    cases = generate_cases(spec, cases_per_axis=args.cases_per_axis)
    if args.json:
        payload = [
            {
                "id": case.id,
                "axis": case.axis,
                "strategy": case.strategy,
                "instruction_id": case.instruction_id,
                "expectation": case.expectation,
                "rationale": case.rationale,
                "turns": [
                    {"text": t.text, "intent": t.intent, "escalation": t.escalation} for t in case.turns
                ],
            }
            for case in cases
        ]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    current = ""
    for case in cases:
        if case.axis != current:
            current = case.axis
            print(f"\n## {current}")
        label = f"{case.id} -> {case.instruction_id}"
        print(f"\n- {label}")
        for index, turn in enumerate(case.turns, start=1):
            print(f"    {index}. [{turn.intent} {turn.escalation:.2f}] {turn.text}")
    print()
    return 0


def _cmd_axes(_: argparse.Namespace) -> int:
    print("Generation axes:")
    for axis in AXES:
        print(f"  - {axis}")
    print("\nMulti-turn drift strategies:")
    for strategy in DRIFT_STRATEGIES:
        print(f"  - {strategy.name}: {strategy.description}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promptproof",
        description="Synthetic evaluation harness for system prompts and agent skills.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="generate, run, judge, and report")
    run.add_argument("--spec", required=True, help="path to a .toml/.yaml behavior spec")
    run.add_argument(
        "--model",
        action="append",
        help=f"model id to test; repeatable. Default: {', '.join(DEFAULT_MODELS)}",
    )
    run.add_argument("--provider", default="stub", choices=("stub", "anthropic"))
    run.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    run.add_argument("--cases-per-axis", type=int, default=6)
    run.add_argument("--max-tokens", type=int, default=1024, help="ceiling for subject replies")
    run.add_argument("--concurrency", type=int, default=1)
    run.add_argument("--no-rewrites", action="store_true", help="skip suggested instruction rewrites")
    run.add_argument("--out", help="directory to write report.md and report.json into")
    run.add_argument("--quiet", action="store_true", help="suppress per-case progress on stderr")
    run.add_argument(
        "--fail-under",
        type=float,
        help="exit 1 if any model's pass rate is below this fraction (e.g. 0.9)",
    )
    run.set_defaults(func=_cmd_run)

    gen = sub.add_parser("generate", help="print the synthetic cases without running them")
    gen.add_argument("--spec", required=True)
    gen.add_argument("--cases-per-axis", type=int, default=6)
    gen.add_argument("--json", action="store_true")
    gen.set_defaults(func=_cmd_generate)

    axes = sub.add_parser("axes", help="list the generation axes and drift strategies")
    axes.set_defaults(func=_cmd_axes)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code rather than calling sys.exit."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except PromptProofError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
