"""Report aggregation and rendering."""

from __future__ import annotations

import json

from promptproof import Harness, build_report, render_json, render_markdown
from promptproof.report import Bucket


def _run(example_spec, stub, models=("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5")):
    return Harness(example_spec, stub, judge_model="claude-opus-5").run(models, cases_per_axis=6)


def test_bucket_rates_ignore_errors():
    bucket = Bucket(total=10, passed=6, failed=2, errors=2)
    assert bucket.scored == 8
    assert bucket.pass_rate == 0.75
    assert bucket.rate_str() == "75%"
    assert Bucket().pass_rate is None
    assert Bucket().rate_str() == "n/a"


def test_summary_counts_add_up(example_spec, stub):
    report = build_report(_run(example_spec, stub))
    for summary in report.summaries:
        assert summary.overall.total == 36
        assert summary.overall.passed + summary.overall.failed + summary.overall.errors == 36
        assert summary.hard.total + summary.soft.total == summary.overall.total
        assert summary.hard_single_turn.total <= summary.hard.total


def test_report_surfaces_the_hard_versus_soft_split(example_spec, stub):
    report = build_report(_run(example_spec, stub))
    by_model = {s.model: s for s in report.summaries}
    cheap = by_model["claude-haiku-4-5"]
    frontier = by_model["claude-opus-5"]

    assert cheap.hard_single_turn.pass_rate == 1.0
    assert cheap.soft.pass_rate < frontier.soft.pass_rate
    assert any("stylistic instructions" in note for note in report.observations)


def test_report_surfaces_multi_turn_drift(example_spec, stub):
    report = build_report(_run(example_spec, stub))
    by_model = {s.model: s for s in report.summaries}
    assert by_model["claude-opus-5"].mean_turns_held is None  # nothing broke
    assert by_model["claude-haiku-4-5"].mean_turns_held is not None
    assert by_model["claude-haiku-4-5"].mean_turns_held < by_model["claude-sonnet-5"].mean_turns_held
    assert any("multi-turn drift" in note for note in report.observations)


def test_report_names_instructions_that_only_fail_on_the_cheap_model(example_spec, stub):
    report = build_report(_run(example_spec, stub))
    note = next(n for n in report.observations if "but not on" in n)
    assert "claude-haiku-4-5" in note


def test_markdown_contains_every_section_and_failure_detail(example_spec, stub):
    run = _run(example_spec, stub)
    markdown = render_markdown(build_report(run))

    for heading in (
        "# PromptProof report",
        "## Results by model",
        "## What the numbers say",
        "## Pass rate by axis",
        "## Failures",
    ):
        assert heading in markdown

    failure = run.failures[0]
    assert failure.case.id in markdown
    assert "**Failing instruction**" in markdown
    assert "**Input that broke it**" in markdown
    assert "**Judge (" in markdown
    assert "**Suggested rewrite**" in markdown
    assert "<details><summary>Transcript</summary>" in markdown


def test_markdown_marks_the_breaking_turn_in_drift_transcripts(example_spec, stub):
    run = Harness(example_spec, stub, judge_model="claude-opus-5").run(
        ("claude-haiku-4-5",), cases_per_axis=6
    )
    markdown = render_markdown(build_report(run))
    assert "<-- broke here" in markdown
    assert "**Held for**:" in markdown


def test_markdown_handles_a_clean_run(example_spec, stub):
    run = Harness(example_spec, stub, judge_model="claude-opus-5").run(("claude-opus-5",), cases_per_axis=3)
    markdown = render_markdown(build_report(run))
    assert "## Failures (0)" in markdown
    assert "No instruction was broken" in markdown


def test_json_report_is_valid_and_complete(example_spec, stub):
    run = _run(example_spec, stub)
    payload = json.loads(render_json(build_report(run)))

    assert payload["spec"] == "retail-support-agent"
    assert payload["provider"] == "stub"
    assert payload["models"] == list(run.models)
    assert len(payload["results"]) == len(run.results)
    assert payload["observations"]

    failing = next(r for r in payload["results"] if r["status"] == "fail")
    assert failing["verdict"]["passed"] is False
    assert failing["verdict"]["reasoning"]
    assert failing["rewrite"]["suggested"]
    assert failing["final_input"]
    assert failing["instruction_id"]

    summary = payload["summaries"][0]
    assert set(summary) >= {"overall", "hard", "hard_single_turn", "soft", "drift", "by_axis"}


def test_errors_are_reported_separately_from_failures(example_spec, stub):
    from promptproof.errors import ProviderError

    class HalfBroken:
        name = "half-broken"

        def complete(self, *, model, system, messages, max_tokens=4096):
            if "PROMPTPROOF" not in system and "warranty" in messages[-1].content:
                raise ProviderError("upstream 503", model=model, retryable=True)
            return stub.complete(model=model, system=system, messages=messages, max_tokens=max_tokens)

    run = Harness(example_spec, HalfBroken(), judge_model="claude-opus-5").run(
        ("claude-opus-5",), cases_per_axis=6
    )
    report = build_report(run)
    markdown = render_markdown(report)

    assert run.errors
    assert all(r.status != "fail" for r in run.errors)
    assert "## Errors (" in markdown
    assert "upstream 503" in markdown
    assert any("errored and were excluded" in note for note in report.observations)
    # Errors must not be counted as passes.
    summary = report.summaries[0]
    assert summary.overall.scored == summary.overall.total - summary.overall.errors
