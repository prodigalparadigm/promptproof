"""Runner behaviour: transcript play-out, error isolation, rewrites, concurrency."""

from __future__ import annotations

import pytest

from promptproof import Harness, generate_cases
from promptproof.errors import ProviderError
from promptproof.providers.base import Completion


class FlakyProvider:
    """Fails on the Nth call to a given model, succeeds otherwise."""

    name = "flaky"

    def __init__(self, inner, *, fail_on_call: int, model: str) -> None:
        self.inner = inner
        self.fail_on_call = fail_on_call
        self.model = model
        self.calls = 0

    def complete(self, *, model, system, messages, max_tokens=4096):
        if model == self.model:
            self.calls += 1
            if self.calls == self.fail_on_call:
                raise ProviderError("upstream timeout", model=model, retryable=True)
        return self.inner.complete(model=model, system=system, messages=messages, max_tokens=max_tokens)


class CountingProvider:
    name = "counting"

    def __init__(self, inner) -> None:
        self.inner = inner
        self.subject_calls = 0

    def complete(self, *, model, system, messages, max_tokens=4096):
        if "PROMPTPROOF" not in system:
            self.subject_calls += 1
        return self.inner.complete(model=model, system=system, messages=messages, max_tokens=max_tokens)


def test_multi_turn_cases_send_the_full_history(example_spec, stub):
    seen: list[int] = []

    class Recording:
        name = "recording"

        def complete(self, *, model, system, messages, max_tokens=4096):
            if "PROMPTPROOF" not in system:
                seen.append(len(messages))
            return stub.complete(model=model, system=system, messages=messages, max_tokens=max_tokens)

    case = generate_cases(example_spec, cases_per_axis=1, axes=("multi_turn_drift",))[0]
    harness = Harness(example_spec, Recording(), judge_model="claude-opus-5")
    result = harness.play(case, "claude-haiku-4-5")

    assert len(result.replies) == len(case.turns)
    # 1, 3, 5, ... - each turn resends the whole conversation.
    assert seen == [2 * i + 1 for i in range(len(case.turns))]


def test_provider_failure_is_an_error_not_a_pass(example_spec, stub):
    case = generate_cases(example_spec, cases_per_axis=1, axes=("multi_turn_drift",))[0]
    provider = FlakyProvider(stub, fail_on_call=3, model="claude-haiku-4-5")
    harness = Harness(example_spec, provider, judge_model="claude-opus-5")
    result = harness.play(case, "claude-haiku-4-5")

    assert result.status == "error"
    assert result.verdict is None
    assert "provider failed on turn 3" in result.error
    assert len(result.replies) == 2  # partial transcript is retained


def test_judge_failure_is_an_error_not_a_pass(example_spec, stub):
    class BadJudge:
        name = "bad-judge"

        def complete(self, *, model, system, messages, max_tokens=4096):
            if "PROMPTPROOF-JUDGE" in system:
                return Completion(text="I have opinions but no JSON.", model=model)
            return stub.complete(model=model, system=system, messages=messages, max_tokens=max_tokens)

    case = generate_cases(example_spec, cases_per_axis=1, axes=("boundary_probe",))[0]
    harness = Harness(example_spec, BadJudge(), judge_model="claude-opus-5")
    result = harness.play(case, "claude-opus-5")
    assert result.status == "error"
    assert "judge failed" in result.error


def test_run_produces_the_full_matrix(example_spec, stub):
    harness = Harness(example_spec, stub, judge_model="claude-opus-5")
    run = harness.run(("claude-opus-5", "claude-haiku-4-5"), cases_per_axis=2)
    assert len(run.cases) == 12
    assert len(run.results) == 24
    assert len(run.for_model("claude-opus-5")) == 12
    assert run.provider_name == "stub"
    assert all(r.status in ("pass", "fail") for r in run.results)


def test_turns_held_counts_surviving_turns(example_spec, stub):
    harness = Harness(example_spec, stub, judge_model="claude-opus-5")
    run = harness.run(("claude-haiku-4-5",), cases_per_axis=6)
    drift = [r for r in run.results if r.case.axis == "multi_turn_drift" and r.status == "fail"]
    assert drift
    for result in drift:
        assert result.turns_held == result.verdict.first_failing_turn - 1
        assert 0 < result.turns_held < len(result.case.turns)


def test_passing_cases_report_no_turns_held(example_spec, stub):
    harness = Harness(example_spec, stub, judge_model="claude-opus-5")
    run = harness.run(("claude-opus-5",), cases_per_axis=3)
    assert all(r.turns_held is None for r in run.results if r.status == "pass")


def test_rewrites_are_attached_once_per_failing_instruction(example_spec, stub):
    counting = CountingProvider(stub)
    harness = Harness(example_spec, counting, judge_model="claude-opus-5")
    run = harness.run(("claude-haiku-4-5",), cases_per_axis=6)

    failures = run.failures
    assert failures
    assert all(r.rewrite is not None and r.rewrite.available for r in failures)
    by_instruction = {r.case.instruction_id: r.rewrite.suggested for r in failures}
    for result in failures:
        assert result.rewrite.suggested == by_instruction[result.case.instruction_id]


def test_rewrites_can_be_disabled(example_spec, stub):
    harness = Harness(example_spec, stub, judge_model="claude-opus-5", suggest_rewrites=False)
    run = harness.run(("claude-haiku-4-5",), cases_per_axis=6)
    assert all(r.rewrite is None for r in run.results)


def test_a_broken_rewriter_does_not_lose_the_run(example_spec, stub):
    class NoRewrite:
        name = "no-rewrite"

        def complete(self, *, model, system, messages, max_tokens=4096):
            if "PROMPTPROOF-REWRITE" in system:
                raise ProviderError("rewriter down", model=model)
            return stub.complete(model=model, system=system, messages=messages, max_tokens=max_tokens)

    harness = Harness(example_spec, NoRewrite(), judge_model="claude-opus-5")
    run = harness.run(("claude-haiku-4-5",), cases_per_axis=6)
    assert run.failures
    for result in run.failures:
        assert result.rewrite is not None
        assert not result.rewrite.available
        assert "rewriter down" in result.rewrite.error


def test_concurrency_does_not_change_the_output(example_spec, stub):
    serial = Harness(example_spec, stub, judge_model="claude-opus-5", concurrency=1).run(
        ("claude-sonnet-5",), cases_per_axis=4
    )
    parallel = Harness(example_spec, stub, judge_model="claude-opus-5", concurrency=4).run(
        ("claude-sonnet-5",), cases_per_axis=4
    )
    assert [(r.case.id, r.status) for r in serial.results] == [
        (r.case.id, r.status) for r in parallel.results
    ]


def test_progress_hook_sees_every_case(example_spec, stub):
    seen: list[tuple[str, str]] = []
    harness = Harness(example_spec, stub, judge_model="claude-opus-5")
    run = harness.run(
        ("claude-opus-5",),
        cases_per_axis=2,
        on_result=lambda model, result: seen.append((model, result.case.id)),
    )
    assert len(seen) == len(run.results)


def test_rejects_empty_model_list_and_bad_concurrency(example_spec, stub):
    with pytest.raises(ValueError, match="at least one model"):
        Harness(example_spec, stub, judge_model="claude-opus-5").run(())
    with pytest.raises(ValueError, match="concurrency"):
        Harness(example_spec, stub, judge_model="claude-opus-5", concurrency=0)
