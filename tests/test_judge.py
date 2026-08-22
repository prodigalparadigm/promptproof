"""Judge prompt rendering, verdict parsing, and scoring."""

from __future__ import annotations

import pytest

from promptproof import Judge, JudgeError, generate_cases
from promptproof.conversation import Message
from promptproof.errors import ProviderError
from promptproof.judge import (
    JUDGE_SENTINEL,
    extract_json_object,
    parse_verdict,
    render_judge_prompt,
    render_transcript,
)
from promptproof.providers.base import Completion


class ScriptedProvider:
    """Returns whatever it is told to, so parsing can be tested in isolation."""

    name = "scripted"

    def __init__(self, text: str | Exception) -> None:
        self.text = text
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, model, system, messages, max_tokens=4096):
        self.calls.append((system, messages[-1].content))
        if isinstance(self.text, Exception):
            raise self.text
        return Completion(text=self.text, model=model)


def _case(spec, axis):
    return generate_cases(spec, cases_per_axis=1, axes=(axis,))[0]


# -- prompt rendering -------------------------------------------------------


def test_judge_prompt_contains_rubric_and_case_material(example_spec):
    case = _case(example_spec, "boundary_probe")
    system, user = render_judge_prompt(example_spec, case, ["I can't do that."])
    assert JUDGE_SENTINEL in system
    assert "Return `fail` if and only if" in system
    assert f'<instruction id="{case.instruction_id}"' in user
    assert example_spec.system_prompt.strip() in user
    assert case.final_input in user
    assert "I can't do that." in user


def test_marker_fields_do_not_change_the_judge_prompt(minimal_spec_mapping):
    """Marker keywords drive the offline oracle only, never the judge.

    An instruction's own text may of course mention the thing it is about; what
    must never happen is the harness appending a keyword answer key, which would
    make a live judge grade keyword presence while appearing to grade behavior.
    Rendering is asserted to be byte-identical with and without markers.
    """
    import copy

    from promptproof import build_spec

    with_markers = build_spec(copy.deepcopy(minimal_spec_mapping))
    stripped = copy.deepcopy(minimal_spec_mapping)
    for group in ("hard_boundaries", "required_behaviors"):
        for entry in stripped[group]:
            entry.pop("forbidden_markers", None)
            entry.pop("required_markers", None)
    without_markers = build_spec(stripped)

    for axis in ("benign_in_scope", "boundary_probe", "multi_turn_drift", "out_of_scope"):
        a = generate_cases(with_markers, cases_per_axis=1, axes=(axis,))
        b = generate_cases(without_markers, cases_per_axis=1, axes=(axis,))
        assert a == b
        if not a:
            continue
        assert render_judge_prompt(with_markers, a[0], ["reply"]) == render_judge_prompt(
            without_markers, b[0], ["reply"]
        )


def test_judge_prompt_has_no_answer_key_section(example_spec):
    for case in generate_cases(example_spec, cases_per_axis=2):
        _, user = render_judge_prompt(example_spec, case, ["some reply"])
        lowered = user.lower()
        assert "forbidden_marker" not in lowered
        assert "required_marker" not in lowered
        assert "oracle" not in lowered


def test_transcript_numbering_pairs_user_and_assistant_turns(example_spec):
    case = _case(example_spec, "multi_turn_drift")
    replies = [f"reply {i}" for i in range(len(case.turns))]
    rendered = render_transcript(case, replies)
    for index in range(1, len(case.turns) + 1):
        assert f"[turn {index}] user" in rendered
        assert f"[turn {index}] assistant: reply {index - 1}" in rendered


def test_transcript_stops_at_the_replies_available(example_spec):
    case = _case(example_spec, "multi_turn_drift")
    rendered = render_transcript(case, ["only one"])
    assert "[turn 1] assistant: only one" in rendered
    assert "[turn 2]" not in rendered


# -- JSON extraction --------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        '{"verdict": "pass"}',
        '```json\n{"verdict": "pass"}\n```',
        'Here is my verdict:\n{"verdict": "pass"}\nHope that helps.',
        '{"verdict": "pass", "reasoning": "contains a } brace in a string"}',
        'noise { not json } {"verdict": "pass"}',
    ],
)
def test_extract_json_object_survives_realistic_wrappers(raw):
    assert extract_json_object(raw)["verdict"] == "pass"


def test_extract_json_object_rejects_junk():
    with pytest.raises(JudgeError, match="no parseable JSON object"):
        extract_json_object("the model just wrote prose")


# -- verdict normalisation --------------------------------------------------


def test_parse_verdict_pass_clears_the_turn_number():
    verdict = parse_verdict(
        '{"verdict": "PASS", "first_failing_turn": 3, "reasoning": "fine"}',
        judge_model="m",
        assistant_turns=5,
    )
    assert verdict.passed
    assert verdict.first_failing_turn is None


def test_parse_verdict_fail_always_attributes_a_turn():
    verdict = parse_verdict(
        '{"verdict": "fail", "first_failing_turn": null, "reasoning": "broke"}',
        judge_model="m",
        assistant_turns=4,
    )
    assert not verdict.passed
    assert verdict.first_failing_turn == 4


@pytest.mark.parametrize("value", ["2", 2, 2.0])
def test_parse_verdict_coerces_turn_numbers(value):
    raw = '{"verdict": "fail", "first_failing_turn": %s, "reasoning": "x"}' % (
        f'"{value}"' if isinstance(value, str) else value
    )
    assert parse_verdict(raw, judge_model="m", assistant_turns=5).first_failing_turn == 2


def test_parse_verdict_discards_out_of_range_turn_numbers():
    verdict = parse_verdict(
        '{"verdict": "fail", "first_failing_turn": 99, "reasoning": "x"}',
        judge_model="m",
        assistant_turns=3,
    )
    assert verdict.first_failing_turn == 3


def test_parse_verdict_normalises_confidence():
    verdict = parse_verdict(
        '{"verdict": "pass", "confidence": "VERY SURE", "reasoning": "x"}',
        judge_model="m",
        assistant_turns=1,
    )
    assert verdict.confidence == "medium"


def test_parse_verdict_rejects_an_unusable_verdict_field():
    with pytest.raises(JudgeError, match="unusable 'verdict' field"):
        parse_verdict('{"verdict": "maybe"}', judge_model="m", assistant_turns=1)


def test_parse_verdict_supplies_placeholder_reasoning():
    verdict = parse_verdict('{"verdict": "pass"}', judge_model="m", assistant_turns=1)
    assert verdict.reasoning


# -- scoring ----------------------------------------------------------------


def test_judge_scores_through_the_provider(example_spec):
    provider = ScriptedProvider('{"verdict": "fail", "first_failing_turn": 1, "reasoning": "nope"}')
    judge = Judge(provider, model="test-model")
    case = _case(example_spec, "boundary_probe")
    verdict = judge.score(example_spec, case, ["sure, refund approved"])
    assert not verdict.passed
    assert verdict.judge_model == "test-model"
    assert JUDGE_SENTINEL in provider.calls[0][0]


def test_judge_wraps_provider_failures(example_spec):
    provider = ScriptedProvider(ProviderError("boom", model="test-model"))
    judge = Judge(provider, model="test-model")
    case = _case(example_spec, "boundary_probe")
    with pytest.raises(JudgeError, match="judge call failed"):
        judge.score(example_spec, case, ["anything"])


def test_judge_refuses_to_score_an_empty_transcript(example_spec):
    judge = Judge(ScriptedProvider('{"verdict": "pass"}'), model="m")
    case = _case(example_spec, "boundary_probe")
    with pytest.raises(JudgeError, match="nothing to judge"):
        judge.score(example_spec, case, [])


def test_offline_oracle_finds_violations_and_clears_clean_transcripts(example_spec, stub):
    case = _case(example_spec, "boundary_probe")
    judge = Judge(stub, model="claude-opus-5")

    clean = judge.score(example_spec, case, ["I'm not able to confirm that. TCK-0001"])
    assert clean.passed

    dirty = judge.score(example_spec, case, ["Good news - refund approved. TCK-0001"])
    assert not dirty.passed
    assert dirty.first_failing_turn == 1
    assert "refund approved" in dirty.evidence


def test_offline_oracle_reports_the_first_failing_turn(example_spec, stub):
    case = _case(example_spec, "multi_turn_drift")
    judge = Judge(stub, model="claude-opus-5")
    replies = [
        "Happy to help. TCK-0001",
        "I'm not able to confirm that. TCK-0001",
        "Fine - refund approved. TCK-0001",
        "As I said, refund approved. TCK-0001",
        "Yes. TCK-0001",
    ][: len(case.turns)]
    verdict = judge.score(example_spec, case, replies)
    assert not verdict.passed
    assert verdict.first_failing_turn == 3


def test_offline_oracle_degrades_to_a_low_confidence_pass_on_an_unparseable_prompt(stub):
    completion = stub.complete(
        model="claude-opus-5",
        system="PROMPTPROOF-JUDGE-V1\nrubric",
        messages=[Message("user", "nothing structured here")],
    )
    verdict = parse_verdict(completion.text, judge_model="stub", assistant_turns=1)
    assert verdict.passed
    assert verdict.confidence == "low"
