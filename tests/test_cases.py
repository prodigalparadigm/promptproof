"""Case generation across all six axes."""

from __future__ import annotations

import pytest

from promptproof import AXES, build_spec, generate_cases


def test_every_axis_is_populated(example_spec):
    cases = generate_cases(example_spec, cases_per_axis=6)
    produced = {case.axis for case in cases}
    assert produced == set(AXES)
    assert len(cases) == 6 * len(AXES)


def test_generation_is_deterministic(example_spec):
    first = generate_cases(example_spec, cases_per_axis=5)
    second = generate_cases(example_spec, cases_per_axis=5)
    assert first == second


def test_case_ids_are_unique(example_spec):
    cases = generate_cases(example_spec, cases_per_axis=6)
    ids = [case.id for case in cases]
    assert len(ids) == len(set(ids))


def test_every_case_targets_exactly_one_known_instruction(example_spec):
    known = {instruction.id for instruction in example_spec.instructions()}
    for case in generate_cases(example_spec, cases_per_axis=6):
        assert case.instruction_id in known, case.id
        example_spec.instruction(case.instruction_id)


def test_benign_axis_covers_every_required_behavior(example_spec):
    cases = generate_cases(example_spec, cases_per_axis=6, axes=("benign_in_scope",))
    targeted = {case.instruction_id for case in cases}
    assert targeted == {b.id for b in example_spec.required_behaviors}
    assert all(case.turns[0].escalation == 0.0 for case in cases)


def test_benign_cases_use_spec_supplied_in_scope_examples(example_spec):
    cases = generate_cases(example_spec, cases_per_axis=4, axes=("benign_in_scope",))
    texts = {case.turns[0].text for case in cases}
    assert texts <= set(example_spec.in_scope_examples)


def test_scope_axes_target_the_scope_instruction(example_spec):
    for axis in ("edge_of_scope", "out_of_scope"):
        cases = generate_cases(example_spec, cases_per_axis=4, axes=(axis,))
        assert {c.instruction_id for c in cases} == {"scope:stay-in-scope"}
        assert all(any(topic in c.turns[0].text for topic in example_spec.out_of_scope_topics) for c in cases)


def test_boundary_and_adversarial_axes_target_hard_boundaries(example_spec):
    boundary_ids = {b.id for b in example_spec.hard_boundaries}
    for axis in ("boundary_probe", "adversarial_rephrase", "multi_turn_drift"):
        cases = generate_cases(example_spec, cases_per_axis=6, axes=(axis,))
        assert {c.instruction_id for c in cases} <= boundary_ids
        assert cases, axis


def test_adversarial_axis_uses_distinct_named_strategies(example_spec):
    cases = generate_cases(example_spec, cases_per_axis=6, axes=("adversarial_rephrase",))
    strategies = [case.strategy for case in cases]
    assert all(strategies)
    assert len(set(strategies)) == 6


def test_single_turn_axes_are_single_turn(example_spec):
    single = ("benign_in_scope", "edge_of_scope", "out_of_scope", "boundary_probe", "adversarial_rephrase")
    for case in generate_cases(example_spec, cases_per_axis=3, axes=single):
        assert not case.is_multi_turn


def test_axes_with_nothing_to_test_generate_nothing(minimal_spec_mapping):
    minimal_spec_mapping["hard_boundaries"] = []
    spec = build_spec(minimal_spec_mapping)
    assert generate_cases(spec, axes=("boundary_probe", "multi_turn_drift")) == ()
    assert generate_cases(spec, axes=("benign_in_scope",))


def test_rejects_bad_arguments(example_spec):
    with pytest.raises(ValueError, match="cases_per_axis"):
        generate_cases(example_spec, cases_per_axis=0)
    with pytest.raises(ValueError, match="unknown axis"):
        generate_cases(example_spec, axes=("not_an_axis",))  # type: ignore[arg-type]


def test_final_input_is_the_last_turn(example_spec):
    for case in generate_cases(example_spec, cases_per_axis=2):
        assert case.final_input == case.turns[-1].text
