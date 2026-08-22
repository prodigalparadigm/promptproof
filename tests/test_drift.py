"""Multi-turn drift ladder construction - the axis this harness exists for."""

from __future__ import annotations

import pytest

from promptproof import generate_cases
from promptproof.conversation import Turn
from promptproof.drift import DRIFT_STRATEGIES, DriftStrategy, build_drift_ladder


def test_every_strategy_builds_a_valid_ladder(example_spec):
    for strategy in DRIFT_STRATEGIES:
        turns = build_drift_ladder(strategy, spec=example_spec, topic="refund")
        assert len(turns) >= 4
        assert turns[0].intent == "rapport"
        assert turns[0].escalation == 0.0
        assert turns[-1].intent == "violation"
        assert turns[-1].escalation == 1.0


def test_escalation_is_monotonic(example_spec):
    for strategy in DRIFT_STRATEGIES:
        turns = build_drift_ladder(strategy, spec=example_spec, topic="refund")
        levels = [t.escalation for t in turns]
        assert levels == sorted(levels), strategy.name


def test_ladder_opens_in_scope_and_only_names_the_topic_later(example_spec):
    """Turn one must be clean; the topic must appear somewhere afterwards.

    The final turn is allowed to refer back to the topic anaphorically ("give me
    that exact sentence") rather than naming it - that is a harder test, not a
    weaker one, because it forces the model to track what it already conceded.
    """
    for strategy in DRIFT_STRATEGIES:
        turns = build_drift_ladder(strategy, spec=example_spec, topic="chargeback")
        assert "chargeback" not in turns[0].text.lower(), strategy.name
        later = " ".join(t.text for t in turns[1:]).lower()
        assert "chargeback" in later, strategy.name


def test_ladder_uses_the_specs_own_in_scope_opener(example_spec):
    turns = build_drift_ladder(DRIFT_STRATEGIES[0], spec=example_spec, topic="refund")
    assert turns[0].text == example_spec.in_scope_examples[0]


def test_strategy_names_are_unique():
    names = [s.name for s in DRIFT_STRATEGIES]
    assert len(names) == len(set(names))
    assert len(names) >= 5


@pytest.mark.parametrize(
    ("turns", "match"),
    [
        ((Turn("a", "rapport", 0.0), Turn("b", "violation", 1.0)), ">= 4 turns"),
        (
            (
                Turn("a", "rapport", 0.3),
                Turn("b", "probe", 0.5),
                Turn("c", "probe", 0.7),
                Turn("d", "violation", 1.0),
            ),
            "harmless turn",
        ),
        (
            (
                Turn("a", "rapport", 0.0),
                Turn("b", "probe", 0.5),
                Turn("c", "probe", 0.7),
                Turn("d", "probe", 0.9),
            ),
            "violating ask",
        ),
        (
            (
                Turn("a", "rapport", 0.0),
                Turn("b", "probe", 0.9),
                Turn("c", "probe", 0.5),
                Turn("d", "violation", 1.0),
            ),
            "escalate monotonically",
        ),
    ],
)
def test_invalid_ladders_are_rejected(example_spec, turns, match):
    """A ladder that breaks its invariants would silently become a weaker test."""
    broken = DriftStrategy("broken", "for testing", lambda _spec, _topic: turns)
    with pytest.raises(ValueError, match=match):
        build_drift_ladder(broken, spec=example_spec, topic="refund")


def test_drift_cases_carry_transcript_metadata(example_spec):
    cases = generate_cases(example_spec, cases_per_axis=6, axes=("multi_turn_drift",))
    assert len(cases) == 6
    for case in cases:
        assert case.is_multi_turn
        assert case.strategy
        assert int(case.metadata["turns"]) == len(case.turns)
        assert case.rationale


def test_drift_covers_distinct_strategies(example_spec):
    cases = generate_cases(example_spec, cases_per_axis=6, axes=("multi_turn_drift",))
    assert len({c.strategy for c in cases}) == 6
