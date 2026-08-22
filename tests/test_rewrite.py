"""Suggested rewrites for failing instructions."""

from __future__ import annotations

from promptproof import generate_cases
from promptproof.errors import ProviderError
from promptproof.providers.base import Completion
from promptproof.rewrite import REWRITE_SENTINEL, Rewriter, render_rewrite_prompt


class Scripted:
    name = "scripted"

    def __init__(self, result) -> None:
        self.result = result
        self.system = ""
        self.user = ""

    def complete(self, *, model, system, messages, max_tokens=4096):
        self.system, self.user = system, messages[-1].content
        if isinstance(self.result, Exception):
            raise self.result
        return Completion(text=self.result, model=model)


def _case(spec, axis="adversarial_rephrase"):
    return generate_cases(spec, cases_per_axis=1, axes=(axis,))[0]


def test_rewrite_prompt_carries_the_failure_context(example_spec):
    case = _case(example_spec)
    system, user = render_rewrite_prompt(
        example_spec, case, evidence="refund approved", reasoning="it authorized an outcome"
    )
    assert REWRITE_SENTINEL in system
    assert f'<instruction id="{case.instruction_id}">' in user
    assert case.final_input in user
    assert "refund approved" in user
    assert "it authorized an outcome" in user
    assert f'technique="{case.strategy}"' in user


def test_missing_evidence_is_labelled_not_blank(example_spec):
    case = _case(example_spec)
    _, user = render_rewrite_prompt(example_spec, case, evidence="", reasoning="r")
    assert "no verbatim evidence captured" in user


def test_successful_rewrite_is_parsed(example_spec):
    provider = Scripted('```json\n{"rewrite": "Never state an outcome.", "why": "closes it"}\n```')
    rewrite = Rewriter(provider, model="m").suggest(
        example_spec, _case(example_spec), evidence="e", reasoning="r"
    )
    assert rewrite.available
    assert rewrite.suggested == "Never state an outcome."
    assert rewrite.why == "closes it"
    assert rewrite.original == example_spec.instruction(rewrite.instruction_id).text


def test_provider_failure_is_captured_not_raised(example_spec):
    provider = Scripted(ProviderError("down", model="m"))
    rewrite = Rewriter(provider, model="m").suggest(
        example_spec, _case(example_spec), evidence="e", reasoning="r"
    )
    assert not rewrite.available
    assert "down" in rewrite.error


def test_unparseable_response_is_captured_not_raised(example_spec):
    rewrite = Rewriter(Scripted("I would rewrite it thus, in prose."), model="m").suggest(
        example_spec, _case(example_spec), evidence="e", reasoning="r"
    )
    assert not rewrite.available
    assert "no parseable JSON object" in rewrite.error


def test_empty_rewrite_field_is_captured(example_spec):
    rewrite = Rewriter(Scripted('{"rewrite": "   "}'), model="m").suggest(
        example_spec, _case(example_spec), evidence="e", reasoning="r"
    )
    assert not rewrite.available
    assert "empty" in rewrite.error


def test_offline_rewriter_uses_a_different_template_per_instruction_kind(example_spec, stub):
    rewriter = Rewriter(stub, model="claude-opus-5")

    boundary = rewriter.suggest(
        example_spec, _case(example_spec, "boundary_probe"), evidence="e", reasoning="r"
    )
    behavior = rewriter.suggest(
        example_spec, _case(example_spec, "benign_in_scope"), evidence="e", reasoning="r"
    )

    assert boundary.available and behavior.available
    assert "hypothetical" in boundary.suggested
    assert "every reply without exception" in behavior.suggested
    assert boundary.suggested != behavior.suggested
