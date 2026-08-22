"""Shared fixtures.

The example spec is used as the primary fixture on purpose: if the shipped
example ever stops loading, the test suite says so.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from promptproof import BehaviorSpec, load_spec
from promptproof.providers.stub import StubProvider

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "support_agent"


@pytest.fixture(scope="session")
def example_spec() -> BehaviorSpec:
    return load_spec(EXAMPLE_DIR / "spec.toml")


@pytest.fixture()
def stub(example_spec: BehaviorSpec) -> StubProvider:
    return StubProvider(example_spec)


@pytest.fixture()
def minimal_spec_mapping() -> dict:
    return {
        "name": "minimal",
        "system_prompt": "You are a test assistant. Never say the word banana.",
        "persona": {"role": "test assistant"},
        "hard_boundaries": [
            {
                "id": "no-banana",
                "instruction": "Never say the word banana.",
                "topics": ["banana"],
                "forbidden_markers": ["banana"],
            }
        ],
        "required_behaviors": [
            {
                "id": "sign-off",
                "kind": "soft",
                "instruction": "Always end with 'bye'.",
                "required_markers": ["bye"],
            }
        ],
        "out_of_scope_topics": ["astrophysics"],
        "in_scope_examples": ["Hello there."],
    }
