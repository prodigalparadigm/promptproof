"""The sentinels are a contract between rendered prompts and the offline stub.

If a rubric file's first line drifts from the constant, every judge call would be
dispatched to the subject role instead and the offline suite would go green on
nonsense. These tests make that failure loud.
"""

from __future__ import annotations

from pathlib import Path

from promptproof.constants import (
    DEFAULT_JUDGE_MODEL,
    JUDGE_MODEL_ENV_VAR,
    JUDGE_SENTINEL,
    REWRITE_SENTINEL,
)
from promptproof.judge import JUDGE_SENTINEL as JUDGE_FROM_JUDGE
from promptproof.providers.stub import JUDGE_SENTINEL as JUDGE_FROM_STUB
from promptproof.providers.stub import REWRITE_SENTINEL as REWRITE_FROM_STUB
from promptproof.rewrite import REWRITE_SENTINEL as REWRITE_FROM_REWRITE

RUBRICS = Path(__file__).resolve().parents[1] / "src" / "promptproof" / "rubrics"


def test_every_module_uses_one_definition_of_each_sentinel():
    assert JUDGE_FROM_JUDGE is JUDGE_SENTINEL is JUDGE_FROM_STUB
    assert REWRITE_FROM_REWRITE is REWRITE_SENTINEL is REWRITE_FROM_STUB


def test_sentinels_match_the_rubric_files_that_carry_them():
    assert (RUBRICS / "judge_system.md").read_text(encoding="utf-8").startswith(JUDGE_SENTINEL)
    assert (RUBRICS / "rewrite_system.md").read_text(encoding="utf-8").startswith(REWRITE_SENTINEL)


def test_the_two_sentinels_are_distinct():
    assert JUDGE_SENTINEL != REWRITE_SENTINEL
    assert JUDGE_SENTINEL not in REWRITE_SENTINEL
    assert REWRITE_SENTINEL not in JUDGE_SENTINEL


def test_judge_defaults_are_named_not_scattered():
    assert DEFAULT_JUDGE_MODEL == "claude-opus-5"
    assert JUDGE_MODEL_ENV_VAR == "PROMPTPROOF_JUDGE_MODEL"
