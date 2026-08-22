"""Spec parsing and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from promptproof import SpecError, build_spec, load_spec

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "support_agent"


def test_loads_toml_example(example_spec):
    assert example_spec.name == "retail-support-agent"
    assert example_spec.persona.role.startswith("first-line support")
    assert "Never authorize" in example_spec.system_prompt
    assert {b.id for b in example_spec.hard_boundaries} == {
        "no-outcome-authorization",
        "no-claims-advice",
        "no-payment-data",
    }


def test_yaml_and_toml_specs_agree():
    yaml = pytest.importorskip("yaml")  # noqa: F841 - optional extra
    from_toml = load_spec(EXAMPLE_DIR / "spec.toml")
    from_yaml = load_spec(EXAMPLE_DIR / "spec.yaml")
    assert from_toml.instructions() == from_yaml.instructions()
    assert from_toml.system_prompt == from_yaml.system_prompt


def test_scope_instruction_is_synthesised(example_spec):
    scope = example_spec.scope_instruction
    assert scope.id == "scope:stay-in-scope"
    assert scope.expect_refusal is True
    assert "legal advice" in scope.text
    assert scope in example_spec.instructions()


def test_instruction_lookup_and_unknown_id(example_spec):
    assert example_spec.instruction("ticket-reference").severity == "hard"
    assert example_spec.instruction("device-not-unit").severity == "soft"
    with pytest.raises(SpecError, match="unknown instruction id"):
        example_spec.instruction("nope")


def test_missing_file_is_a_spec_error(tmp_path):
    with pytest.raises(SpecError, match="not found"):
        load_spec(tmp_path / "absent.toml")


def test_unsupported_extension(tmp_path):
    path = tmp_path / "spec.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(SpecError, match="unsupported spec format"):
        load_spec(path)


def test_malformed_toml_names_the_file(tmp_path):
    path = tmp_path / "spec.toml"
    path.write_text("name = 'x'\n[[bad\n", encoding="utf-8")
    with pytest.raises(SpecError, match="could not parse spec"):
        load_spec(path)


def test_requires_something_to_test(minimal_spec_mapping):
    minimal_spec_mapping["hard_boundaries"] = []
    minimal_spec_mapping["required_behaviors"] = []
    with pytest.raises(SpecError, match="at least one"):
        build_spec(minimal_spec_mapping)


def test_rejects_both_prompt_forms(minimal_spec_mapping, tmp_path):
    minimal_spec_mapping["system_prompt_file"] = str(tmp_path / "p.md")
    with pytest.raises(SpecError, match="exactly one"):
        build_spec(minimal_spec_mapping)


def test_rejects_duplicate_instruction_ids(minimal_spec_mapping):
    minimal_spec_mapping["required_behaviors"].append(
        {"id": "no-banana", "instruction": "Different rule, same id."}
    )
    with pytest.raises(SpecError, match="duplicate instruction id"):
        build_spec(minimal_spec_mapping)


def test_rejects_bad_severity(minimal_spec_mapping):
    minimal_spec_mapping["required_behaviors"][0]["kind"] = "medium"
    with pytest.raises(SpecError, match="'hard' or 'soft'"):
        build_spec(minimal_spec_mapping)


def test_rejects_string_where_list_expected(minimal_spec_mapping):
    minimal_spec_mapping["hard_boundaries"][0]["topics"] = "banana"
    with pytest.raises(SpecError, match="must be a list of strings"):
        build_spec(minimal_spec_mapping)


def test_system_prompt_file_is_resolved_relative_to_spec(tmp_path, minimal_spec_mapping):
    (tmp_path / "prompt.md").write_text("You are a test assistant.", encoding="utf-8")
    minimal_spec_mapping.pop("system_prompt")
    minimal_spec_mapping["system_prompt_file"] = "prompt.md"
    spec = build_spec(minimal_spec_mapping, base_dir=tmp_path)
    assert spec.system_prompt.startswith("You are a test assistant")
