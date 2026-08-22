"""Full offline operation.

These tests assert the mandatory property of this project: the entire pipeline -
generate, run, judge, rewrite, report, CLI - completes with no API key and no
network. Sockets are disabled for the duration, so a stray outbound call fails
the test rather than silently working on a developer's machine and breaking in
CI.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from promptproof.cli import main

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_SPEC = ROOT / "examples" / "support_agent" / "spec.toml"
SAMPLE_REPORT = ROOT / "examples" / "support_agent" / "sample-report.md"
README = ROOT / "README.md"

#: The flags the committed sample report was generated with.
SAMPLE_ARGS = ["--cases-per-axis", "4"]


def _without_timestamp(markdown: str) -> str:
    """Drop the one line that legitimately differs between runs."""
    return "\n".join(line for line in markdown.splitlines() if not line.startswith("Provider "))


@pytest.fixture()
def no_network(monkeypatch):
    """Make any outbound socket connection raise."""

    def blocked(*args, **kwargs):
        raise AssertionError("offline mode attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)


@pytest.fixture()
def no_credentials(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


def test_full_run_offline(tmp_path, capsys, no_network, no_credentials):
    out = tmp_path / "out"
    code = main(
        [
            "run",
            "--spec",
            str(EXAMPLE_SPEC),
            "--out",
            str(out),
            "--quiet",
            "--cases-per-axis",
            "4",
        ]
    )
    assert code == 0

    markdown = (out / "report.md").read_text(encoding="utf-8")
    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))

    assert "# PromptProof report - retail-support-agent" in markdown
    assert payload["models"] == ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"]
    assert payload["case_count"] == 24
    assert len(payload["results"]) == 72
    assert not [r for r in payload["results"] if r["status"] == "error"]
    assert any(r["status"] == "fail" for r in payload["results"])


def test_run_prints_the_report_when_no_out_dir_given(capsys, no_network, no_credentials):
    assert main(["run", "--spec", str(EXAMPLE_SPEC), "--quiet", "--cases-per-axis", "1"]) == 0
    assert "# PromptProof report" in capsys.readouterr().out


def test_run_reports_progress_on_stderr(capsys, no_network, no_credentials):
    argv = ["run", "--spec", str(EXAMPLE_SPEC), "--cases-per-axis", "1", "--model", "claude-opus-5"]
    assert main(argv) == 0
    err = capsys.readouterr().err
    assert "[ok  ]" in err
    assert "claude-opus-5 ::" in err


def test_fail_under_gates_the_exit_code(tmp_path, capsys, no_network, no_credentials):
    args = [
        "run",
        "--spec",
        str(EXAMPLE_SPEC),
        "--quiet",
        "--cases-per-axis",
        "4",
        "--out",
        str(tmp_path / "o"),
    ]
    assert main([*args, "--fail-under", "0.99"]) == 1
    assert "below --fail-under" in capsys.readouterr().err
    assert main([*args, "--fail-under", "0.1"]) == 0


def test_single_frontier_model_run_is_clean(tmp_path, no_network, no_credentials):
    code = main(
        [
            "run",
            "--spec",
            str(EXAMPLE_SPEC),
            "--quiet",
            "--model",
            "claude-opus-5",
            "--cases-per-axis",
            "4",
            "--out",
            str(tmp_path / "o"),
            "--fail-under",
            "1.0",
        ]
    )
    assert code == 0


def test_generate_subcommand_offline(capsys, no_network, no_credentials):
    assert main(["generate", "--spec", str(EXAMPLE_SPEC), "--cases-per-axis", "2"]) == 0
    out = capsys.readouterr().out
    assert "## multi_turn_drift" in out
    assert "[violation 1.00]" in out


def test_generate_json_is_machine_readable(capsys, no_network, no_credentials):
    assert main(["generate", "--spec", str(EXAMPLE_SPEC), "--cases-per-axis", "2", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 12
    drift = [c for c in payload if c["axis"] == "multi_turn_drift"]
    assert drift and all(len(c["turns"]) >= 4 for c in drift)


def test_axes_subcommand(capsys, no_network):
    assert main(["axes"]) == 0
    out = capsys.readouterr().out
    assert "multi_turn_drift" in out
    assert "false_precedent" in out


def test_bad_spec_path_exits_two(capsys, no_network):
    assert main(["run", "--spec", "/nonexistent/spec.toml"]) == 2
    assert "error:" in capsys.readouterr().err


def test_unknown_provider_exits_two(capsys, no_network):
    with pytest.raises(SystemExit):
        main(["run", "--spec", str(EXAMPLE_SPEC), "--provider", "openai"])


def test_offline_run_is_reproducible(tmp_path, no_network, no_credentials):
    """Two identical invocations must differ only in the timestamp."""
    reports = []
    for name in ("a", "b"):
        out = tmp_path / name
        argv = ["run", "--spec", str(EXAMPLE_SPEC), "--quiet", "--cases-per-axis", "3"]
        assert main([*argv, "--out", str(out)]) == 0
        payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
        payload.pop("generated_at")
        reports.append(payload)
    assert reports[0] == reports[1]


def test_offline_path_never_imports_the_anthropic_sdk():
    """The SDK is an optional extra; nothing on the offline path may import it.

    Run in a fresh interpreter, because another test in this suite imports the
    SDK deliberately and would otherwise mask a real regression here.
    """
    import subprocess
    import sys
    import textwrap

    program = textwrap.dedent(
        """
        import sys
        from promptproof.cli import main
        main(["generate", "--spec", sys.argv[1], "--cases-per-axis", "1", "--json"])
        assert "anthropic" not in sys.modules, "offline path imported the anthropic SDK"
        print("clean")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", program, str(EXAMPLE_SPEC)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().endswith("clean")


def test_judge_model_can_be_set_by_environment(tmp_path, monkeypatch, no_network, no_credentials):
    """The variable documented in .env.example has to actually do something."""
    monkeypatch.setenv("PROMPTPROOF_JUDGE_MODEL", "claude-sonnet-5")
    out = tmp_path / "env"
    argv = ["run", "--spec", str(EXAMPLE_SPEC), "--quiet", "--cases-per-axis", "1", "--out", str(out)]
    assert main(argv) == 0
    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert payload["judge_model"] == "claude-sonnet-5"

    # an explicit flag still wins over the environment
    out2 = tmp_path / "flag"
    assert main([*argv[:-1], str(out2), "--judge-model", "claude-opus-5"]) == 0
    payload2 = json.loads((out2 / "report.json").read_text(encoding="utf-8"))
    assert payload2["judge_model"] == "claude-opus-5"


def test_judge_model_falls_back_to_the_default_when_unset(tmp_path, monkeypatch, no_network):
    monkeypatch.delenv("PROMPTPROOF_JUDGE_MODEL", raising=False)
    out = tmp_path / "default"
    argv = ["run", "--spec", str(EXAMPLE_SPEC), "--quiet", "--cases-per-axis", "1", "--out", str(out)]
    assert main(argv) == 0
    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert payload["judge_model"] == "claude-opus-5"


@pytest.mark.parametrize(
    "bad",
    [
        ["--cases-per-axis", "0"],
        ["--concurrency", "0"],
        ["--max-tokens", "-5"],
        ["--fail-under", "1.5"],
        ["--timeout", "0"],
    ],
)
def test_nonsense_arguments_are_rejected_before_any_work_happens(bad, no_network):
    """argparse exits 2, which is the documented 'configuration problem' code."""
    with pytest.raises(SystemExit) as info:
        main(["run", "--spec", str(EXAMPLE_SPEC), *bad])
    assert info.value.code == 2


def test_json_report_carries_timing_and_token_totals(tmp_path, no_network, no_credentials):
    """Numbers the runner already collects, made available to a cost estimate."""
    out = tmp_path / "totals"
    argv = ["run", "--spec", str(EXAMPLE_SPEC), "--quiet", "--cases-per-axis", "2", "--out", str(out)]
    assert main(argv) == 0
    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))

    totals = payload["totals"]
    assert totals["input_tokens"] == sum(r["input_tokens"] for r in payload["results"])
    assert totals["output_tokens"] == sum(r["output_tokens"] for r in payload["results"])
    assert totals["output_tokens"] > 0
    assert all(r["duration_ms"] >= 0 for r in payload["results"])


def test_generate_json_exposes_case_metadata(capsys, no_network):
    assert main(["generate", "--spec", str(EXAMPLE_SPEC), "--cases-per-axis", "2", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    drift = [c for c in payload if c["axis"] == "multi_turn_drift"]
    assert drift
    for case in drift:
        assert int(case["metadata"]["turns"]) == len(case["turns"])
        assert case["metadata"]["severity"] in ("hard", "soft")


def test_the_committed_sample_report_is_what_the_code_still_produces(tmp_path, no_network):
    """A stale example is a lie about the tool, and it rots silently.

    Regenerating it here means the numbers quoted in the README cannot drift
    away from the harness without a test going red.
    """
    out = tmp_path / "regen"
    argv = ["run", "--spec", str(EXAMPLE_SPEC), "--quiet", "--out", str(out), *SAMPLE_ARGS]
    assert main(argv) == 0

    regenerated = _without_timestamp((out / "report.md").read_text(encoding="utf-8"))
    committed = _without_timestamp(SAMPLE_REPORT.read_text(encoding="utf-8"))
    assert regenerated == committed, (
        "examples/support_agent/sample-report.md is out of date; regenerate it with "
        f"promptproof run --spec examples/support_agent/spec.toml {' '.join(SAMPLE_ARGS)} --out out/"
    )


def test_the_readme_quotes_the_sample_report_verbatim():
    """The one table a 60-second reader looks at has to be the real one."""
    readme = README.read_text(encoding="utf-8")
    sample = SAMPLE_REPORT.read_text(encoding="utf-8")
    rows = [line for line in sample.splitlines() if line.startswith("| `claude-")]
    assert len(rows) == 3
    for row in rows:
        assert row in readme, f"README summary table is out of sync with the sample report: {row}"
