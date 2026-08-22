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

EXAMPLE_SPEC = Path(__file__).resolve().parents[1] / "examples" / "support_agent" / "spec.toml"


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
