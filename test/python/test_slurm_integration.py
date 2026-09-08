# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Check failure and resource bounds without starting a Slurm cluster."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

if sys.platform == "win32":
    pytest.skip("the Slurm fixture requires a POSIX Docker host", allow_module_level=True)


def load_runner() -> ModuleType:
    """Load an independent fixture invocation without creating its resources.

    Returns:
        The isolated runner module.
    """
    script = Path(__file__).parents[1] / "slurm" / "run_integration.py"
    spec = importlib.util.spec_from_file_location("slurm_integration", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def test_timeout_kills_children_holding_output_pipes(tmp_path: Path) -> None:
    """A Compose-like grandchild must not defeat the command deadline."""
    marker = tmp_path / "started"
    program = (
        "import subprocess, sys, time; from pathlib import Path; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        f"Path({str(marker)!r}).touch(); time.sleep(30)"
    )
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        runner.run((sys.executable, "-c", program), timeout=1)
    assert marker.exists()
    assert time.monotonic() - started < 5


def test_diagnostic_timeout_is_best_effort() -> None:
    """Bound failed diagnostics without replacing the original test failure."""
    result = runner.run((sys.executable, "-c", "import time; time.sleep(30)"), check=False, timeout=0.1)
    assert result.returncode == 124


def test_command_failure_keeps_output() -> None:
    """Retain command diagnostics when a checked command fails."""
    with pytest.raises(subprocess.CalledProcessError) as error:
        runner.run((sys.executable, "-c", "import sys; print('reason'); sys.exit(7)"))
    assert error.value.returncode == 7
    assert "reason" in error.value.output


@pytest.mark.parametrize(
    ("state", "exit_code", "expected", "result"),
    [
        ("RUNNING", "0:0", "COMPLETED", False),
        ("COMPLETING", "0:0", "FAILED", False),
        ("COMPLETED", "0:0", "COMPLETED", True),
        ("FAILED", "1:0", "FAILED", True),
        ("FAILED", "1:0", "COMPLETED", None),
        ("COMPLETED", "0:0", "FAILED", None),
        ("COMPLETED", "0:9", "COMPLETED", None),
        ("FAILED", "0:0", "FAILED", None),
    ],
)
def test_job_completion_requires_state_and_exit_code(
    monkeypatch: pytest.MonkeyPatch, state: str, exit_code: str, expected: str, *, result: bool | None
) -> None:
    """A result file or a diagnostic cannot turn a failed job into a success."""
    record = f"JobId=42 JobState={state} ExitCode={exit_code}"
    monkeypatch.setattr(runner, "controller", lambda *args: subprocess.CompletedProcess(args, 0, record, ""))
    if result is None:
        with pytest.raises(AssertionError, match="Slurm job 42 ended"):
            runner.job_finished("42", expected_state=expected)
    else:
        assert runner.job_finished("42", expected_state=expected) is result


def test_preflight_does_not_touch_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing wheel must fail before starting Docker or writing a Munge key."""
    runtime = tmp_path / "runtime"
    monkeypatch.setattr(runner, "DIST", tmp_path)
    monkeypatch.setattr(runner, "RUNTIME", runtime)
    with pytest.raises(RuntimeError, match="exactly one MQT Core wheel"):
        runner.main()
    assert not runtime.exists()


def test_diagnostic_failure_still_tears_down(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unavailable diagnostic service must not orphan a failed cluster."""
    (tmp_path / "core.whl").touch()
    monkeypatch.setattr(runner, "DIST", tmp_path)
    monkeypatch.setattr(runner, "RUNTIME", tmp_path / "runtime")
    monkeypatch.setattr(runner, "run", lambda *args: subprocess.CompletedProcess(args, 0, "2", ""))
    commands = []

    def compose(*args: str, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        if args[0] == "up":
            msg = "startup failure"
            raise RuntimeError(msg)
        return subprocess.CompletedProcess(args, 0, "", "")

    def diagnostics() -> None:
        msg = "diagnostics"
        raise subprocess.TimeoutExpired(msg, 30)

    monkeypatch.setattr(runner, "compose", compose)
    monkeypatch.setattr(runner, "print_diagnostics", diagnostics)
    with pytest.raises(RuntimeError, match="startup failure"):
        runner.main()
    assert commands[-1][0] == "down"


def test_invocations_use_distinct_projects_and_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrent worktrees or runs must not stop or clean each other's cluster."""
    first, second = load_runner(), load_runner()
    calls = []

    def run(command: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs["env"]))
        return subprocess.CompletedProcess(command, 0, "", "")

    for invocation in (first, second):
        monkeypatch.setattr(invocation, "run", run)
        invocation.compose("ps")
    assert first.RUNTIME != second.RUNTIME
    assert calls[0][0] != calls[1][0]
    assert calls[0][1] != calls[1][1]
