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


def test_runner_accepts_scaled_nodes() -> None:
    """Run the same cluster with more compute nodes without a new Compose file."""
    assert runner.parse_arguments(("--nodes", "3")).nodes == 3


def test_prepare_keeps_private_key_and_preserves_existing_cluster(tmp_path: Path) -> None:
    """Create usable shared files without overwriting a cluster's authentication."""
    runtime = tmp_path / "cluster"
    command = ("sh", str(runner.CLUSTER / "prepare.sh"), str(runtime))
    runner.run(command)
    key = runtime / "munge.key"
    assert len(key.read_bytes()) == 1024
    assert key.stat().st_mode & 0o777 == 0o600
    assert (runtime / "slurm.conf").stat().st_mode & 0o777 == 0o644
    original = key.read_bytes()
    assert runner.run(command, check=False).returncode != 0
    assert key.read_bytes() == original


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


@pytest.mark.parametrize("record", ["FAILED|1:0", "COMPLETED|0:9"])
def test_failed_accounting_record_is_not_success(monkeypatch: pytest.MonkeyPatch, record: str) -> None:
    """A result file cannot hide a failed job or termination by signal."""
    monkeypatch.setattr(runner, "job", lambda *args: subprocess.CompletedProcess(args, 0, record, ""))
    with pytest.raises(AssertionError, match="Slurm job 42 ended"):
        runner.job_finished("42")


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
    (tmp_path / "mqt_core-0.whl").touch()
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


def test_provider_options_preserve_command_arguments(tmp_path: Path) -> None:
    """Keep workload arguments separate from fixture options."""
    setup = tmp_path / "setup.sh"
    setup.touch()
    options = runner.parse_arguments((
        "--workload",
        str(tmp_path),
        "--setup-script",
        "setup.sh",
        "--device-license",
        "provider.device",
        "--",
        "python3",
        "probe.py",
        "--label",
        "one argument",
    ))
    assert options.workload == tmp_path
    assert options.command == ["python3", "probe.py", "--label", "one argument"]


@pytest.mark.parametrize(
    "arguments",
    [
        ("--device-license", "provider.device"),
        ("--", "python3", "probe.py"),
        ("--device-license", "provider.device:2", "--", "/bin/true"),
    ],
)
def test_invalid_provider_inputs_fail_before_docker(arguments: tuple[str, ...]) -> None:
    """Reject incomplete or unrepresentable fixture inputs at the CLI."""
    with pytest.raises(SystemExit) as error:
        runner.parse_arguments(arguments)
    assert error.value.code == 2


def test_setup_script_must_stay_inside_workload(tmp_path: Path) -> None:
    """Do not accept a setup script that the workload build context cannot supply."""
    workload = tmp_path / "workload"
    workload.mkdir()
    (tmp_path / "outside.sh").touch()
    with pytest.raises(SystemExit) as error:
        runner.parse_arguments(("--workload", str(workload), "--setup-script", "../outside.sh"))
    assert error.value.code == 2


def test_result_written_during_accounting_query(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Read the shared output after accounting reports completion."""
    (tmp_path / "jobs").mkdir()
    monkeypatch.setattr(runner, "RUNTIME", tmp_path)

    def finished(_job_id: str) -> bool:
        (tmp_path / "jobs" / "sc-42.json").write_text("{}", encoding="utf-8")
        return True

    monkeypatch.setattr(runner, "job_finished", finished)
    runner.wait_for_result("sc", "42", "SC output")
