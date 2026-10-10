# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the mqt-core CLI."""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from subprocess import check_output
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from mqt.core import __version__ as mqt_core_version

if TYPE_CHECKING:
    from pytest_console_scripts import ScriptRunner


def test_cli_no_arguments(script_runner: ScriptRunner) -> None:
    """Test running the CLI with no arguments."""
    ret = script_runner.run(["mqt-core-cli"])
    assert ret.success
    assert "mqt-core-cli" in ret.stdout
    assert "--version" in ret.stdout
    assert "--include_dir" in ret.stdout
    assert "--cmake_dir" in ret.stdout


def test_cli_help(script_runner: ScriptRunner) -> None:
    """Test running the CLI with the --help argument."""
    ret = script_runner.run(["mqt-core-cli", "--help"])
    assert ret.success
    assert "mqt-core-cli" in ret.stdout
    assert "--version" in ret.stdout
    assert "--include_dir" in ret.stdout
    assert "--cmake_dir" in ret.stdout


def test_cli_version(script_runner: ScriptRunner) -> None:
    """Test running the CLI with the --version argument."""
    ret = script_runner.run(["mqt-core-cli", "--version"])
    assert ret.success
    assert mqt_core_version in ret.stdout


def test_cli_include_dir(script_runner: ScriptRunner) -> None:
    """Test running the CLI with the --include_dir argument."""
    ret = script_runner.run(["mqt-core-cli", "--include_dir"])
    assert ret.success
    include_dir = Path(ret.stdout.strip())
    assert include_dir.exists()
    assert include_dir.is_dir()


def test_cli_cmake_dir(script_runner: ScriptRunner) -> None:
    """Test running the CLI with the --cmake_dir argument."""
    ret = script_runner.run(["mqt-core-cli", "--cmake_dir"])
    assert ret.success
    cmake_dir = Path(ret.stdout.strip())
    assert cmake_dir.exists()
    assert cmake_dir.is_dir()


def test_cli_include_dir_not_installed(script_runner: ScriptRunner) -> None:
    """Test running the CLI with the --include_dir argument, but mqt-core is not installed."""
    with patch("importlib.metadata.Distribution.from_name") as mock:
        mock.side_effect = PackageNotFoundError()
        ret = script_runner.run(["mqt-core-cli", "--include_dir"])
        assert not ret.success
        assert "mqt-core not installed, installation required to access the include files." in ret.stderr


def test_cli_cmake_dir_not_installed(script_runner: ScriptRunner) -> None:
    """Test running the CLI with the --cmake_dir argument, but mqt-core is not installed."""
    with patch("importlib.metadata.Distribution.from_name") as mock:
        mock.side_effect = PackageNotFoundError()
        ret = script_runner.run(["mqt-core-cli", "--cmake_dir"])
        assert not ret.success
        assert "mqt-core not installed, installation required to access the CMake files." in ret.stderr


def test_cli_include_dir_not_found(script_runner: ScriptRunner) -> None:
    """Test running the CLI with the --include_dir argument, but the include directory is not found."""
    with patch("importlib.metadata.Distribution.from_name") as mock:
        mock.return_value.locate_file.return_value = "dir-not-found"
        ret = script_runner.run(["mqt-core-cli", "--include_dir"])
        assert not ret.success
        assert "mqt-core include files not found." in ret.stderr


def test_cli_cmake_dir_not_found(script_runner: ScriptRunner) -> None:
    """Test running the CLI with the --cmake_dir argument, but the CMake directory is not found."""
    with patch("importlib.metadata.Distribution.from_name") as mock:
        mock.return_value.locate_file.return_value = "dir-not-found"
        ret = script_runner.run(["mqt-core-cli", "--cmake_dir"])
        assert not ret.success
        assert "mqt-core CMake files not found." in ret.stderr


@pytest.mark.skipif(sys.platform.startswith("win"), reason="The subprocess calls do not work properly on Windows.")
def test_cli_execute_module() -> None:
    """Test running the CLI by executing the mqt-core module."""
    output = check_output(["python", "-m", "mqt.core", "--version"])  # ruff:ignore[start-process-with-partial-path]
    assert mqt_core_version in output.decode()


@pytest.mark.script_launch_mode("subprocess")
def test_compiler_cli(script_runner: ScriptRunner, tmp_path: Path) -> None:
    """Compile OpenQASM with the compiler bundled in the wheel."""
    source = tmp_path / "bell.qasm"
    source.write_text('OPENQASM 3.0; include "stdgates.inc"; qubit[2] q; h q[0]; cx q[0], q[1];')
    ret = script_runner.run(["mqt-cc", str(source), "--emit=qco"])
    assert ret.success
    assert "qco.h" in ret.stdout


@pytest.mark.script_launch_mode("subprocess")
def test_benchmark_cli(script_runner: ScriptRunner) -> None:
    """Run the bundled benchmark driver through its console script."""
    ret = script_runner.run(["mqt-core-bench", "list"])
    assert ret.success
    assert '"modular-multiplier"' in ret.stdout
    assert '"ghz"' in ret.stdout
    assert '"grover"' in ret.stdout
    assert '"multiplexer"' in ret.stdout
    assert '"qft-adder"' in ret.stdout
    assert '"qpe"' in ret.stdout
    assert '"repeat-until-success"' in ret.stdout
    assert '"teleportation"' in ret.stdout


@pytest.mark.parametrize(
    "tool",
    [
        "mqt-cc",
        "mqt-core-bench",
        "mqt-core-qdmi-check",
    ],
)
@pytest.mark.script_launch_mode("subprocess")
def test_native_tool_entry_point(script_runner: ScriptRunner, tool: str) -> None:
    """Report help and propagate native argument errors through the installed command."""
    ret = script_runner.run([tool, "--help"])
    assert ret.success
    assert ret.stdout
    assert not script_runner.run([tool, "--mqt-invalid-test-option"]).success


@pytest.mark.script_launch_mode("subprocess")
def test_qdmi_availability(script_runner: ScriptRunner) -> None:
    """Probe a device through the installed command and catalogue."""
    result = script_runner.run(["mqt-core-qdmi-check", "--device", "mqt.sc.default"])
    assert result.success
    assert not result.stdout
