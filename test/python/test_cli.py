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

# Import the private module to test its process replacement directly.
import mqt.core._bench as benchmark_cli  # ruff: ignore[import-private-name]
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
def test_benchmark_cli(script_runner: ScriptRunner) -> None:
    """Run the bundled benchmark driver through its console script."""
    ret = script_runner.run(["mqt-core-bench", "list"])
    assert ret.success
    assert '"ghz"' in ret.stdout
    assert '"grover"' in ret.stdout
    assert '"multiplexer"' in ret.stdout
    assert '"qpe"' in ret.stdout


@pytest.mark.parametrize(("platform", "suffix"), [("linux", ""), ("win32", ".exe")])
def test_benchmark_cli_launcher(platform: str, suffix: str) -> None:
    """Locate and execute the bundled benchmark driver on each platform."""
    executable = Path(f"installation/mqt/core/bin/mqt-core-bench{suffix}")
    with (
        patch.object(benchmark_cli.sys, "platform", platform),
        patch.object(benchmark_cli.sys, "argv", ["mqt-core-bench", "list"]),
        patch.object(benchmark_cli, "distribution") as distribution_mock,
        patch.object(benchmark_cli.os, "execv") as execv_mock,
    ):
        distribution_mock.return_value.locate_file.return_value = executable
        benchmark_cli.main()

    distribution_mock.assert_called_once_with("mqt-core")
    distribution_mock.return_value.locate_file.assert_called_once_with(f"mqt/core/bin/mqt-core-bench{suffix}")
    execv_mock.assert_called_once_with(executable, [str(executable), "list"])
