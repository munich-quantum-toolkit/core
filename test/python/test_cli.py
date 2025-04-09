# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
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
    assert "usage: mqt-core-cli [-h] [--version] [--include_dir] [--cmake_dir]" in ret.stdout


def test_cli_help(script_runner: ScriptRunner) -> None:
    """Test running the CLI with the --help argument."""
    ret = script_runner.run(["mqt-core-cli", "--help"])
    assert ret.success
    assert "usage: mqt-core-cli [-h] [--version] [--include_dir] [--cmake_dir]" in ret.stdout


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
    from subprocess import check_output

    output = check_output(["python", "-m", "mqt.core", "--version"])  # noqa: S607
    assert mqt_core_version in output.decode()
