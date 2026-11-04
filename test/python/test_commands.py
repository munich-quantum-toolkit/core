# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Native command argument forwarding and exit status."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_native_command(monkeypatch: pytest.MonkeyPatch, platform: str) -> None:
    """Forward arguments intact and preserve a failing native tool's exit code."""
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(sys, "argv", ["mqt-core", "bench", "input with spaces.qasm", "--help"])
    execute = Mock(side_effect=SystemExit(7))
    run = Mock(return_value=Mock(returncode=7))
    monkeypatch.setattr(os, "execv", execute)
    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(SystemExit) as error:
        distribution("mqt-core").entry_points["mqt-core"].load()()
    assert error.value.code == 7
    suffix = ".exe" if platform == "win32" else ""
    executable = Path(str(distribution("mqt-core").locate_file(f"mqt/core/bin/mqt-core{suffix}")))
    arguments = [str(executable), "bench", "input with spaces.qasm", "--help"]
    if platform == "win32":
        run.assert_called_once_with(arguments, check=False)
    else:
        execute.assert_called_once_with(executable, arguments)
