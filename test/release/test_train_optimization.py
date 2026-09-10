# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Training preserves the selected binaries and excludes discovery profiles."""

# ruff: file-ignore[implicit-namespace-package]
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    import pytest


def test_training_finds_selected_package_commands(tmp_path: Path) -> None:
    """Training subprocesses find the virtual environment and staged wheel CLIs."""
    runner = Path(__file__).with_name("train_optimization.py")
    package = tmp_path / "mqt/core"
    core = SimpleNamespace(__file__=str(package / "__init__.py"))
    with (
        patch.dict(sys.modules, {"mqt": SimpleNamespace(core=core)}),
        patch.object(sys, "argv", [str(runner), "--tests", "--expected-root", str(tmp_path)]),
        patch("subprocess.run") as execute,
    ):
        runpy.run_path(str(runner), run_name="__main__")
    assert execute.call_count == 3
    for call in execute.call_args_list:
        assert call.kwargs["env"]["PATH"].split(os.pathsep)[:2] == [
            str(Path(sys.executable).parent),
            str(package / "bin"),
        ]


def test_discovery_profiles_stay_outside_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CTest discovery must not add initialization-only profiles to training."""
    raw = tmp_path / "training"
    monkeypatch.setenv("LLVM_PROFILE_FILE", str(raw / "%m-%p.profraw"))
    monkeypatch.setenv("MQT_PGO_PROFILE_DIR", str(raw))
    monkeypatch.setattr(sys, "argv", ["train_cpp_optimization.py", str(tmp_path)])
    runner = runpy.run_path(str(Path(__file__).with_name("train_cpp_optimization.py")))
    with patch("subprocess.check_output", return_value='{"tests": []}') as discovery, patch("subprocess.run"):
        runner["main"]()
    env = discovery.call_args.kwargs["env"]
    assert not Path(env["LLVM_PROFILE_FILE"]).is_relative_to(raw)
    assert not Path(env["MQT_PGO_PROFILE_DIR"]).is_relative_to(raw)
    assert os.environ["LLVM_PROFILE_FILE"] == str(raw / "%m-%p.profraw")
    assert os.environ["MQT_PGO_PROFILE_DIR"] == str(raw)
