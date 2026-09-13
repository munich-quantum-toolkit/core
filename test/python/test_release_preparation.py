# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Reject untrained SDK profiles before configuring a release wheel."""

from __future__ import annotations

import runpy
from pathlib import Path
from unittest.mock import patch

import pytest


def test_profile_requires_executed_requested_components(tmp_path: Path) -> None:
    """Core execution alone must not qualify combined PGO."""
    check = runpy.run_path(str(Path(__file__).parents[2] / "scripts/prepare_release.py"))["check_profile"]
    with (
        patch("subprocess.check_output", side_effect=["Function count: 1\n", "Block counts: [0, 0]\n"]),
        pytest.raises(RuntimeError, match="sdk profile has no executed counters"),
    ):
        check("llvm-profdata", tmp_path / "profile", sdk=True)
    with patch("subprocess.check_output", side_effect=["Block counts: [0, 7]\n", "Function count: 2\n"]):
        assert set(check("llvm-profdata", tmp_path / "profile", sdk=True)) == {"core", "sdk"}
    with (
        patch("subprocess.check_output", return_value="Function count: 0\n"),
        pytest.raises(RuntimeError, match="core profile has no executed counters"),
    ):
        check("llvm-profdata", tmp_path / "profile", sdk=False)
