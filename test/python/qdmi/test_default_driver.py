# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test MQT Core's optional packaged-driver extension."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from mqt.core.qdmi import default_driver

if TYPE_CHECKING:
    from pathlib import Path


def test_add_manifest_reports_invalid_files(tmp_path: Path) -> None:
    """Explicit manifest staging reports errors instead of warning and skipping."""
    malformed = tmp_path / "malformed.qdmi.json"
    malformed.write_text("{")
    script = """
import sys
from pathlib import Path
from mqt.core.qdmi import default_driver

try:
    default_driver.add_manifest(Path(sys.argv[1]))
except RuntimeError as error:
    assert "Library not found" in str(error)
else:
    raise AssertionError("missing manifest must fail")

try:
    default_driver.add_manifest(Path(sys.argv[2]))
except ValueError as error:
    assert "Invalid argument" in str(error)
else:
    raise AssertionError("malformed manifest must fail")
"""
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", script, tmp_path / "missing.qdmi.json", malformed],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_open_device_uses_strict_fresh_sessions() -> None:
    """Targeted opens apply strict overrides and own independent sessions."""
    first = default_driver.open_device("mqt.ddsim.default")
    second = default_driver.open_device("mqt.ddsim.default")

    assert first.id == "mqt.ddsim.default"
    assert second.id == "mqt.ddsim.default"
    assert first != second

    with pytest.raises(RuntimeError, match="Not supported"):
        default_driver.open_device("mqt.ddsim.default", custom4="strict")


def test_open_device_rejects_conflicting_device_configuration() -> None:
    """The Python wrapper rejects two sources for one typed configuration."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        default_driver.open_device(
            "mqt.sc.default",
            device_config="{}",
            device_config_file="device.json",
        )
