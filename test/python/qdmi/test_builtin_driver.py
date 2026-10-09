# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test MQT Core's optional packaged-driver extension."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

import pytest

from mqt.core.qdmi import builtin_driver


@pytest.mark.parametrize("device", ["ddsim", "sc"])
def test_relocated_device_library(device: str, tmp_path: Path) -> None:
    """Load a device without the wheel's private C++ libraries or Python imports."""
    package = distribution("mqt-core")
    manifest_name = f"mqt-core-qdmi-{device}-device.qdmi.json"
    assert package.files is not None
    manifest = next(package.locate_file(file) for file in package.files if file.name == manifest_name)
    entry = json.loads(manifest.read_text())["qdmi"]["devices"][0]
    library = tmp_path / entry["library"]
    shutil.copy2(manifest.parent / entry["library"], library)
    script = """
import ctypes
import sys

library = ctypes.CDLL(sys.argv[1])
assert getattr(library, sys.argv[2] + "_QDMI_device_initialize")() == 0
assert getattr(library, sys.argv[2] + "_QDMI_device_finalize")() == 0
"""
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", script, str(library), entry["prefix"]],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_manifest_registration_and_offline_enumeration(tmp_path: Path) -> None:
    """List configured IDs even when their libraries cannot be loaded."""
    malformed = tmp_path / "malformed.qdmi.json"
    malformed.write_text("{")
    (tmp_path / "not-a-library").touch()
    manifest = tmp_path / "offline.qdmi.json"
    manifest.write_text(
        json.dumps({
            "schema-version": 1,
            "qdmi": {
                "devices": [
                    {"id": "test.offline", "library": "not-a-library", "prefix": "MISSING"},
                    {"id": "test.disabled", "enabled": False},
                ]
            },
        })
    )
    script = """
import sys
from pathlib import Path
from mqt.core.qdmi import builtin_driver

try:
    builtin_driver.add_manifest(Path(sys.argv[1]))
except RuntimeError as error:
    assert "Library not found" in str(error)
else:
    raise AssertionError("missing manifest must fail")

try:
    builtin_driver.add_manifest(Path(sys.argv[2]))
except ValueError as error:
    assert "Invalid argument" in str(error)
else:
    raise AssertionError("malformed manifest must fail")

builtin_driver.add_manifest(Path(sys.argv[3]))
ids = builtin_driver.registered_device_ids()
assert "test.offline" in ids
assert "test.disabled" not in ids
assert ids == builtin_driver.registered_device_ids()
"""
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", script, tmp_path / "missing.qdmi.json", malformed, manifest],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Skipping configured QDMI device" not in result.stderr


def test_open_device_uses_strict_fresh_sessions() -> None:
    """Targeted opens apply strict overrides and own independent sessions."""
    first = builtin_driver.open_device("mqt.ddsim.default")
    second = builtin_driver.open_device("mqt.ddsim.default")

    assert first.id == "mqt.ddsim.default"
    assert second.id == "mqt.ddsim.default"
    assert first != second

    with pytest.raises(RuntimeError, match="Not supported"):
        builtin_driver.open_device("mqt.ddsim.default", custom4="strict")


def test_open_device_rejects_conflicting_device_configuration() -> None:
    """The Python wrapper rejects two sources for one typed configuration."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        builtin_driver.open_device(
            "mqt.sc.default",
            device_config="{}",
            device_config_file="device.json",
        )


def test_sc_open_device_accepts_runtime_configuration(tmp_path: Path) -> None:
    """The built-in SC provider should materialize a per-open file model."""
    configuration = json.loads(Path("json/sc/mqt-core-qdmi-sc-device.json").read_text(encoding="utf-8"))
    configuration["name"] = "Python custom SC device"
    configuration["numQubits"] = 5
    configuration["couplings"] = [[0, 1], [1, 2], [2, 3], [3, 4]]
    configuration["qubitProperties"]["overrides"] = []
    for operation in configuration["operations"]:
        operation.pop("sites", None)
        operation["siteOverrides"] = []
    configuration_file = tmp_path / "sc-device.json"
    configuration_file.write_text(json.dumps(configuration), encoding="utf-8")

    device = builtin_driver.open_device(
        "mqt.sc.default",
        device_config_file=configuration_file,
    )
    assert device.name() == "Python custom SC device"
    assert device.qubits_num() == 5

    inline = builtin_driver.open_device("mqt.sc.default", device_config=json.dumps(configuration))
    assert inline.name() == device.name()
    assert inline.qubits_num() == device.qubits_num()
    with pytest.raises(ValueError, match="parse error"):
        builtin_driver.open_device("mqt.sc.default", device_config="{")
