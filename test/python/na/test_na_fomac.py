# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test NADevice class creation."""

from __future__ import annotations

from difflib import unified_diff
from json import dumps, load
from pathlib import Path

from mqt.core.na.fomac import NADevice
from mqt.core.qdmi.fomac import devices


def test_from_device() -> None:
    """Test NADevice creation from device."""
    dev = NADevice(devices()[0])
    print(f"dev = {dev}")
    with Path("json/na/device.json").open(encoding="utf-8") as f:
        dev2 = load(f)
    # remove all operations to avoid diff due to unimplemented functionality
    dev2 = {k: v for k, v in dev2.items() if "operation" not in k.lower() and "shuttling" not in k.lower()}
    d1_lines = dumps(dev.dict(), indent=2, sort_keys=True).splitlines()
    d2_lines = dumps(dev2, indent=2, sort_keys=True).splitlines()
    diff_lines = list(unified_diff(d1_lines, d2_lines, fromfile="dict1", tofile="dict2", lineterm=""))
    delimiter = "\n"
    assert not diff_lines, (
        f"The generated device dictionary does not match the expected one.\nDiff:\n{delimiter.join(diff_lines)}"
    )
