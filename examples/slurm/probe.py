#!/opt/venv/bin/python
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Return success only when the selected QDMI device is operational."""

from __future__ import annotations

import sys


def probe(device_id: str) -> int:
    """Return zero for IDLE or BUSY, or one when initialization or status fails."""
    try:
        from mqt.core.qdmi import Device
        from mqt.core.qdmi import builtin_driver as qdmi_driver

        device = qdmi_driver.open_device(device_id)
        return int(device.status() not in {Device.Status.IDLE, Device.Status.BUSY})
    except Exception:  # ruff: ignore[blind-except]
        # Device diagnostics may contain credentials.
        return 1


if __name__ == "__main__":
    sys.exit(probe(sys.argv[1]) if len(sys.argv) == 2 else 1)
