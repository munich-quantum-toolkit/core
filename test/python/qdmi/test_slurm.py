# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the mechanism-specific Slurm adapter."""

from __future__ import annotations

import pytest

from mqt.core.qdmi import slurm


def test_open_device_from_license(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open a fresh registered DDSIM device from a unit license."""
    monkeypatch.setenv("SLURM_JOB_LICENSES", "mqt.ddsim.default:1")
    device = slurm.open_device_from_license()
    assert device.name() == "MQT Core DDSIM QDMI Device"


@pytest.mark.parametrize(
    "value",
    [
        "mqt.ddsim.default:2",
        "mqt.ddsim.default,mqt.sc.default",
        "mqt.ddsim.default|mqt.sc.default",
    ],
)
def test_reject_non_unit_and_compound_licenses(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Reject license requests that cannot select one QDMI device."""
    monkeypatch.setenv("SLURM_JOB_LICENSES", value)
    with pytest.raises(RuntimeError):
        slurm.open_device_from_license()
