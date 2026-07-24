# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared fixtures for installed-package QDMI integration tests."""

from __future__ import annotations

import pytest

from mqt.core import qdmi


def _open_isolated(device_id: str) -> qdmi.Device:
    discovery = qdmi.DeviceManager()
    definition = next(
        (candidate for candidate in discovery.definitions if candidate.device_id == device_id),
        None,
    )
    if definition is None:
        pytest.skip(f"QDMI device '{device_id}' is not installed")
    options = qdmi.ConfigOptions(isolated=True, runtime_overrides=[definition])
    return qdmi.DeviceManager(options).open(device_id)


@pytest.fixture(scope="session")
def ddsim_device() -> qdmi.Device:
    """Return the packaged DDSIM device through an isolated definition."""
    return _open_isolated("mqt.ddsim.default")


@pytest.fixture(scope="session")
def na_device() -> qdmi.Device:
    """Return the packaged neutral-atom device through an isolated definition."""
    return _open_isolated("mqt.na.default")
