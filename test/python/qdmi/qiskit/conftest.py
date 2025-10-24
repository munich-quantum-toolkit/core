# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared fixtures for QDMI Qiskit backend tests."""

from __future__ import annotations

import importlib.util

import pytest

_qiskit_present = importlib.util.find_spec("qiskit") is not None

if _qiskit_present:
    from mqt.core import fomac
    from mqt.core.qdmi.qiskit import QiskitBackend


@pytest.fixture
def na_backend() -> QiskitBackend:
    """Fixture providing a QiskitBackend configured with the NA device.

    Returns:
        QiskitBackend instance configured with the MQT NA Default QDMI Device.

    Raises:
        RuntimeError: If the MQT NA Default QDMI Device is not found.

    Note:
        This fixture is used for tests that rely on specific NA device characteristics.
        These tests verify site handling, zone filtering, and QDMI specification compliance
        specific to the NA device structure (100 qubits + 3 zone sites).
        In the future, these tests should be generalized or parameterized across device types.
    """
    devices_list = list(fomac.devices())
    for idx, device in enumerate(devices_list):
        if device.name() == "MQT NA Default QDMI Device":
            return QiskitBackend(device_index=idx)
    msg = "MQT NA Default QDMI Device not found"
    raise RuntimeError(msg)


@pytest.fixture
def na_device() -> fomac.Device:
    """Get the NA FoMaC device for direct inspection.

    Returns:
        The MQT NA Default QDMI Device.

    Raises:
        RuntimeError: If the MQT NA Default QDMI Device is not found.
    """
    devices_list = list(fomac.devices())
    for device in devices_list:
        if device.name() == "MQT NA Default QDMI Device":
            return device
    msg = "MQT NA Default QDMI Device not found"
    raise RuntimeError(msg)
