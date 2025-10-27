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
from typing import TYPE_CHECKING, NoReturn

import pytest

_qiskit_present = importlib.util.find_spec("qiskit") is not None

if TYPE_CHECKING or _qiskit_present:
    from mqt.core import fomac
    from mqt.core.qdmi.qiskit import QiskitBackend


def _get_na_device_index() -> int:
    """Find the index of the MQT NA Default QDMI Device.

    Returns:
        Index of the NA device in fomac.devices().
    """
    devices_list = list(fomac.devices())
    for idx, device in enumerate(devices_list):
        if device.name() == "MQT NA Default QDMI Device":
            return idx
    return _skip_na_device_not_found()


def _skip_na_device_not_found() -> NoReturn:
    """Skip the test when NA device is not found."""
    pytest.skip("MQT NA Default QDMI Device not found in environment")


@pytest.fixture
def na_backend() -> QiskitBackend:
    """Fixture providing a QiskitBackend configured with the NA device.

    Returns:
        QiskitBackend instance configured with the MQT NA Default QDMI Device.

    Note:
        This fixture is used for tests that rely on specific NA device characteristics.
        These tests verify site handling, zone filtering, and QDMI specification compliance
        specific to the NA device structure (100 qubits + 3 zone sites).
        In the future, these tests should be generalized or parameterized across device types.
    """
    return QiskitBackend(device_index=_get_na_device_index())


@pytest.fixture
def na_device() -> fomac.Device:
    """Fixture providing the NA FoMaC device for direct inspection.

    Returns:
        The MQT NA Default QDMI Device.

    Note:
        This fixture is used for tests that need direct access to the FoMaC device
        to inspect raw device capabilities, site information, and other low-level
        properties specific to the NA device structure (100 qubits + 3 zone sites).
        In the future, these tests should be generalized or parameterized across device types.
    """
    return list(fomac.devices())[_get_na_device_index()]
