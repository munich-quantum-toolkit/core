# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for deriving native gate menus from device operation names."""

from __future__ import annotations

import pytest
from plugins.qiskit.test_mock_backend import MockQDMIDevice

from mqt.core.mlir import (
    native_gates_from_device,
    native_gates_from_operation_names,
)


@pytest.fixture
def mock_qdmi_device_factory() -> type[MockQDMIDevice]:
    """Return the mock QDMI device class for parameterized device tests."""
    return MockQDMIDevice


def test_native_gates_from_operation_names_ibm_like() -> None:
    """Map an IBM-like op list to an x/sx/rz/cx menu."""
    assert native_gates_from_operation_names(["x", "sx", "rz", "cx", "h", "measure"]) == "x,sx,rz,cx"


def test_native_gates_from_operation_names_iqm_prx() -> None:
    """Alias prx to r and prefer cz."""
    assert native_gates_from_operation_names(["prx", "cz"]) == "r,cz"


def test_native_gates_from_operation_names_rejects_insufficient() -> None:
    """Reject name lists that lack a supported Euler + entangler pair."""
    with pytest.raises(ValueError, match="native-gates"):
        native_gates_from_operation_names(["h", "measure"])


def test_native_gates_from_device_ibm_like(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Derive an IBM-like menu from a FoMaC-style device."""
    device = mock_qdmi_device_factory(operations=["x", "sx", "rz", "cx", "h", "measure"])
    assert native_gates_from_device(device) == "x,sx,rz,cx"


def test_native_gates_from_device_iqm_prx(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Derive an IQM-like prx/cz menu from a FoMaC-style device."""
    device = mock_qdmi_device_factory(operations=["prx", "cz"])
    assert native_gates_from_device(device) == "r,cz"


def test_native_gates_from_device_rejects_insufficient(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Raise when a device exposes no supported native menu."""
    device = mock_qdmi_device_factory(operations=["h", "measure"])
    with pytest.raises(ValueError, match="native-gates"):
        native_gates_from_device(device)
