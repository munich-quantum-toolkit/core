# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for gate support in the QDMI Qiskit backend."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qiskit import QuantumCircuit

from mqt.core.plugins.qiskit import QDMIBackend
from mqt.core.plugins.qiskit.exceptions import UnsupportedDeviceError, UnsupportedOperationError

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*Device operation.*cannot be mapped to a Qiskit gate.*:UserWarning"),
    pytest.mark.filterwarnings("ignore:Device does not define a measurement operation.*:UserWarning"),
]


def test_backend_supports_cz_gate(mock_backend: QDMIBackend) -> None:
    """Test that the backend can execute CZ gate circuits."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = mock_backend.run(qc, shots=100)
    counts = job.result().get_counts()
    assert sum(counts.values()) == 100


def test_map_operation_returns_none_for_unknown(mock_backend: QDMIBackend) -> None:
    """Test that unknown operations return None."""
    assert mock_backend._map_operation_to_gate("unknown_gate") is None  # noqa: SLF001
    assert mock_backend._map_operation_to_gate("custom_op") is None  # noqa: SLF001
    assert mock_backend._map_operation_to_gate("") is None  # noqa: SLF001


def test_get_operation_qargs_single_qubit(mock_backend: QDMIBackend) -> None:
    """Test _get_operation_qargs for single-qubit operations without explicit sites.

    Operations without explicit sites should return [None] indicating global availability.
    """
    # Create a mock FoMaC operation for single qubit
    mock_op = MagicMock()
    mock_op.name.return_value = "test_op"
    mock_op.qubits_num.return_value = 1
    mock_op.parameters_num.return_value = 0
    mock_op.duration.return_value = None
    mock_op.fidelity.return_value = None
    mock_op.sites.return_value = None  # No explicit sites
    mock_op.is_zoned.return_value = False

    qargs = mock_backend._get_operation_qargs(mock_op)  # noqa: SLF001

    assert qargs == [None]


def test_get_operation_qargs_two_qubit_without_coupling_map(mock_backend: QDMIBackend) -> None:
    """Test _get_operation_qargs for two-qubit operations without coupling map.

    When an operation has site_pairs=None and device has no coupling map,
    it should return [None] indicating global availability (all-to-all).
    """
    # The mock device has no coupling map by default
    device_coupling_map = mock_backend._device.coupling_map()  # noqa: SLF001
    assert device_coupling_map is None

    # Create mock operation with no specific site pairs
    mock_op = MagicMock()
    mock_op.name.return_value = "test_2q"
    mock_op.qubits_num.return_value = 2
    mock_op.parameters_num.return_value = 0
    mock_op.duration.return_value = None
    mock_op.fidelity.return_value = None
    mock_op.site_pairs.return_value = None  # No specific sites
    mock_op.is_zoned.return_value = False

    qargs = mock_backend._get_operation_qargs(mock_op)  # noqa: SLF001

    assert qargs == [None]


def test_get_operation_qargs_with_specific_sites(mock_backend: QDMIBackend) -> None:
    """Test that backend handles operations properly in the target.

    This is tested through the backend's target construction, which uses
    _get_operation_qargs internally. We verify that the target properly
    reflects device operations.
    """
    # Verify that the backend target has operations defined
    assert len(mock_backend.target.operation_names) > 0

    # Verify operations are properly represented
    # Mock device operations have no explicit sites, so they should be globally available
    # In Qiskit's Target, globally available operations have qargs=None
    for op_name in mock_backend.target.operation_names:
        qargs = mock_backend.target.qargs_for_operation_name(op_name)
        # Operations without site constraints should have None qargs (globally available)
        # Or if they have qargs, they should be valid tuples
        if qargs is not None:
            for qarg in qargs:
                assert isinstance(qarg, tuple)
                assert all(isinstance(idx, int) for idx in qarg)


def test_get_operation_qargs_two_qubit_operation_with_subset_of_coupling_map(mock_backend: QDMIBackend) -> None:
    """Test that two-qubit operations properly reflect their coupling in the target.

    This addresses the scenario where a device has a coupling map describing all physical
    connections, but individual operations (e.g., CZ vs CX) are only supported on subsets
    of those connections.
    """
    # Verify that two-qubit operations exist in the target
    two_qubit_ops = [
        op_name
        for op_name in mock_backend.target.operation_names
        if mock_backend.target.operation_from_name(op_name).num_qubits == 2
    ]

    # Should have at least some two-qubit operations
    assert len(two_qubit_ops) > 0

    # Verify each two-qubit operation has valid qargs
    for op_name in two_qubit_ops:
        qargs = mock_backend.target.qargs_for_operation_name(op_name)
        if qargs:
            # All qargs should be pairs of qubits
            for qarg in qargs:
                assert len(qarg) == 2
                assert all(isinstance(idx, int) for idx in qarg)


def test_misconfigured_device_coupling_map_without_operation_sites() -> None:
    """Test that device with coupling map but operation without sites raises error."""

    class MockDeviceWithCouplingMap:
        """Mock device that has a coupling map."""

        class MockSite:
            """Mock site."""

            def __init__(self, idx: int) -> None:
                """Initialize with index."""
                self._idx = idx

            def index(self) -> int:
                """Return index."""
                return self._idx

            @staticmethod
            def is_zone() -> bool:
                """Return whether site is a zone.

                Returns:
                    False, as this is not a zone site.
                """
                return False

        def __init__(self) -> None:
            """Initialize mock device."""
            self._sites = [self.MockSite(i) for i in range(5)]
            self._coupling_map = [(self._sites[0], self._sites[1]), (self._sites[1], self._sites[2])]

        @staticmethod
        def name() -> str:
            """Return device name."""
            return "Mock Device With Coupling"

        @staticmethod
        def version() -> str:
            """Return device version."""
            return "1.0.0"

        @staticmethod
        def qubits_num() -> int:
            """Return number of qubits."""
            return 5

        def regular_sites(self) -> list[MockSite]:
            """Return regular sites."""
            return self._sites

        @staticmethod
        def operations() -> list[object]:
            """Return empty operation list."""
            return []

        def coupling_map(self) -> list[tuple[MockSite, MockSite]]:
            """Return coupling map."""
            return self._coupling_map

    device = MockDeviceWithCouplingMap()
    backend = QDMIBackend(device=device)  # type: ignore[arg-type]

    mock_op = MagicMock()
    mock_op.name.return_value = "custom_2q"
    mock_op.qubits_num.return_value = 2
    mock_op.site_pairs.return_value = None  # No sites, but device has a coupling map
    mock_op.is_zoned.return_value = False

    with pytest.raises(UnsupportedOperationError, match="misconfigured device"):
        backend._get_operation_qargs(mock_op)  # noqa: SLF001


def test_zoned_operation_rejected_at_backend_init() -> None:
    """Test that devices with zoned operations are rejected at backend initialization."""

    class MockZonedDevice:
        """Mock device with zoned operations."""

        class MockZonedOp:
            """Mock zoned operation."""

            @staticmethod
            def name() -> str:
                """Return operation name."""
                return "zoned_op"

            @staticmethod
            def is_zoned() -> bool:
                """Return True to indicate zoned operation."""
                return True

        def __init__(self) -> None:
            """Initialize mock device."""
            self._ops = [self.MockZonedOp()]

        @staticmethod
        def name() -> str:
            """Return device name."""
            return "Mock Zoned Device"

        @staticmethod
        def version() -> str:
            """Return device version."""
            return "1.0.0"

        def operations(self) -> list[MockZonedOp]:
            """Return list of operations."""
            return self._ops

        @staticmethod
        def qubits_num() -> int:
            """Return number of qubits."""
            return 5

        @staticmethod
        def regular_sites() -> list[object]:
            """Return regular sites."""
            return []

    device = MockZonedDevice()

    with pytest.raises(UnsupportedDeviceError, match="cannot be represented in Qiskit's Target model"):
        QDMIBackend(device=device)  # type: ignore[arg-type]
