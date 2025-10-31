# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for gate support in the QDMI Qiskit backend."""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit

from mqt.core.qdmi.qiskit import QiskitBackend

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*Device operation.*cannot be mapped to a Qiskit gate.*:UserWarning"),
    pytest.mark.filterwarnings("ignore:Device does not define a measurement operation.*:UserWarning"),
]


def test_backend_supports_cz_gate(na_backend: QiskitBackend) -> None:
    """Test that the backend can execute CZ gate circuits (supported by NA device)."""
    # The NA device supports CZ
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = na_backend.run(qc, shots=100)
    counts = job.result().get_counts()
    assert sum(counts.values()) == 100


def test_map_operation_returns_none_for_unknown() -> None:
    """Test that unknown operations return None."""
    backend = QiskitBackend()

    assert backend._map_operation_to_gate("unknown_gate") is None  # noqa: SLF001
    assert backend._map_operation_to_gate("custom_op") is None  # noqa: SLF001
    assert backend._map_operation_to_gate("") is None  # noqa: SLF001


def test_backend_target_includes_measure() -> None:
    """Test that the backend target includes measure operation if device provides it."""
    backend = QiskitBackend()

    # Check if measure operation exists in device operations
    has_measure = any(op.name() == "measure" for op in backend._device.operations())  # noqa: SLF001
    if has_measure:
        assert "measure" in backend.target.operation_names


def test_get_operation_qargs_single_qubit() -> None:
    """Test _get_operation_qargs for single-qubit operations."""
    from unittest.mock import MagicMock

    backend = QiskitBackend()

    # Create a mock FoMaC operation for single qubit
    mock_op = MagicMock()
    mock_op.name.return_value = "test_op"
    mock_op.qubits_num.return_value = 1
    mock_op.parameters_num.return_value = 0
    mock_op.duration.return_value = None
    mock_op.fidelity.return_value = None
    mock_op.sites.return_value = None
    mock_op.is_zoned.return_value = False

    qargs = backend._get_operation_qargs(mock_op)  # noqa: SLF001

    # Should return all single qubit indices
    assert len(qargs) == backend._device.qubits_num()  # noqa: SLF001
    assert all(len(qarg) == 1 for qarg in qargs)


def test_get_operation_qargs_two_qubit_with_coupling_map_fallback() -> None:
    """Test _get_operation_qargs for two-qubit operations with coupling map fallback.

    When an operation has sites=None, it should fall back to using the device's
    full coupling map (all physical connections).
    """
    from unittest.mock import MagicMock

    backend = QiskitBackend()

    # If device has coupling map, it should use it as fallback when sites=None
    device_coupling_map = backend._device.coupling_map()  # noqa: SLF001
    if device_coupling_map:
        # Create mock operation with no specific sites
        mock_op = MagicMock()
        mock_op.name.return_value = "test_2q"
        mock_op.qubits_num.return_value = 2
        mock_op.parameters_num.return_value = 0
        mock_op.duration.return_value = None
        mock_op.fidelity.return_value = None
        mock_op.sites.return_value = None  # No specific sites - should use device coupling map
        mock_op.is_zoned.return_value = False

        qargs = backend._get_operation_qargs(mock_op)  # noqa: SLF001

        # Should match the full device coupling map (remapped to logical qubit indices)
        assert all(len(qarg) == 2 for qarg in qargs)
        assert len(qargs) > 0


def test_get_operation_qargs_with_specific_sites() -> None:
    """Test that backend handles operations with specific sites.

    This is tested through the backend's target construction, which uses
    _get_operation_qargs internally. We verify that the target properly
    reflects device operations.
    """
    backend = QiskitBackend()

    # Verify that the backend target has operations defined
    assert len(backend.target.operation_names) > 0

    # Verify operations have appropriate qargs
    for op_name in backend.target.operation_names:
        qargs = backend.target.qargs_for_operation_name(op_name)
        assert qargs is not None
        # Each qarg should be a tuple of qubit indices
        if qargs:
            for qarg in qargs:
                assert isinstance(qarg, tuple)
                assert all(isinstance(idx, int) for idx in qarg)


def test_get_operation_qargs_two_qubit_operation_with_subset_of_coupling_map() -> None:
    """Test that two-qubit operations properly reflect their coupling in the target.

    This addresses the scenario where a device has a coupling map describing all physical
    connections, but individual operations (e.g., CZ vs CX) are only supported on subsets
    of those connections.
    """
    backend = QiskitBackend()

    # Verify that two-qubit operations exist in the target
    two_qubit_ops = [
        op_name
        for op_name in backend.target.operation_names
        if backend.target.operation_from_name(op_name).num_qubits == 2
    ]

    # Should have at least some two-qubit operations
    assert len(two_qubit_ops) > 0

    # Verify each two-qubit operation has valid qargs
    for op_name in two_qubit_ops:
        qargs = backend.target.qargs_for_operation_name(op_name)
        if qargs:
            # All qargs should be pairs of qubits
            for qarg in qargs:
                assert len(qarg) == 2
                assert all(isinstance(idx, int) for idx in qarg)


def test_get_operation_qargs_multi_qubit_generates_all_combinations() -> None:
    """Test that multi-qubit operations (3+ qubits) generate all possible combinations.

    This verifies that the fallback for multi-qubit operations properly advertises
    the full capability by generating all combinations of qubits, not just the first
    contiguous qubits (e.g., for a 3-qubit operation on a 5-qubit device, it should
    generate all C(5,3) = 10 combinations, not just (0,1,2)).
    """
    from unittest.mock import MagicMock

    backend = QiskitBackend()

    # 3-qubit operation without specific sites - should generate all combinations
    mock_op = MagicMock()
    mock_op.name.return_value = "ccx"  # Toffoli gate as example
    mock_op.qubits_num.return_value = 3
    mock_op.parameters_num.return_value = 0
    mock_op.duration.return_value = None
    mock_op.fidelity.return_value = None
    mock_op.sites.return_value = None  # No specific sites - should generate all combinations
    mock_op.is_zoned.return_value = False

    qargs = backend._get_operation_qargs(mock_op)  # noqa: SLF001

    # Should generate all 3-qubit combinations
    num_qubits = backend._device.qubits_num()  # noqa: SLF001
    expected_count = 0
    if num_qubits >= 3:
        # Calculate C(num_qubits, 3)
        from math import comb

        expected_count = comb(num_qubits, 3)

    assert len(qargs) == expected_count
    assert all(len(qarg) == 3 for qarg in qargs)

    # Verify we get different combinations, not just (0,1,2)
    if num_qubits >= 4:
        # Should have (0,1,2), (0,1,3), (0,2,3), (1,2,3), etc.
        assert (0, 1, 2) in qargs
        assert (0, 1, 3) in qargs
        assert (0, 2, 3) in qargs
        assert (1, 2, 3) in qargs

    # Verify no duplicates
    assert len(qargs) == len(set(qargs))

    # Test with 4-qubit operation if device has enough qubits
    if num_qubits >= 4:
        mock_op_4q = MagicMock()
        mock_op_4q.name.return_value = "test_4q"
        mock_op_4q.qubits_num.return_value = 4
        mock_op_4q.parameters_num.return_value = 0
        mock_op_4q.duration.return_value = None
        mock_op_4q.fidelity.return_value = None
        mock_op_4q.sites.return_value = None
        mock_op_4q.is_zoned.return_value = False

        qargs_4q = backend._get_operation_qargs(mock_op_4q)  # noqa: SLF001
        from math import comb

        expected_4q_count = comb(num_qubits, 4)

        assert len(qargs_4q) == expected_4q_count
        assert all(len(qarg) == 4 for qarg in qargs_4q)
        assert len(qargs_4q) == len(set(qargs_4q))  # No duplicates
