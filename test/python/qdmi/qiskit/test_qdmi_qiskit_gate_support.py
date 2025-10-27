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
    counts = job.get_counts()
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

    # Measurement should be in target if device provides it
    # If not provided, a warning should be issued (tested separately)
    if "measure" in backend._capabilities.operations:  # noqa: SLF001
        assert "measure" in backend.target.operation_names


def test_get_operation_qargs_single_qubit() -> None:
    """Test _get_operation_qargs for single-qubit operations."""
    backend = QiskitBackend()

    # Create a mock operation info for single qubit
    from mqt.core.qdmi.qiskit.capabilities import DeviceOperationInfo

    op_info = DeviceOperationInfo(
        name="test_op",
        qubits_num=1,
        parameters_num=0,
        duration=None,
        fidelity=None,
        interaction_radius=None,
        blocking_radius=None,
        idling_fidelity=None,
        is_zoned=None,
        mean_shuttling_speed=None,
        sites=None,
    )

    qargs = backend._get_operation_qargs(op_info)  # noqa: SLF001

    # Should return all single qubit indices
    assert len(qargs) == backend._capabilities.num_qubits  # noqa: SLF001
    assert all(len(qarg) == 1 for qarg in qargs)


def test_get_operation_qargs_two_qubit_with_coupling_map_fallback() -> None:
    """Test _get_operation_qargs for two-qubit operations with coupling map fallback.

    When an operation has sites=None, it should fall back to using the device's
    full coupling map (all physical connections).
    """
    backend = QiskitBackend()

    # If device has coupling map, should use it as fallback when sites=None
    if backend._capabilities.coupling_map:  # noqa: SLF001
        from mqt.core.qdmi.qiskit.capabilities import DeviceOperationInfo

        op_info = DeviceOperationInfo(
            name="test_2q",
            qubits_num=2,
            parameters_num=0,
            duration=None,
            fidelity=None,
            interaction_radius=None,
            blocking_radius=None,
            idling_fidelity=None,
            is_zoned=None,
            mean_shuttling_speed=None,
            sites=None,  # No specific sites - should use device coupling map
        )

        qargs = backend._get_operation_qargs(op_info)  # noqa: SLF001

        # Should match the full device coupling map
        assert all(len(qarg) == 2 for qarg in qargs)

        # Verify it returns the device's coupling map
        device_coupling_map = backend._capabilities.coupling_map  # noqa: SLF001
        assert len(qargs) == len(device_coupling_map)
        assert set(qargs) == {(int(pair[0]), int(pair[1])) for pair in device_coupling_map}


def test_get_operation_qargs_with_specific_sites() -> None:
    """Test _get_operation_qargs when operation has specific sites."""
    backend = QiskitBackend()

    from mqt.core.qdmi.qiskit.capabilities import DeviceOperationInfo

    # Single qubit on specific sites
    op_info = DeviceOperationInfo(
        name="test_op",
        qubits_num=1,
        parameters_num=0,
        duration=None,
        fidelity=None,
        interaction_radius=None,
        blocking_radius=None,
        idling_fidelity=None,
        is_zoned=None,
        mean_shuttling_speed=None,
        sites=(0, 1, 2),  # Specific sites
    )

    qargs = backend._get_operation_qargs(op_info)  # noqa: SLF001

    # Should only include specified sites
    assert len(qargs) == 3
    assert (0,) in qargs
    assert (1,) in qargs
    assert (2,) in qargs


def test_get_operation_qargs_two_qubit_operation_with_subset_of_coupling_map() -> None:
    """Test that two-qubit operations can specify their own subset of the device coupling map.

    This addresses the scenario where a device has a coupling map describing all physical
    connections, but individual operations (e.g., CZ vs CX) are only supported on subsets
    of those connections.

    Example: Device has coupling map {(0,1), (1,2), (2,3), (3,0), (0,3), (3,2), (2,1), (1,0)}
    but CZ is only supported on {(0,1), (1,2), (2,3), (3,2), (2,1), (1,0)} and CX only on
    {(0,3), (3,0)}.
    """
    backend = QiskitBackend()

    from mqt.core.qdmi.qiskit.capabilities import DeviceOperationInfo

    # Two-qubit operation with specific coupling pairs (subset of device coupling map)
    # This represents e.g., a CZ gate that's only available on specific edges
    cz_pairs = ((0, 1), (1, 2), (2, 3), (3, 2), (2, 1), (1, 0))

    op_info = DeviceOperationInfo(
        name="cz",
        qubits_num=2,
        parameters_num=0,
        duration=None,
        fidelity=None,
        interaction_radius=None,
        blocking_radius=None,
        idling_fidelity=None,
        is_zoned=None,
        mean_shuttling_speed=None,
        sites=cz_pairs,  # Specific coupling pairs for this operation
    )

    qargs = backend._get_operation_qargs(op_info)  # noqa: SLF001

    # Should return exactly the specified coupling pairs, not the full device coupling map
    assert len(qargs) == len(cz_pairs)
    assert all(len(qarg) == 2 for qarg in qargs)

    # Verify all specified pairs are present
    for pair in cz_pairs:
        assert pair in qargs

    # Verify no extra pairs beyond what was specified
    assert set(qargs) == set(cz_pairs)

    # Now test another operation with a different subset (e.g., CX)
    cx_pairs = ((0, 3), (3, 0))

    cx_op_info = DeviceOperationInfo(
        name="cx",
        qubits_num=2,
        parameters_num=0,
        duration=None,
        fidelity=None,
        interaction_radius=None,
        blocking_radius=None,
        idling_fidelity=None,
        is_zoned=None,
        mean_shuttling_speed=None,
        sites=cx_pairs,
    )

    cx_qargs = backend._get_operation_qargs(cx_op_info)  # noqa: SLF001

    # Should return exactly the CX-specific pairs
    assert len(cx_qargs) == len(cx_pairs)
    assert set(cx_qargs) == set(cx_pairs)

    # Verify the two operations have different coupling maps
    assert set(qargs) != set(cx_qargs)


def test_get_operation_qargs_multi_qubit_generates_all_combinations() -> None:
    """Test that multi-qubit operations (3+ qubits) generate all possible combinations.

    This verifies that the fallback for multi-qubit operations properly advertises
    the full capability by generating all combinations of qubits, not just the first
    contiguous qubits (e.g., for a 3-qubit operation on a 5-qubit device, it should
    generate all C(5,3) = 10 combinations, not just (0,1,2)).
    """
    backend = QiskitBackend()

    from mqt.core.qdmi.qiskit.capabilities import DeviceOperationInfo

    # 3-qubit operation without specific sites - should generate all combinations
    op_info = DeviceOperationInfo(
        name="ccx",  # Toffoli gate as example
        qubits_num=3,
        parameters_num=0,
        duration=None,
        fidelity=None,
        interaction_radius=None,
        blocking_radius=None,
        idling_fidelity=None,
        is_zoned=None,
        mean_shuttling_speed=None,
        sites=None,  # No specific sites - should generate all combinations
    )

    qargs = backend._get_operation_qargs(op_info)  # noqa: SLF001

    # Should generate all 3-qubit combinations
    num_qubits = backend._capabilities.num_qubits  # noqa: SLF001
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
        op_info_4q = DeviceOperationInfo(
            name="test_4q",
            qubits_num=4,
            parameters_num=0,
            duration=None,
            fidelity=None,
            interaction_radius=None,
            blocking_radius=None,
            idling_fidelity=None,
            is_zoned=None,
            mean_shuttling_speed=None,
            sites=None,
        )

        qargs_4q = backend._get_operation_qargs(op_info_4q)  # noqa: SLF001
        from math import comb

        expected_4q_count = comb(num_qubits, 4)

        assert len(qargs_4q) == expected_4q_count
        assert all(len(qarg) == 4 for qarg in qargs_4q)
        assert len(qargs_4q) == len(set(qargs_4q))  # No duplicates
