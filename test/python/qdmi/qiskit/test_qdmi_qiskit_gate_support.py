# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for gate support in the QDMI Qiskit backend."""

from __future__ import annotations

import importlib.util

import pytest

from mqt.core.qdmi.qiskit import clear_operation_translators, list_operation_translators

_qiskit_present = importlib.util.find_spec("qiskit") is not None

pytestmark = pytest.mark.skipif(not _qiskit_present, reason="qiskit not installed")

if _qiskit_present:
    from qiskit import QuantumCircuit

    from mqt.core.qdmi.qiskit import QiskitBackend


def setup_module() -> None:  # noqa: D103
    clear_operation_translators(keep_defaults=True)


def test_gate_mapping_to_qiskit_gates() -> None:
    """Test that device operations are correctly mapped to Qiskit gates."""
    from qiskit.circuit.library import CZGate, HGate, RXGate, XGate

    backend = QiskitBackend()

    # Test the _map_operation_to_gate function
    # Single-qubit Pauli gates
    gate = backend._map_operation_to_gate("x")  # noqa: SLF001
    assert isinstance(gate, XGate)

    gate = backend._map_operation_to_gate("h")  # noqa: SLF001
    assert isinstance(gate, HGate)

    # Two-qubit gates
    gate = backend._map_operation_to_gate("cz")  # noqa: SLF001
    assert isinstance(gate, CZGate)

    # Parametric gates
    gate = backend._map_operation_to_gate("rx")  # noqa: SLF001
    assert isinstance(gate, RXGate)

    # Unsupported operation should return None
    gate = backend._map_operation_to_gate("unsupported_gate")  # noqa: SLF001
    assert gate is None


def test_backend_supports_cz_gate() -> None:
    """Test that the backend can execute CZ gate circuits (supported by mock device)."""
    backend = QiskitBackend()

    # The mock device supports CZ
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = backend.run(qc, shots=100)
    counts = job.get_counts()
    assert sum(counts.values()) == 100


def test_default_translators_registered() -> None:
    """Verify that all expected default translators are registered."""
    translators = list_operation_translators()
    expected_translators = {
        "x",
        "y",
        "z",
        "h",
        "i",
        "id",
        "s",
        "sdg",
        "t",
        "tdg",
        "sx",
        "sxdg",
        "rx",
        "ry",
        "rz",
        "p",
        "phase",
        "r",
        "prx",
        "u",
        "u2",
        "u3",
        "cx",
        "cnot",
        "cy",
        "cz",
        "ch",
        "swap",
        "iswap",
        "dcx",
        "ecr",
        "rxx",
        "ryy",
        "rzz",
        "rzx",
        "xx_plus_yy",
        "xx_minus_yy",
        "measure",
    }

    for gate_name in expected_translators:
        assert gate_name in translators, f"Expected translator '{gate_name}' not found"


def test_map_operation_pauli_gates() -> None:
    """Test mapping of Pauli gates."""
    from qiskit.circuit.library import IGate, XGate, YGate, ZGate

    backend = QiskitBackend()

    assert isinstance(backend._map_operation_to_gate("x"), XGate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("y"), YGate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("z"), ZGate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("i"), IGate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("id"), IGate)  # noqa: SLF001


def test_map_operation_phase_gates() -> None:
    """Test mapping of phase gates."""
    from qiskit.circuit.library import PhaseGate, SdgGate, SGate, SXdgGate, SXGate, TdgGate, TGate

    backend = QiskitBackend()

    assert isinstance(backend._map_operation_to_gate("s"), SGate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("sdg"), SdgGate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("t"), TGate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("tdg"), TdgGate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("sx"), SXGate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("sxdg"), SXdgGate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("p"), PhaseGate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("phase"), PhaseGate)  # noqa: SLF001


def test_map_operation_rotation_gates() -> None:
    """Test mapping of rotation gates."""
    from qiskit.circuit.library import RGate, RXGate, RYGate, RZGate

    backend = QiskitBackend()

    assert isinstance(backend._map_operation_to_gate("rx"), RXGate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("ry"), RYGate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("rz"), RZGate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("r"), RGate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("prx"), RGate)  # noqa: SLF001


def test_map_operation_universal_gates() -> None:
    """Test mapping of universal gates."""
    from qiskit.circuit.library import U2Gate, U3Gate

    backend = QiskitBackend()

    assert isinstance(backend._map_operation_to_gate("u"), U3Gate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("u3"), U3Gate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("u2"), U2Gate)  # noqa: SLF001


def test_map_operation_two_qubit_gates() -> None:
    """Test mapping of two-qubit gates."""
    from qiskit.circuit.library import (
        CHGate,
        CXGate,
        CYGate,
        CZGate,
        DCXGate,
        ECRGate,
        SwapGate,
        iSwapGate,
    )

    backend = QiskitBackend()

    assert isinstance(backend._map_operation_to_gate("cx"), CXGate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("cnot"), CXGate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("cy"), CYGate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("cz"), CZGate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("ch"), CHGate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("swap"), SwapGate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("iswap"), iSwapGate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("dcx"), DCXGate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("ecr"), ECRGate)  # noqa: SLF001


def test_map_operation_parametric_two_qubit_gates() -> None:
    """Test mapping of parametric two-qubit gates."""
    from qiskit.circuit.library import (
        RXXGate,
        RYYGate,
        RZXGate,
        RZZGate,
        XXMinusYYGate,
        XXPlusYYGate,
    )

    backend = QiskitBackend()

    assert isinstance(backend._map_operation_to_gate("rxx"), RXXGate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("ryy"), RYYGate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("rzz"), RZZGate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("rzx"), RZXGate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("xx_plus_yy"), XXPlusYYGate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("xx_minus_yy"), XXMinusYYGate)  # noqa: SLF001


def test_map_operation_case_insensitive() -> None:
    """Test that gate mapping is case-insensitive."""
    from qiskit.circuit.library import HGate, XGate

    backend = QiskitBackend()

    # Lower case
    assert isinstance(backend._map_operation_to_gate("x"), XGate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("h"), HGate)  # noqa: SLF001

    # Upper case should also work (op_name.lower() is called)
    assert isinstance(backend._map_operation_to_gate("X"), XGate)  # noqa: SLF001
    assert isinstance(backend._map_operation_to_gate("H"), HGate)  # noqa: SLF001


def test_map_operation_returns_none_for_unknown() -> None:
    """Test that unknown operations return None."""
    backend = QiskitBackend()

    assert backend._map_operation_to_gate("unknown_gate") is None  # noqa: SLF001
    assert backend._map_operation_to_gate("custom_op") is None  # noqa: SLF001
    assert backend._map_operation_to_gate("") is None  # noqa: SLF001


def test_backend_target_includes_measure() -> None:
    """Test that the backend target includes measure operation."""
    backend = QiskitBackend()

    assert "measure" in backend.target.operation_names


def test_backend_target_qubit_count_matches_device() -> None:
    """Test that backend target qubit count is based on device capabilities."""
    backend = QiskitBackend()

    # Target qubit count should be at least the device num_qubits
    # (may be higher if zones are included in the target)
    assert backend.target.num_qubits >= backend._capabilities.num_qubits  # noqa: SLF001


def test_backend_build_target_adds_operations() -> None:
    """Test that _build_target adds operations from capabilities."""
    backend = QiskitBackend()
    target = backend._build_target()  # noqa: SLF001

    # Should have at least measure
    assert "measure" in target.operation_names

    # Should have operations from device capabilities
    assert len(target.operation_names) > 0


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


def test_get_operation_qargs_two_qubit_with_coupling_map() -> None:
    """Test _get_operation_qargs for two-qubit operations with coupling map."""
    backend = QiskitBackend()

    # If device has coupling map, should use it
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
            sites=None,
        )

        qargs = backend._get_operation_qargs(op_info)  # noqa: SLF001

        # Should match coupling map
        assert all(len(qarg) == 2 for qarg in qargs)


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
