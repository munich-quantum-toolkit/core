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
import math

import pytest

from mqt.core.qdmi.qiskit import (
    QiskitBackend,
    clear_operation_translators,
    list_operation_translators,
)

_qiskit_present = importlib.util.find_spec("qiskit") is not None

pytestmark = pytest.mark.skipif(not _qiskit_present, reason="qiskit not installed")

if _qiskit_present:
    from qiskit import QuantumCircuit


def setup_module() -> None:  # noqa: D103
    clear_operation_translators(keep_defaults=True)


def test_default_translators_registered() -> None:
    """Verify that all default gate translators are registered."""
    translators = list_operation_translators()

    # Check single-qubit Pauli gates
    assert "x" in translators
    assert "y" in translators
    assert "z" in translators
    assert "h" in translators
    assert "i" in translators or "id" in translators

    # Check phase gates
    assert "s" in translators
    assert "sdg" in translators
    assert "t" in translators
    assert "tdg" in translators
    assert "sx" in translators
    assert "sxdg" in translators

    # Check rotation gates
    assert "rx" in translators
    assert "ry" in translators
    assert "rz" in translators

    # Check two-qubit gates
    assert "cx" in translators or "cnot" in translators
    assert "cy" in translators
    assert "cz" in translators
    assert "swap" in translators

    # Check measurement
    assert "measure" in translators


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


def test_translator_count() -> None:
    """Verify that many default translators are registered."""
    translators = list_operation_translators()

    # Should have at least 30 translators now (including new gates from mqt_to_qiskit.py)
    # This verifies the expanded gate support is in place
    assert len(translators) >= 30


def test_gate_name_case_insensitivity() -> None:
    """Test that gate mapping is case-insensitive."""
    backend = QiskitBackend()

    # Test lowercase
    gate_lower = backend._map_operation_to_gate("cz")  # noqa: SLF001
    assert gate_lower is not None

    # Test uppercase
    gate_upper = backend._map_operation_to_gate("CZ")  # noqa: SLF001
    assert gate_upper is not None

    # Test mixed case
    gate_mixed = backend._map_operation_to_gate("Cz")  # noqa: SLF001
    assert gate_mixed is not None


def test_parametric_gate_translators() -> None:
    """Test that parametric gates have translators with parameter handling."""
    from mqt.core.qdmi.qiskit import InstructionContext, get_operation_translator

    # Test RX translator
    rx_translator = get_operation_translator("rx")
    ctx = InstructionContext(name="rx", qubits=[0], params=[1.5707963267948966])
    instructions = rx_translator(ctx)

    assert len(instructions) == 1
    assert instructions[0].name == "rx"
    assert instructions[0].qubits == [0]
    assert instructions[0].params == [1.5707963267948966]

    # Test RZZ translator
    rzz_translator = get_operation_translator("rzz")
    ctx = InstructionContext(name="rzz", qubits=[0, 1], params=[math.pi])
    instructions = rzz_translator(ctx)

    assert len(instructions) == 1
    assert instructions[0].name == "rzz"
    assert instructions[0].qubits == [0, 1]
    assert instructions[0].params == [math.pi]

    # Test U3 gate (three parameters)
    u3_translator = get_operation_translator("u3")
    ctx = InstructionContext(name="u3", qubits=[0], params=[1.57, 0.0, math.pi])
    instructions = u3_translator(ctx)

    assert len(instructions) == 1
    assert instructions[0].name == "u3"
    assert instructions[0].qubits == [0]
    assert instructions[0].params == [1.57, 0.0, math.pi]


def test_backend_target_includes_operations() -> None:
    """Verify that the backend Target includes operations from device capabilities."""
    backend = QiskitBackend()
    target = backend.target

    # Check that target has operations
    operation_names = target.operation_names
    assert "measure" in operation_names

    # The mock device should expose CZ
    assert "cz" in operation_names

    # Target should have the correct number of qubits
    assert target.num_qubits > 0


def test_comprehensive_translator_coverage() -> None:
    """Test that all major gate categories have translators."""
    translators = list_operation_translators()

    # Pauli gates
    pauli_gates = ["x", "y", "z"]
    for gate in pauli_gates:
        assert gate in translators, f"Missing translator for {gate}"

    # Clifford gates
    clifford_gates = ["h", "s", "sdg"]
    for gate in clifford_gates:
        assert gate in translators, f"Missing translator for {gate}"

    # T gates
    t_gates = ["t", "tdg"]
    for gate in t_gates:
        assert gate in translators, f"Missing translator for {gate}"

    # Rotation gates
    rotation_gates = ["rx", "ry", "rz"]
    for gate in rotation_gates:
        assert gate in translators, f"Missing translator for {gate}"

    # Two-qubit gates
    two_qubit_gates = ["cx", "cy", "cz", "swap", "dcx", "ecr"]
    for gate in two_qubit_gates:
        assert gate in translators, f"Missing translator for {gate}"

    # Two-qubit parametric gates
    two_qubit_param = ["rxx", "ryy", "rzz", "rzx", "xx_plus_yy", "xx_minus_yy"]
    for gate in two_qubit_param:
        assert gate in translators, f"Missing translator for {gate}"

    # Universal gates
    universal_gates = ["u", "u2", "u3"]
    for gate in universal_gates:
        assert gate in translators, f"Missing translator for {gate}"


def test_mqt_to_qiskit_parity() -> None:
    """Verify that all gates from mqt_to_qiskit.py are supported."""
    from qiskit.circuit.library import DCXGate, ECRGate, RZXGate, U2Gate, U3Gate, XXMinusYYGate, XXPlusYYGate

    backend = QiskitBackend()

    # Test DCX gate
    gate = backend._map_operation_to_gate("dcx")  # noqa: SLF001
    assert isinstance(gate, DCXGate)

    # Test ECR gate
    gate = backend._map_operation_to_gate("ecr")  # noqa: SLF001
    assert isinstance(gate, ECRGate)

    # Test RZX gate
    gate = backend._map_operation_to_gate("rzx")  # noqa: SLF001
    assert isinstance(gate, RZXGate)

    # Test U2 gate
    gate = backend._map_operation_to_gate("u2")  # noqa: SLF001
    assert isinstance(gate, U2Gate)

    # Test U3 gate
    gate = backend._map_operation_to_gate("u3")  # noqa: SLF001
    assert isinstance(gate, U3Gate)

    # Test XXPlusYY gate
    gate = backend._map_operation_to_gate("xx_plus_yy")  # noqa: SLF001
    assert isinstance(gate, XXPlusYYGate)

    # Test XXMinusYY gate
    gate = backend._map_operation_to_gate("xx_minus_yy")  # noqa: SLF001
    assert isinstance(gate, XXMinusYYGate)
