# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the quantum computation IR."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mqt.core.ir import QuantumComputation

if TYPE_CHECKING:
    from pathlib import Path


def test_bell_state_circuit() -> None:
    """Test the creation of a Bell state circuit."""
    qc = QuantumComputation()
    q = qc.add_qubit_register(2)
    c = qc.add_classical_register(2)

    qc.h(q[0])
    qc.cx(q[0], q[1])
    qc.measure(q[0], c[0])
    qc.measure(q[1], c[1])

    qasm = qc.qasm3_str()
    expected = """
        // i 0 1
        // o 0 1
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        bit[2] c;
        h q[0];
        cx q[0], q[1];
        c[0] = measure q[0];
        c[1] = measure q[1];
    """
    # Remove all whitespace from both strings before comparison
    assert "".join(qasm.split()) == "".join(expected.split())


def test_num_output_qubits_excludes_garbage() -> None:
    """Test that the output count reflects the circuit's garbage metadata."""
    qc = QuantumComputation(3)
    assert qc.num_output_qubits == 3

    qc.set_circuit_qubit_garbage(1)
    assert qc.num_output_qubits == 2


def test_qasm_file_exports_match_string_exports(tmp_path: Path) -> None:
    """Test that both OpenQASM file exporters use the circuit serializer."""
    qc = QuantumComputation(2)
    qc.h(0)
    qc.cx(0, 1)

    qasm2_path = tmp_path / "circuit.qasm2"
    qc.qasm2(str(qasm2_path))
    assert qasm2_path.read_text() == qc.qasm2_str()

    qasm3_path = tmp_path / "circuit.qasm3"
    qc.qasm3(str(qasm3_path))
    assert qasm3_path.read_text() == qc.qasm3_str()
