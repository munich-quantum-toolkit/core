# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the quantum computation IR."""

from __future__ import annotations

from mqt.core.ir import QuantumComputation
from mqt.core.ir.operations import OpType, StandardOperation


def test_if_else_operation_register() -> None:
    """Test the creation of an if-else operation."""
    qc = QuantumComputation()
    qc.add_qubit_register(1)
    c = qc.add_classical_register(1)

    qc.if_else(
        then_operation=StandardOperation(0, OpType.x),
        else_operation=StandardOperation(0, OpType.y),
        control_register=c,
    )

    qasm = qc.qasm3_str()
    expected = """
        // i 0
        // o 0
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        bit[1] c;
        if (c == 1) {
            x q[0];
        } else {
            y q[0];
        }
    """
    # Remove all whitespace from both strings before comparison
    assert "".join(qasm.split()) == "".join(expected.split())


def test_if_else_operation_bit() -> None:
    """Test the creation of an if-else operation."""
    qc = QuantumComputation()
    qc.add_qubit_register(1)
    qc.add_classical_register(1)

    qc.if_else(
        then_operation=StandardOperation(0, OpType.x),
        else_operation=StandardOperation(0, OpType.y),
        control_bit=0,
    )

    qasm = qc.qasm3_str()
    expected = """
        // i 0
        // o 0
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        bit[1] c;
        if (c[0]) {
            x q[0];
        } else {
            y q[0];
        }
    """
    # Remove all whitespace from both strings before comparison
    assert "".join(qasm.split()) == "".join(expected.split())


def test_if_operation_register() -> None:
    """Test the creation of an if-else operation."""
    qc = QuantumComputation()
    qc.add_qubit_register(1)
    c = qc.add_classical_register(1)

    qc.if_(OpType.x, target=0, control_register=c)

    qasm = qc.qasm3_str()
    expected = """
        // i 0
        // o 0
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        bit[1] c;
        if (c == 1) {
            x q[0];
        }
    """
    # Remove all whitespace from both strings before comparison
    assert "".join(qasm.split()) == "".join(expected.split())


def test_if_operation_bit() -> None:
    """Test the creation of an if-else operation."""
    qc = QuantumComputation()
    qc.add_qubit_register(1)
    qc.add_classical_register(1)

    qc.if_(OpType.x, target=0, control_bit=0)

    qasm = qc.qasm3_str()
    expected = """
        // i 0
        // o 0
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        bit[1] c;
        if (c[0]) {
            x q[0];
        }
    """
    # Remove all whitespace from both strings before comparison
    assert "".join(qasm.split()) == "".join(expected.split())


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
