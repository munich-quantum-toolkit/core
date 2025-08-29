# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test the quantum computation IR."""

from __future__ import annotations

import pytest

from mqt.core.ir import QuantumComputation
from mqt.core.ir.operations import ComparisonKind, OpType, StandardOperation


@pytest.mark.parametrize(
    ("comparison_kind", "comparison_kind_string"),
    [
        (ComparisonKind.eq, "=="),
        (ComparisonKind.neq, "!="),
        (ComparisonKind.lt, "<"),
        (ComparisonKind.leq, "<="),
        (ComparisonKind.gt, ">"),
        (ComparisonKind.geq, ">="),
    ],
)
def test_if_else_operation_register(comparison_kind: ComparisonKind, comparison_kind_string: str) -> None:
    """Test the creation of an if-else operation."""
    qc = QuantumComputation()
    qc.add_qubit_register(1)
    c = qc.add_classical_register(1)

    qc.if_else(
        then_operation=StandardOperation(0, OpType.x),
        else_operation=StandardOperation(0, OpType.y),
        control_register=c,
        comparison_kind=comparison_kind,
    )

    qasm = qc.qasm3_str()
    expected = f"""
        // i 0
        // o 0
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        bit[1] c;
        if (c {comparison_kind_string} 1) {{
            x q[0];
        }} else {{
            y q[0];
        }}
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
