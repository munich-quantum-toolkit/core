# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Test Qiskit import."""

from __future__ import annotations

from typing import cast

import pytest
from packaging import version
from qiskit import __version__ as qiskit_version
from qiskit import transpile
from qiskit.circuit import (
    AncillaRegister,
    ClassicalRegister,
    Parameter,
    QuantumCircuit,
    QuantumRegister,
)
from qiskit.circuit.library import U2Gate, XXMinusYYGate, XXPlusYYGate
from qiskit.providers.fake_provider import GenericBackendV2

from mqt.core.ir.operations import (
    ComparisonKind,
    CompoundOperation,
    IfElseOperation,
    OpType,
    StandardOperation,
    SymbolicOperation,
)
from mqt.core.ir.symbolic import Expression
from mqt.core.plugins.qiskit import mqt_to_qiskit, qiskit_to_mqt


def test_empty_circuit() -> None:
    """Test roundtrip of empty circuit."""
    qc = QuantumCircuit()
    print(qc)

    mqt_qc = qiskit_to_mqt(qc)
    print(mqt_qc)
    assert mqt_qc.num_qubits == 0
    assert mqt_qc.num_ops == 0

    qiskit_qc = mqt_to_qiskit(mqt_qc)
    print(qiskit_qc)
    assert qiskit_qc.num_qubits == 0
    assert len(qiskit_qc) == 0
    assert qc == qiskit_qc


def test_multiple_quantum_registers() -> None:
    """Test roundtrip of circuit with multiple quantum registers."""
    p = QuantumRegister(2, "p")
    q = QuantumRegister(2, "q")
    qc = QuantumCircuit(p, q)
    print(qc)

    mqt_qc = qiskit_to_mqt(qc)
    print(mqt_qc)
    assert mqt_qc.num_qubits == 4
    assert mqt_qc.num_ops == 0
    assert len(mqt_qc.qregs) == 2
    assert "p" in mqt_qc.qregs
    assert "q" in mqt_qc.qregs
    assert mqt_qc.qregs["p"].size == 2
    assert mqt_qc.qregs["p"].start == 0
    assert mqt_qc.qregs["q"].size == 2
    assert mqt_qc.qregs["q"].start == 2

    qiskit_qc = mqt_to_qiskit(mqt_qc)
    print(qiskit_qc)
    assert qiskit_qc.num_qubits == 4
    assert len(qiskit_qc.qregs) == 2
    assert qc == qiskit_qc


def test_quantum_and_ancillary_registers() -> None:
    """Test roundtrip of circuit with quantum and ancillary registers."""
    q = QuantumRegister(2, "q")
    a = AncillaRegister(1, "a")
    qc = QuantumCircuit(q, a)
    print(qc)

    mqt_qc = qiskit_to_mqt(qc)
    print(mqt_qc)
    assert mqt_qc.num_qubits == 3
    assert mqt_qc.num_ops == 0
    assert mqt_qc.num_ancilla_qubits == 1
    assert mqt_qc.is_circuit_qubit_ancillary(2)

    qiskit_qc = mqt_to_qiskit(mqt_qc)
    print(qiskit_qc)
    assert qiskit_qc.num_qubits == 3
    assert qiskit_qc.num_ancillas == 1
    assert qc == qiskit_qc


def test_multiple_classical_registers() -> None:
    """Test roundtrip of circuit with multiple classical registers."""
    c = ClassicalRegister(2, "c")
    d = ClassicalRegister(2, "d")
    qc = QuantumCircuit(c, d)
    print(qc)

    mqt_qc = qiskit_to_mqt(qc)
    print(mqt_qc)
    assert mqt_qc.num_classical_bits == 4
    assert len(mqt_qc.cregs) == 2
    assert "c" in mqt_qc.cregs
    assert "d" in mqt_qc.cregs
    assert mqt_qc.cregs["c"].size == 2
    assert mqt_qc.cregs["c"].start == 0
    assert mqt_qc.cregs["d"].size == 2
    assert mqt_qc.cregs["d"].start == 2

    qiskit_qc = mqt_to_qiskit(mqt_qc)
    print(qiskit_qc)
    assert qiskit_qc.num_clbits == 4
    assert qc == qiskit_qc


def test_single_gate() -> None:
    """Test roundtrip of single-gate circuit."""
    qc = QuantumCircuit(1)
    qc.h(0)
    print(qc)

    mqt_qc = qiskit_to_mqt(qc)
    print(mqt_qc)
    assert mqt_qc.num_qubits == 1
    assert mqt_qc.num_ops == 1
    assert mqt_qc[0].name.strip() == "h"

    qiskit_qc = mqt_to_qiskit(mqt_qc)
    print(qiskit_qc)
    assert qiskit_qc.num_qubits == 1
    assert len(qiskit_qc) == 1
    assert qc == qiskit_qc


def test_two_qubit_gate() -> None:
    """Test roundtrip of two-qubit gate."""
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    print(qc)

    mqt_qc = qiskit_to_mqt(qc)
    print(mqt_qc)
    assert mqt_qc.num_qubits == 2
    assert mqt_qc.num_ops == 1
    assert mqt_qc[0].name.strip() == "x"
    assert {control.qubit for control in mqt_qc[0].controls} == {0}

    qiskit_qc = mqt_to_qiskit(mqt_qc)
    print(qiskit_qc)
    assert qiskit_qc.num_qubits == 2
    assert len(qiskit_qc) == 1
    assert qc == qiskit_qc


def test_ccx() -> None:
    """Test roundtrip of ccx gate."""
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    print(qc)

    mqt_qc = qiskit_to_mqt(qc)
    print(mqt_qc)
    assert mqt_qc.num_qubits == 3
    assert mqt_qc.num_ops == 1
    assert mqt_qc[0].name.strip() == "x"
    assert {control.qubit for control in mqt_qc[0].controls} == {0, 1}

    qiskit_qc = mqt_to_qiskit(mqt_qc)
    print(qiskit_qc)
    assert qiskit_qc.num_qubits == 3
    assert len(qiskit_qc) == 1
    assert qiskit_qc[0].operation.name == "ccx"


skip_if_qiskit_version_ge_2_1 = pytest.mark.skipif(
    version.parse(qiskit_version) >= version.parse("2.1"),
    reason="Qiskit 2.1+ deprecates setting a mode for `mcx` gates",
)


@skip_if_qiskit_version_ge_2_1
def test_mcx_recursive() -> None:
    """Test roundtrip of large mcx gate."""
    qc = QuantumCircuit(9)
    qc.mcx(control_qubits=list(range(7)), target_qubit=7, ancilla_qubits=list(range(8, 9)), mode="recursion")
    print(qc)

    mqt_qc = qiskit_to_mqt(qc)
    print(mqt_qc)
    assert mqt_qc.num_qubits == 9
    assert mqt_qc.num_ops == 1
    assert mqt_qc[0].name.strip() == "x"
    assert {control.qubit for control in mqt_qc[0].controls} == {0, 1, 2, 3, 4, 5, 6}
    assert not mqt_qc[0].acts_on(8)

    qiskit_qc = mqt_to_qiskit(mqt_qc)
    print(qiskit_qc)
    assert qiskit_qc.num_qubits == 9
    assert len(qiskit_qc) == 1
    assert qiskit_qc[0].operation.name == "mcx"


@skip_if_qiskit_version_ge_2_1
def test_small_mcx_recursive() -> None:
    """Test roundtrip of small mcx_recursive gate."""
    qc = QuantumCircuit(5)
    qc.mcx(target_qubit=4, control_qubits=list(range(4)), mode="recursion")
    print(qc)

    mqt_qc = qiskit_to_mqt(qc)
    print(mqt_qc)
    assert mqt_qc.num_qubits == 5
    assert mqt_qc.num_ops == 1
    assert mqt_qc[0].name.strip() == "x"
    assert {control.qubit for control in mqt_qc[0].controls} == {0, 1, 2, 3}

    qiskit_qc = mqt_to_qiskit(mqt_qc)
    print(qiskit_qc)
    assert qiskit_qc.num_qubits == 5
    assert len(qiskit_qc) == 1
    assert qiskit_qc[0].operation.name == "mcx"


@skip_if_qiskit_version_ge_2_1
def test_mcx_vchain() -> None:
    """Test roundtrip of mcx gate with v-chain."""
    qc = QuantumCircuit(9)
    qc.mcx(target_qubit=5, control_qubits=list(range(5)), ancilla_qubits=list(range(6, 9)), mode="v-chain")
    print(qc)

    mqt_qc = qiskit_to_mqt(qc)
    print(mqt_qc)
    assert mqt_qc.num_qubits == 9
    assert mqt_qc.num_ops == 1
    assert mqt_qc[0].name.strip() == "x"
    assert {control.qubit for control in mqt_qc[0].controls} == {0, 1, 2, 3, 4}
    for i in range(6, 9):
        assert not mqt_qc[0].acts_on(i)

    qiskit_qc = mqt_to_qiskit(mqt_qc)
    print(qiskit_qc)
    assert qiskit_qc.num_qubits == 9
    assert len(qiskit_qc) == 1
    assert qiskit_qc[0].operation.name == "mcx"


def test_mcx_without_mode() -> None:
    """Test roundtrip of mcx gate without a dedicated mode being set (Qiskit 2.1+)."""
    qc = QuantumCircuit(5)
    qc.mcx(target_qubit=4, control_qubits=list(range(4)))
    print(qc)

    mqt_qc = qiskit_to_mqt(qc)
    print(mqt_qc)
    assert mqt_qc.num_qubits == 5
    assert mqt_qc.num_ops == 1
    assert mqt_qc[0].name.strip() == "x"
    assert {control.qubit for control in mqt_qc[0].controls} == {0, 1, 2, 3}

    qiskit_qc = mqt_to_qiskit(mqt_qc)
    print(qiskit_qc)
    assert qiskit_qc.num_qubits == 5
    assert len(qiskit_qc) == 1
    assert qiskit_qc[0].operation.name == "mcx"


def test_custom_gate() -> None:
    """Test roundtrip of custom gate."""
    custom_instr = QuantumCircuit(3, 1)
    custom_instr.h(0)
    custom_instr.cx(0, 1)
    custom_instr.cx(0, 2)
    custom_instr.measure(0, 0)
    custom_instr = custom_instr.to_instruction()
    qc = QuantumCircuit(3, 1)
    qc.append(custom_instr, range(3), range(1))
    print(qc.draw(cregbundle=False))

    mqt_qc = qiskit_to_mqt(qc)
    print(mqt_qc)
    assert mqt_qc.num_qubits == 3
    assert mqt_qc.num_ops == 1
    assert isinstance(mqt_qc[0], CompoundOperation)
    assert mqt_qc[0][0].name.strip() == "h"
    assert mqt_qc[0][1].name.strip() == "x"
    assert mqt_qc[0][2].name.strip() == "x"
    assert mqt_qc[0][3].name.strip() == "measure"
    assert {control.qubit for control in mqt_qc[0][1].controls} == {0}
    assert {control.qubit for control in mqt_qc[0][2].controls} == {0}

    qiskit_qc = mqt_to_qiskit(mqt_qc)
    print(qiskit_qc.draw(cregbundle=False))
    assert qiskit_qc.num_qubits == 3
    assert len(qiskit_qc) == 1


def test_ancilla() -> None:
    """Test roundtrip of ancilla register."""
    anc_reg = AncillaRegister(1, "anc")
    q_reg = QuantumRegister(1, "q")
    qc = QuantumCircuit(q_reg, anc_reg)
    qc.h(anc_reg[0])
    qc.cx(anc_reg[0], q_reg[0])
    print(qc)

    mqt_qc = qiskit_to_mqt(qc)
    print(mqt_qc)
    assert mqt_qc.num_data_qubits == 1
    assert mqt_qc.num_ancilla_qubits == 1

    qiskit_qc = mqt_to_qiskit(mqt_qc)
    print(qiskit_qc)
    assert qiskit_qc.num_qubits == 2
    assert len(qiskit_qc) == 2
    assert qc == qiskit_qc


def test_classical() -> None:
    """Test roundtrip of classical register."""
    c_reg = ClassicalRegister(1, "c")
    q_reg = QuantumRegister(1, "q")
    qc = QuantumCircuit(q_reg, c_reg)
    qc.h(q_reg[0])
    qc.barrier(q_reg[0])
    qc.measure(q_reg[0], c_reg[0])
    qc.reset(q_reg[0])
    print(qc)

    mqt_qc = qiskit_to_mqt(qc)
    print(mqt_qc)
    assert mqt_qc.num_qubits == 1
    assert mqt_qc.num_classical_bits == 1

    qiskit_qc = mqt_to_qiskit(mqt_qc)
    print(qiskit_qc)
    assert qiskit_qc.num_qubits == 1
    assert qiskit_qc.num_clbits == 1
    assert qc == qiskit_qc


def test_operations() -> None:
    """Test roundtrip of operations."""
    qc = QuantumCircuit(3)
    qc.id(0)
    qc.x(0)
    qc.cx(0, 1)
    qc.mcx([0, 1], 2)
    qc.y(0)
    qc.cy(0, 1)
    qc.z(0)
    qc.cz(0, 1)
    qc.h(0)
    qc.ch(0, 1)
    qc.s(0)
    qc.cs(0, 1)
    qc.sdg(0)
    qc.csdg(0, 1)
    qc.t(0)
    qc.tdg(0)
    qc.sx(0)
    qc.csx(0, 1)
    qc.sxdg(0)
    qc.rx(0.5, 0)
    qc.crx(0.5, 0, 1)
    qc.ry(0.5, 0)
    qc.cry(0.5, 0, 1)
    qc.rz(0.5, 0)
    qc.crz(0.5, 0, 1)
    qc.p(0.5, 0)
    qc.cp(0.5, 0, 1)
    qc.mcp(0.5, [0, 1], 2)
    qc.u(0.5, 0.5, 0.5, 0)
    qc.swap(0, 1)
    qc.iswap(0, 1)
    qc.dcx(0, 1)
    qc.ecr(0, 1)
    qc.rxx(0.5, 0, 1)
    qc.ryy(0.5, 0, 1)
    qc.rzz(0.5, 0, 1)
    qc.rzx(0.5, 0, 1)
    qc.append(U2Gate(0.5, 0.5), [0])
    qc.append(XXMinusYYGate(0.5, 0.5), [0, 1])
    qc.append(XXPlusYYGate(0.5, 0.5), [0, 1])
    print(qc)

    mqt_qc = qiskit_to_mqt(qc)
    print(mqt_qc)
    assert mqt_qc.num_qubits == 3
    assert mqt_qc.num_ops == len(qc)
    assert mqt_qc.is_variable_free()
    for op in mqt_qc:
        assert isinstance(op, StandardOperation)

    qiskit_qc = mqt_to_qiskit(mqt_qc)
    print(qiskit_qc)
    assert qiskit_qc.num_qubits == 3
    assert len(qiskit_qc) == len(qc)


def test_symbolic() -> None:
    """Test import of symbolic parameters."""
    qc = QuantumCircuit(1)
    lambda_ = Parameter("lambda")
    phi = Parameter("phi")
    theta = Parameter("theta")
    qc.rx(2 * theta + phi / 2 - lambda_ + 2, 0)
    print(qc)

    mqt_qc = qiskit_to_mqt(qc)
    print(mqt_qc)

    assert mqt_qc.num_qubits == 1
    assert mqt_qc.num_ops == 1
    assert mqt_qc[0].name.strip() == "rx"
    assert isinstance(mqt_qc[0], SymbolicOperation)
    assert isinstance(mqt_qc[0].get_parameter(0), Expression)
    expr = cast("Expression", mqt_qc[0].get_parameter(0))
    print(expr)
    assert expr.num_terms() == 3
    coeffs_vars = {(term.coefficient, term.variable.name) for term in expr.terms}
    assert (-1, "lambda") in coeffs_vars
    assert (0.5, "phi") in coeffs_vars
    assert (2, "theta") in coeffs_vars
    assert expr.constant == 2
    assert not mqt_qc.is_variable_free()

    with pytest.raises(NotImplementedError):
        mqt_to_qiskit(mqt_qc)

    qc = qc.assign_parameters({lambda_: 0, phi: 0, theta: 0})
    mqt_qc = qiskit_to_mqt(qc)
    assert mqt_qc.is_variable_free()
    assert mqt_qc[0].parameter[0] == 2

    qiskit_qc = mqt_to_qiskit(mqt_qc)
    print(qiskit_qc)
    assert qiskit_qc[0].operation.params[0] == 2


def test_symbolic_two_qubit() -> None:
    """Test import of symbolic two-qubit gate."""
    qc = QuantumCircuit(2)
    theta = Parameter("theta")
    qc.rxx(theta, 0, 1)
    print(qc)

    mqt_qc = qiskit_to_mqt(qc)
    print(mqt_qc)

    assert mqt_qc.num_qubits == 2
    assert mqt_qc.num_ops == 1
    assert mqt_qc[0].name.strip() == "rxx"
    assert isinstance(mqt_qc[0], SymbolicOperation)
    assert isinstance(mqt_qc[0].get_parameter(0), Expression)
    expr = cast("Expression", mqt_qc[0].get_parameter(0))
    assert expr.num_terms() == 1
    assert expr.constant == 0
    assert not mqt_qc.is_variable_free()

    with pytest.raises(NotImplementedError):
        mqt_to_qiskit(mqt_qc)


def test_if_else_operation() -> None:
    """Test import of if-else operation."""
    from qiskit.qasm3 import dumps

    then_qc = QuantumCircuit(1)
    then_qc.x(0)

    else_qc = QuantumCircuit(1)
    else_qc.y(0)

    qc = QuantumCircuit(1, 1)
    qc.if_else(
        (qc.clbits[0], True),
        then_qc,
        else_qc,
        [qc.qubits[0]],
        [],
    )
    print(qc)

    mqt_qc = qiskit_to_mqt(qc)
    print(mqt_qc)
    assert mqt_qc.num_qubits == 1
    assert mqt_qc.num_ops == 1
    if_else_operation = mqt_qc[0]
    assert isinstance(if_else_operation, IfElseOperation)
    assert if_else_operation.control_register is None
    assert if_else_operation.control_bit == 0
    assert if_else_operation.expected_value_bit is True
    assert if_else_operation.comparison_kind == ComparisonKind.eq
    then_operation = if_else_operation.then_operation
    assert isinstance(then_operation, StandardOperation)
    then_operation.type_ = OpType.x
    else_operation = if_else_operation.else_operation
    assert isinstance(else_operation, StandardOperation)
    else_operation.type_ = OpType.y

    qiskit_qc = mqt_to_qiskit(mqt_qc)
    print(qiskit_qc)

    qasm_input = dumps(qc)
    qasm_input = qasm_input.replace("if (c[0]) {\n", "")

    qasm_output = dumps(qiskit_qc)
    qasm_output = qasm_output.replace("if (c[0] == true) {\n", "")

    assert qasm_input == qasm_output


def test_trivial_initial_layout_multiple_registers() -> None:
    """Test that trivial initial layout import works with multiple registers.

    Correctly inferring the initial layout is not an easy task; especially when
    multiple registers are involved. This test checks that the initial layout
    is imported properly from a circuit with multiple registers that are not
    sorted alphabetically.
    """
    a = QuantumRegister(2, "a")
    b = QuantumRegister(2, "b")
    qc = QuantumCircuit(b, a)
    print(qc)

    initial_layout = [0, 1, 2, 3]
    qc_transpiled = transpile(qc, initial_layout=initial_layout)
    print(qc_transpiled)

    mqt_qc = qiskit_to_mqt(qc_transpiled)
    print(mqt_qc)
    for k, v in [(0, 0), (1, 1), (2, 2), (3, 3)]:
        assert mqt_qc.initial_layout[k] == v

    qiskit_qc = mqt_to_qiskit(mqt_qc, set_layout=True)
    print(qiskit_qc)
    mqt_qc_2 = qiskit_to_mqt(qiskit_qc)
    print(mqt_qc_2)
    assert mqt_qc.initial_layout == mqt_qc_2.initial_layout
    assert mqt_qc.output_permutation == mqt_qc_2.output_permutation


def test_non_trivial_initial_layout_multiple_registers() -> None:
    """Test that non-trivial initial layout import works with multiple registers."""
    a = QuantumRegister(2, "a")
    b = QuantumRegister(2, "b")
    qc = QuantumCircuit(b, a)
    print(qc)

    initial_layout = [3, 2, 1, 0]
    qc_transpiled = transpile(qc, initial_layout=initial_layout)
    print(qc_transpiled)

    mqt_qc = qiskit_to_mqt(qc_transpiled)
    print(mqt_qc)
    for k, v in enumerate(initial_layout):
        assert mqt_qc.initial_layout[k] == v
        assert mqt_qc.output_permutation[k] == v

    qiskit_qc = mqt_to_qiskit(mqt_qc, set_layout=True)
    print(qiskit_qc)
    mqt_qc_2 = qiskit_to_mqt(qiskit_qc)
    print(mqt_qc_2)
    assert mqt_qc.initial_layout == mqt_qc_2.initial_layout
    assert mqt_qc.output_permutation == mqt_qc_2.output_permutation


def test_non_symmetric_initial_layout_multiple_registers() -> None:
    """Test that non-symmetric initial layout import works with multiple registers."""
    a = QuantumRegister(2, "a")
    b = QuantumRegister(1, "b")
    qc = QuantumCircuit(b, a)
    print(qc)

    initial_layout = [1, 2, 0]
    qc_transpiled = transpile(qc, initial_layout=initial_layout)
    print(qc_transpiled)

    mqt_qc = qiskit_to_mqt(qc_transpiled)
    print(mqt_qc)
    for k, v in [(0, 2), (1, 0), (2, 1)]:
        assert mqt_qc.initial_layout[k] == v
        assert mqt_qc.output_permutation[k] == v

    qiskit_qc = mqt_to_qiskit(mqt_qc, set_layout=True)
    print(qiskit_qc)
    mqt_qc_2 = qiskit_to_mqt(qiskit_qc)
    print(mqt_qc_2)
    assert mqt_qc.initial_layout == mqt_qc_2.initial_layout
    assert mqt_qc.output_permutation == mqt_qc_2.output_permutation


def test_initial_layout_with_ancilla_in_front() -> None:
    """Test that initial layout import works with ancilla in front."""
    a = QuantumRegister(2, "a")
    b_anc = AncillaRegister(1, "b")
    qc = QuantumCircuit(b_anc, a)
    qc.x(0)
    print(qc)

    initial_layout = [0, 1, 2]
    qc_transpiled = transpile(qc, initial_layout=initial_layout)
    print(qc_transpiled)

    mqt_qc = qiskit_to_mqt(qc_transpiled)
    print(mqt_qc)
    for k, v in [(0, 0), (1, 1), (2, 2)]:
        assert mqt_qc.initial_layout[k] == v
    assert mqt_qc.num_ancilla_qubits == 1
    assert mqt_qc.is_circuit_qubit_ancillary(0)

    qiskit_qc = mqt_to_qiskit(mqt_qc, set_layout=True)
    print(qiskit_qc)
    mqt_qc_2 = qiskit_to_mqt(qiskit_qc)
    print(mqt_qc_2)
    assert mqt_qc.initial_layout == mqt_qc_2.initial_layout
    assert mqt_qc.output_permutation == mqt_qc_2.output_permutation


def test_initial_layout_with_ancilla_in_back() -> None:
    """Test that initial layout import works with ancilla in back."""
    a = QuantumRegister(2, "a")
    b_anc = AncillaRegister(1, "b")
    qc = QuantumCircuit(a, b_anc)
    qc.x(2)
    print(qc)

    initial_layout = [0, 1, 2]
    qc_transpiled = transpile(qc, initial_layout=initial_layout)
    print(qc_transpiled)

    mqt_qc = qiskit_to_mqt(qc_transpiled)
    print(mqt_qc)
    for k, v in [(0, 0), (1, 1), (2, 2)]:
        assert mqt_qc.initial_layout[k] == v
    assert mqt_qc.num_ancilla_qubits == 1
    assert mqt_qc.is_circuit_qubit_ancillary(2)

    qiskit_qc = mqt_to_qiskit(mqt_qc, set_layout=True)
    print(qiskit_qc)
    mqt_qc_2 = qiskit_to_mqt(qiskit_qc)
    print(mqt_qc_2)
    assert mqt_qc.initial_layout == mqt_qc_2.initial_layout
    assert mqt_qc.output_permutation == mqt_qc_2.output_permutation


def test_symbolic_global_phase() -> None:
    """Test whether symbolic global phase works properly."""
    qc = QuantumCircuit(1)
    theta = Parameter("theta")
    qc.global_phase = theta

    with pytest.warns(RuntimeWarning):
        mqt_qc = qiskit_to_mqt(qc)

    assert mqt_qc.global_phase == 0


def test_final_layout_without_permutation() -> None:
    """Test that the output permutation remains the same as the initial layout when routing is not performed."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    print(qc)

    initial_layout = [1, 2, 0]
    seed = 123
    qc_transpiled = transpile(qc, initial_layout=initial_layout, seed_transpiler=seed)
    print(qc_transpiled)

    mqt_qc = qiskit_to_mqt(qc_transpiled)
    print(mqt_qc)
    assert mqt_qc.initial_layout == {0: 2, 1: 0, 2: 1}
    assert mqt_qc.output_permutation == mqt_qc.initial_layout

    qiskit_qc = mqt_to_qiskit(mqt_qc, set_layout=True)
    print(qiskit_qc)
    mqt_qc_2 = qiskit_to_mqt(qiskit_qc)
    print(mqt_qc_2)
    assert mqt_qc.initial_layout == mqt_qc_2.initial_layout
    assert mqt_qc.output_permutation == mqt_qc_2.output_permutation


# test fixture for the backend using GenericBackendV2
@pytest.fixture
def backend() -> GenericBackendV2:
    """Fixture for the backend using GenericBackendV2.

    Returns:
        A generic five-qubit backend to be used for compilation.
    """
    return GenericBackendV2(
        num_qubits=5,
        basis_gates=["id", "rz", "sx", "x", "cx", "reset"],
        coupling_map=[[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3]],
    )


def test_final_layout_with_permutation(backend: GenericBackendV2) -> None:
    """Test that the output permutation gets updated correctly when routing is performed."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(1, 0)
    qc.cx(1, 2)
    qc.measure_all()
    print(qc)

    initial_layout = [1, 0, 3]
    seed = 123
    qc_transpiled = transpile(qc, backend, initial_layout=initial_layout, seed_transpiler=seed)
    print(qc_transpiled)

    final_index_layout = qc_transpiled.layout.final_index_layout()
    mqt_qc = qiskit_to_mqt(qc_transpiled)
    print(mqt_qc)
    # Check the initial layout is properly translated
    assert mqt_qc.initial_layout == {0: 1, 1: 0, 3: 2, 2: 3, 4: 4}
    # Check initialize_io_mapping doesn't change the final_layout
    assert mqt_qc.output_permutation == dict(enumerate(final_index_layout))

    qiskit_qc = mqt_to_qiskit(mqt_qc, set_layout=True)
    print(qiskit_qc)
    mqt_qc_2 = qiskit_to_mqt(qiskit_qc)
    print(mqt_qc_2)
    assert mqt_qc.initial_layout == mqt_qc_2.initial_layout
    assert mqt_qc.output_permutation == mqt_qc_2.output_permutation


def test_final_layout_with_permutation_ancilla_in_front_and_back(backend: GenericBackendV2) -> None:
    """Test that permutation update is correct with multiple registers and ancilla qubits."""
    e = QuantumRegister(2, "e")
    f_anc = AncillaRegister(1, "f")
    b_anc = AncillaRegister(2, "b")
    qc = QuantumCircuit(f_anc, e, b_anc)
    qc.h(0)
    qc.cx(1, 0)
    qc.cx(1, 2)
    qc.measure_all()
    print(qc)

    initial_layout = [1, 0, 3, 2, 4]
    seed = 123
    qc_transpiled = transpile(qc, backend, initial_layout=initial_layout, seed_transpiler=seed)
    print(qc_transpiled)

    routing_permutation = qc_transpiled.layout.routing_permutation()
    mqt_qc = qiskit_to_mqt(qc_transpiled)
    print(mqt_qc)
    # Check the initial layout is properly translated
    assert mqt_qc.initial_layout == {0: 1, 1: 0, 3: 2, 2: 3, 4: 4}
    # Check that output_permutation matches the result of applying the routing permutation to input_layout
    assert mqt_qc.output_permutation == {idx: routing_permutation[key] for idx, key in enumerate(initial_layout)}

    qiskit_qc = mqt_to_qiskit(mqt_qc, set_layout=True)
    print(qiskit_qc)
    mqt_qc_2 = qiskit_to_mqt(qiskit_qc)
    print(mqt_qc_2)
    assert mqt_qc.initial_layout == mqt_qc_2.initial_layout
    assert mqt_qc.output_permutation == mqt_qc_2.output_permutation


def test_empty_quantum_register() -> None:
    """Test an empty quantum register (valid in Qiskit) is handled correctly."""
    qr = QuantumRegister(0)
    qc = QuantumCircuit(qr)
    mqt_qc = qiskit_to_mqt(qc)
    assert mqt_qc.num_qubits == 0
    assert mqt_qc.num_ops == 0


def test_empty_classical_register() -> None:
    """Test an empty classical register (valid in Qiskit) is handled correctly."""
    cr = ClassicalRegister(0)
    qc = QuantumCircuit(cr)
    mqt_qc = qiskit_to_mqt(qc)
    assert mqt_qc.num_classical_bits == 0
    assert mqt_qc.num_ops == 0
