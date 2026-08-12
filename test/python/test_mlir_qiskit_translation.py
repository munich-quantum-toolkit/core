# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Behavioral tests for Qiskit circuit import and export."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
import qiskit
from packaging.version import Version
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import (
    AnnotatedOperation,
    Clbit,
    ControlModifier,
    Gate,
    InverseModifier,
    Parameter,
    PowerModifier,
    Qubit,
    library,
)
from qiskit.circuit.classical import expr, types
from qiskit.quantum_info import Operator

from mqt.core.mlir import QCProgram, compile_program
from mqt.core.plugins.qiskit import qiskit_to_mqt

installed_qiskit = Version(qiskit.__version__)
candidate_version = os.environ.get("MQT_QISKIT_TEST_CANDIDATE_VERSION")
if not (Version("2.5.0") <= installed_qiskit < Version("2.6.0") or qiskit.__version__ == candidate_version):
    pytest.skip(
        f"Qiskit circuit translation tests require Qiskit 2.5.x (installed: {qiskit.__version__})",
        allow_module_level=True,
    )


STANDARD_GATES = (
    library.IGate(),
    library.XGate(),
    library.YGate(),
    library.ZGate(),
    library.HGate(),
    library.SGate(),
    library.SdgGate(),
    library.TGate(),
    library.TdgGate(),
    library.SXGate(),
    library.SXdgGate(),
    library.PhaseGate(0.1),
    library.RXGate(0.2),
    library.RYGate(0.3),
    library.RZGate(0.4),
    library.RGate(0.5, 0.6),
    library.UGate(0.1, 0.2, 0.3),
    library.U1Gate(0.2),
    library.U2Gate(0.2, 0.3),
    library.U3Gate(0.2, 0.3, 0.4),
    library.CXGate(),
    library.CYGate(),
    library.CZGate(),
    library.CHGate(),
    library.CPhaseGate(0.4),
    library.CRXGate(0.4),
    library.CRYGate(0.4),
    library.CRZGate(0.4),
    library.CUGate(0.1, 0.2, 0.3, 0.4),
    library.CU1Gate(0.2),
    library.CU3Gate(0.1, 0.2, 0.3),
    library.SwapGate(),
    library.CSwapGate(),
    library.iSwapGate(),
    library.DCXGate(),
    library.ECRGate(),
    library.RXXGate(0.5),
    library.RYYGate(0.5),
    library.RZXGate(0.5),
    library.RZZGate(0.5),
    library.XXPlusYYGate(0.6, 0.7),
    library.XXMinusYYGate(0.6, 0.7),
    library.CCXGate(),
    library.CCZGate(),
    library.RCCXGate(),
    library.C3XGate(),
    library.C3SXGate(),
    library.RC3XGate(),
)


@pytest.mark.parametrize("gate", STANDARD_GATES, ids=lambda gate: gate.name)
def test_standard_gates_round_trip(gate: Gate) -> None:
    """Translate each supported gate family in both directions."""
    circuit = QuantumCircuit(gate.num_qubits)
    circuit.append(gate, range(gate.num_qubits))

    restored = QCProgram.from_qiskit(circuit).to_qiskit()

    assert np.allclose(Operator(restored).data, Operator(circuit).data)


@pytest.mark.parametrize(
    ("modifier", "expected"),
    [
        (InverseModifier(), "qc.inv"),
        (PowerModifier(0.5), "qc.pow"),
        (ControlModifier(2), "qc.ctrl"),
    ],
    ids=["inverse", "power", "control"],
)
def test_numeric_modifiers_are_imported(
    modifier: InverseModifier | PowerModifier | ControlModifier, expected: str
) -> None:
    """Represent supported numeric Qiskit modifiers in QC."""
    circuit = QuantumCircuit(AnnotatedOperation(library.RYGate(0.25), modifier).num_qubits)
    circuit.append(AnnotatedOperation(library.RYGate(0.25), modifier), circuit.qubits)

    program = QCProgram.from_qiskit(circuit)

    assert expected in program.ir


def test_excessively_nested_annotated_operation_is_rejected() -> None:
    """Bound annotated-operation traversal before recursive normalization."""
    operation: Gate | AnnotatedOperation = library.XGate()
    for _ in range(64):
        operation = AnnotatedOperation(operation, InverseModifier())
    circuit = QuantumCircuit(1)
    circuit.append(operation, [0])

    with pytest.raises(RuntimeError, match="annotated operations exceed the nesting limit of 64"):
        QCProgram.from_qiskit(circuit)


@pytest.mark.parametrize("controls", [1, 2, 3, 4, 5])
def test_variable_arity_mcx_is_imported(controls: int) -> None:
    """Normalize each MCX arity to X with its actual control count."""
    gate = library.MCXGate(controls)
    circuit = QuantumCircuit(gate.num_qubits)
    circuit.append(gate, circuit.qubits)

    program = QCProgram.from_qiskit(circuit)
    control = next(line for line in program.ir.splitlines() if "qc.ctrl(" in line)

    assert control.split(") targets", maxsplit=1)[0].count("%") == controls
    assert "qc.x" in program.ir


@pytest.mark.parametrize(
    "modifier",
    [InverseModifier(), PowerModifier(-1.0), ControlModifier(1)],
    ids=["inverse", "inverse-power", "control"],
)
def test_constructible_numeric_modifiers_round_trip(
    modifier: InverseModifier | PowerModifier | ControlModifier,
) -> None:
    """Export modifiers that have a Qiskit standard-gate equivalent."""
    operation = AnnotatedOperation(library.RYGate(0.25), modifier)
    circuit = QuantumCircuit(operation.num_qubits)
    circuit.append(operation, circuit.qubits)

    restored = QCProgram.from_qiskit(circuit).to_qiskit()

    assert np.allclose(Operator(restored).data, Operator(circuit).data)


def test_flat_circuit_round_trip_preserves_supported_metadata() -> None:
    """Preserve operations, phase, and canonical register names."""
    qreg = QuantumRegister(2, "input")
    creg = ClassicalRegister(2, "output")
    circuit = QuantumCircuit(qreg, creg, global_phase=0.125)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.reset(0)
    circuit.barrier()
    circuit.measure(range(2), range(2))

    program = QCProgram.from_qiskit(circuit)
    restored = program.to_qiskit()

    assert 'mqt.qubit_register_name = "input"' in program.ir
    assert 'mqt.classical_register_name = "output"' in program.ir
    assert restored.global_phase == pytest.approx(0.125)
    assert [(reg.name, len(reg)) for reg in restored.qregs] == [("input", 2)]
    assert [(reg.name, len(reg)) for reg in restored.cregs] == [("output", 2)]
    assert [item.operation.name for item in restored.data] == [
        "h",
        "cx",
        "reset",
        "barrier",
        "measure",
        "measure",
    ]


def test_layout_is_accepted_and_ignored() -> None:
    """Import laid-out operations without retaining transpiler metadata."""
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    laid_out = transpile(
        circuit,
        coupling_map=[[0, 1]],
        initial_layout=[1, 0],
        optimization_level=0,
    )
    assert laid_out.layout is not None

    program = QCProgram.from_qiskit(laid_out)
    restored = program.to_qiskit()

    assert "qc.ctrl" in program.ir
    assert "layout" not in program.ir
    assert [item.operation.name for item in restored.data] == [item.operation.name for item in laid_out.data]
    assert np.allclose(Operator(restored).data, Operator(laid_out).data)


def test_nested_numeric_custom_definitions_are_inlined() -> None:
    """Bind numeric call parameters and recursively inline definitions."""
    theta = Parameter("theta")
    definition = QuantumCircuit(1)
    definition.rx(theta, 0)
    inner = definition.to_gate(label="inner")
    middle_definition = QuantumCircuit(1)
    middle_definition.append(inner, [0])
    outer = middle_definition.to_gate(label="outer")
    circuit = QuantumCircuit(1)
    circuit.append(outer, [0])
    circuit.assign_parameters({theta: 0.25}, inplace=True)

    program = QCProgram.from_qiskit(circuit)

    assert "qc.rx" in program.ir
    assert "2.500000e-01" in program.ir
    assert circuit.parameters == set()


def test_ambiguous_custom_parameter_binding_is_rejected() -> None:
    """Do not infer formal parameter order from Qiskit's sorted parameter set."""
    z = Parameter("z")
    a = Parameter("a")
    definition = QuantumCircuit(1)
    definition.rz(z, 0)
    definition.rx(a, 0)
    gate = Gate("ambiguous", 1, [0.1, 0.2])
    gate.definition = definition
    circuit = QuantumCircuit(1)
    circuit.append(gate, [0])

    with pytest.raises(RuntimeError, match="must be numerically bound before import"):
        QCProgram.from_qiskit(circuit)


def test_custom_definition_uses_call_parameter_order_after_binding() -> None:
    """Preserve explicit custom-gate parameter order after Qiskit binds it."""
    z = Parameter("z")
    a = Parameter("a")
    definition = QuantumCircuit(1)
    definition.rz(z, 0)
    definition.rx(a, 0)
    gate = Gate("ordered", 1, [z, a])
    gate.definition = definition
    circuit = QuantumCircuit(1)
    circuit.append(gate, [0])
    circuit.assign_parameters({z: 0.1, a: 0.2}, inplace=True)

    restored = QCProgram.from_qiskit(circuit).to_qiskit()

    assert np.allclose(Operator(restored).data, Operator(circuit).data)


@pytest.mark.parametrize("modifier", [InverseModifier(), PowerModifier(0.5), ControlModifier(1)])
def test_modified_custom_definitions_are_rejected(
    modifier: InverseModifier | PowerModifier | ControlModifier,
) -> None:
    """Reject modifiers whose semantics cannot be preserved while inlining."""
    definition = QuantumCircuit(1)
    definition.h(0)
    custom = definition.to_gate(label="custom")
    operation = AnnotatedOperation(custom, modifier)
    circuit = QuantumCircuit(operation.num_qubits)
    circuit.append(operation, circuit.qubits)

    with pytest.raises(RuntimeError, match="does not support modifiers on custom instructions"):
        QCProgram.from_qiskit(circuit)


def test_definition_failures_are_reported_before_import() -> None:
    """Reject missing and arity-mismatched instruction definitions."""
    missing = QuantumCircuit(1)
    missing.append(Gate("missing", 1, []), [0])
    with pytest.raises(RuntimeError, match="no circuit definition"):
        QCProgram.from_qiskit(missing)

    bad_arity = Gate("bad_arity", 2, [])
    bad_arity.definition = QuantumCircuit(1)
    circuit = QuantumCircuit(2)
    circuit.append(bad_arity, [0, 1])
    with pytest.raises(RuntimeError, match="does not match its definition arity"):
        QCProgram.from_qiskit(circuit)


def test_cyclic_and_excessively_nested_definitions_are_rejected() -> None:
    """Bound recursive definition traversal by cycles and depth."""
    cyclic = Gate("cyclic", 1, [])
    cyclic_definition = QuantumCircuit(1)
    cyclic_definition.append(cyclic, [0])
    cyclic.definition = cyclic_definition
    circuit = QuantumCircuit(1)
    circuit.append(cyclic, [0])
    with pytest.raises(RuntimeError, match="contain a cycle"):
        QCProgram.from_qiskit(circuit)

    leaf_definition = QuantumCircuit(1)
    leaf_definition.h(0)
    nested: Gate = leaf_definition.to_gate(label="level_0")
    for level in range(65):
        next_definition = QuantumCircuit(1)
        next_definition.append(nested, [0])
        nested = next_definition.to_gate(label=f"level_{level + 1}")
    too_deep = QuantumCircuit(1)
    too_deep.append(nested, [0])
    with pytest.raises(RuntimeError, match="nesting limit of 64"):
        QCProgram.from_qiskit(too_deep)


def test_exponential_definition_expansion_is_rejected_by_budget() -> None:
    """Count repeated definitions without materializing their full expansion."""
    leaf_definition = QuantumCircuit(1)
    leaf_definition.h(0)
    nested = Gate("leaf", 1, [])
    nested.definition = leaf_definition
    for level in range(22):
        definition = QuantumCircuit(1)
        definition.append(nested, [0])
        definition.append(nested, [0])
        nested = Gate(f"branch_{level}", 1, [])
        nested.definition = definition
    circuit = QuantumCircuit(1)
    circuit.append(nested, [0])

    with pytest.raises(RuntimeError, match="expansion exceeds 10000000 operations"):
        QCProgram.from_qiskit(circuit)


def test_value_list_loop_expansion_counts_each_iteration() -> None:
    """Apply the expansion budget to every statically unrolled loop value."""
    leaf_definition = QuantumCircuit(1)
    leaf_definition.h(0)
    nested = Gate("leaf", 1, [])
    nested.definition = leaf_definition
    for level in range(20):
        definition = QuantumCircuit(1)
        definition.append(nested, [0])
        definition.append(nested, [0])
        nested = Gate(f"branch_{level}", 1, [])
        nested.definition = definition
    circuit = QuantumCircuit(1)
    with circuit.for_loop([0, 2, 5, 9], None, None, None, None, label=None):
        circuit.append(nested, [0])

    with pytest.raises(RuntimeError, match="expansion exceeds 10000000 operations"):
        QCProgram.from_qiskit(circuit)


def test_rejections_do_not_modify_source_circuits() -> None:
    """Reject unsupported parameters, inputs, and unitaries without mutation."""
    theta = Parameter("theta")
    symbolic = QuantumCircuit(1)
    symbolic.rx(theta, 0)
    symbolic_data = list(symbolic.data)
    with pytest.raises(RuntimeError, match="free symbolic parameter 'theta'"):
        QCProgram.from_qiskit(symbolic)
    assert list(symbolic.data) == symbolic_data
    assert symbolic.parameters == {theta}

    value = expr.Var.new("value", types.Uint(8))
    runtime_input = QuantumCircuit(1, inputs=[value])
    with runtime_input.if_test(expr.equal(value, 1)):
        runtime_input.x(0)
    input_data = list(runtime_input.data)
    with pytest.raises(RuntimeError, match="standalone classical variables"):
        QCProgram.from_qiskit(runtime_input)
    assert list(runtime_input.data) == input_data

    unitary = QuantumCircuit(1)
    unitary.unitary(np.eye(2), [0])
    unitary_data = list(unitary.data)
    with pytest.raises(RuntimeError, match="does not support arbitrary unitaries"):
        QCProgram.from_qiskit(unitary)
    assert list(unitary.data) == unitary_data


@pytest.mark.parametrize("resource", ["quantum", "classical"])
@pytest.mark.parametrize("layout", ["alias", "interleaved"])
def test_noncanonical_register_membership_is_rejected(resource: str, layout: str) -> None:
    """Reject aliases and interleaving for both resource kinds."""
    bit_type = Qubit if resource == "quantum" else Clbit
    register_type = QuantumRegister if resource == "quantum" else ClassicalRegister
    bits = [bit_type() for _ in range(3)]
    if layout == "alias":
        first = register_type(bits=bits[:2], name="first")
        second = register_type(bits=bits[1:], name="second")
    else:
        first = register_type(bits=[bits[0], bits[2]], name="first")
        second = register_type(bits=[bits[1]], name="second")
    circuit = QuantumCircuit()
    circuit.add_bits(bits)
    circuit.add_register(first)
    circuit.add_register(second)
    if resource == "quantum":
        circuit.x(0)

    with pytest.raises(
        RuntimeError,
        match=rf"disjoint {resource} register|loose {resource} bits before contiguous registers",
    ):
        QCProgram.from_qiskit(circuit)


def test_nested_structured_control_and_bound_loop_parameter() -> None:
    """Import nested control flow while keeping induction values lexical."""
    circuit = QuantumCircuit(2, 2)
    with circuit.for_loop(range(1, 5, 2), None, None, None, None, label=None) as iteration:
        circuit.rx(iteration, 0)
        with circuit.if_test((circuit.clbits[0], False)):
            circuit.cx(0, 1)
    with circuit.while_loop((circuit.cregs[0], 0), None, None, None, label=None):
        circuit.measure(0, 0)
    with circuit.switch(circuit.cregs[0], None, None, None, label=None) as case:
        with case(0, 1):
            circuit.x(0)
        with case(case.DEFAULT):
            circuit.z(1)

    program = compile_program(circuit)

    assert "scf.for" in program.ir
    assert "scf.if" in program.ir
    assert "scf.while" in program.ir
    assert "scf.index_switch" in program.ir
    with pytest.raises(RuntimeError, match="cannot construct structured control flow"):
        program.to_qiskit()


@pytest.mark.parametrize(
    ("condition", "operation"),
    [
        (expr.logic_and(expr.equal(1, 1), expr.equal(0, 1)), "arith.andi"),
        (expr.equal(expr.bit_xor(expr.lift(2, types.Uint(8)), 3), 5), "arith.xori"),
        (expr.less(expr.add(expr.lift(2, types.Uint(8)), 1), 8), "arith.addi"),
        (
            expr.greater(expr.cast(expr.lift(2, types.Uint(8)), types.Float()), 0.5),
            "arith.uitofp",
        ),
        (expr.greater(expr.negate(expr.lift(0.5, types.Float())), -1.0), "arith.negf"),
    ],
)
def test_bool_uint_and_float_expressions(condition: expr.Expr, operation: str) -> None:
    """Lower representative constant classical expressions."""
    circuit = QuantumCircuit(1)
    with circuit.if_test(condition):
        circuit.x(0)

    program = QCProgram.from_qiskit(circuit)

    assert operation in program.ir


def test_excessively_nested_classical_expression_is_rejected() -> None:
    """Bound native normalization before recursive expression traversal."""
    condition: expr.Expr = expr.equal(1, 1)
    for _ in range(64):
        condition = expr.logic_not(condition)
    circuit = QuantumCircuit(1)
    with circuit.if_test(condition):
        circuit.x(0)

    with pytest.raises(RuntimeError, match="expressions exceed the nesting limit of 64"):
        QCProgram.from_qiskit(circuit)


def test_excessively_nested_control_flow_is_rejected() -> None:
    """Bound control-flow traversal independently of definition depth."""
    body = QuantumCircuit(1, 1)
    body.x(0)
    for _ in range(65):
        outer = QuantumCircuit(1, 1)
        outer.if_test((outer.clbits[0], False), body, outer.qubits, outer.clbits)
        body = outer

    with pytest.raises(RuntimeError, match="control flow exceeds the nesting limit of 64"):
        QCProgram.from_qiskit(body)


def test_flat_export_rejects_symbolic_inputs() -> None:
    """Reject program inputs before allocating an output circuit."""
    symbolic = QCProgram.from_mlir_str(
        """module {
  func.func @main(%theta: f64) attributes {passthrough = ["entry_point"]} {
    %q = qc.alloc : !qc.qubit
    qc.rx(%theta) %q : !qc.qubit
    qc.dealloc %q : !qc.qubit
    return
  }
}
"""
    )
    with pytest.raises(RuntimeError, match="symbolic or runtime inputs"):
        symbolic.to_qiskit()


def test_unknown_version_is_rejected_without_affecting_existing_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep direct version dispatch independent of existing conversion."""
    monkeypatch.setattr(qiskit, "__version__", "2.6.0")
    with pytest.raises(RuntimeError, match=r"installed version '2\.6\.0'.*>=2\.5\.0,<2\.6\.0"):
        QCProgram.from_qiskit(QuantumCircuit(1))

    assert qiskit_to_mqt(QuantumCircuit(1)).num_qubits == 1


def test_mlir_binding_import_does_not_import_qiskit() -> None:
    """Keep importing the MLIR extension independent of optional Qiskit."""
    script = """
import importlib.abc
import sys

class RejectQiskit(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "qiskit" or fullname.startswith("qiskit."):
            raise AssertionError("MLIR binding attempted to import Qiskit")
        return None

sys.meta_path.insert(0, RejectQiskit())
import mqt.core.mlir
assert "qiskit" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)  # ruff: ignore[subprocess-without-shell-equals-true]
