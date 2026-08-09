# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Focused tests for the native Qiskit C-API compiler bridge."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest
import qiskit
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Clbit, Gate, Parameter, Qubit, library
from qiskit.circuit.classical import expr, types
from qiskit.quantum_info import Operator

from mqt.core import mlir as mlir_module
from mqt.core.mlir import QCProgram, compile_program
from mqt.core.plugins.qiskit import qiskit_to_mqt

if TYPE_CHECKING:
    from collections.abc import Callable

availability_name = "_" + "qiskit_compiler_bridge_available"
requires_bridge = pytest.mark.skipif(
    not getattr(mlir_module, availability_name)(),
    reason="installed Qiskit has no registered compiler bridge adapter",
)


@requires_bridge
@pytest.mark.parametrize(
    "gate",
    [
        library.GlobalPhaseGate(0.1),
        library.HGate(),
        library.SXdgGate(),
        library.RGate(0.1, 0.2),
        library.UGate(0.1, 0.2, 0.3),
        library.U1Gate(0.1),
        library.U3Gate(0.1, 0.2, 0.3),
        library.CXGate(),
        library.CPhaseGate(0.4),
        library.CUGate(0.1, 0.2, 0.3, 0.4),
        library.CU1Gate(0.1),
        library.CU3Gate(0.1, 0.2, 0.3),
        library.SwapGate(),
        library.CSwapGate(),
        library.RXXGate(0.5),
        library.XXPlusYYGate(0.6, 0.7),
        library.CCXGate(),
        library.CCZGate(),
        library.RCCXGate(),
        library.C3XGate(),
        library.C3SXGate(),
        library.RC3XGate(),
    ],
    ids=lambda gate: gate.name,
)
def test_qiskit_bridge_standard_gate_round_trip(gate: Gate) -> None:
    """Round-trip representative entries from every standard-gate family."""
    circuit = QuantumCircuit(gate.num_qubits)
    circuit.append(gate, range(gate.num_qubits))

    program = QCProgram.from_qiskit(circuit)
    restored = program.to_qiskit()

    assert program.is_valid
    assert np.allclose(Operator(restored).data, Operator(circuit).data)


@requires_bridge
@pytest.mark.parametrize("name", ["angle-1", "angle with space", "1.5", "θ"])
def test_qiskit_bridge_preserves_arbitrary_bare_parameter_names(name: str) -> None:
    """Classify bare symbols structurally instead of parsing their display text."""
    parameter = Parameter(name)
    circuit = QuantumCircuit(1)
    circuit.rx(parameter, 0)
    circuit.ry(parameter, 0)

    program = QCProgram.from_qiskit(circuit)
    restored = program.to_qiskit()

    assert {value.name for value in restored.parameters} == {name}
    assert restored.data[0].operation.params[0] == restored.data[1].operation.params[0]
    assert "mqt.qiskit.parameter_name" in program.ir


@requires_bridge
def test_qiskit_bridge_preserves_flat_circuit_metadata_and_unitary() -> None:
    """Preserve symbols, phase, registers, measurement, reset, and a dense unitary."""
    theta = Parameter("theta")
    circuit = QuantumCircuit(2, 2, global_phase=0.125)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.rz(theta, 1)
    circuit.unitary(np.diag([1.0, 1.0j]), [0])
    circuit.reset(0)
    circuit.barrier()
    circuit.measure(range(2), range(2))

    program = QCProgram.from_qiskit(circuit)
    restored = program.to_qiskit()

    assert "qc.unitary" in program.ir
    assert 'mqt.qiskit.parameter_name = "theta"' in program.ir
    assert restored.global_phase == pytest.approx(0.125)
    assert {parameter.name for parameter in restored.parameters} == {"theta"}
    assert [(reg.name, len(reg)) for reg in restored.qregs] == [("q", 2)]
    assert [(reg.name, len(reg)) for reg in restored.cregs] == [("c", 2)]
    assert [instruction.operation.name for instruction in restored.data] == [
        "h",
        "cx",
        "rz",
        "unitary",
        "reset",
        "barrier",
        "measure",
        "measure",
    ]
    assert program.is_valid


@requires_bridge
def test_qiskit_bridge_round_trips_ordered_multi_qubit_unitary() -> None:
    """Preserve matrix layout and explicit Qiskit operand order."""
    matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 0, 1j],
            [0, 0, 1, 0],
            [0, 1j, 0, 0],
        ],
        dtype=complex,
    )
    circuit = QuantumCircuit(2)
    circuit.unitary(matrix, [1, 0])

    restored = QCProgram.from_qiskit(circuit).to_qiskit()

    assert np.allclose(Operator(restored).data, Operator(circuit).data)
    assert [restored.find_bit(bit).index for bit in restored.data[0].qubits] == [1, 0]


@requires_bridge
def test_qiskit_bridge_exports_constructible_qc_modifiers() -> None:
    """Flatten supported QC modifiers into standard QkCircuit gates."""
    program = QCProgram.from_mlir_str(
        """module {
  func.func @main() attributes {passthrough = ["entry_point"]} {
    %register = memref.alloc() : memref<2x!qc.qubit>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %theta = arith.constant 0.1234567890123456 : f64
    %minus_one = arith.constant -1.0 : f64
    %q0 = memref.load %register[%c0] : memref<2x!qc.qubit>
    %q1 = memref.load %register[%c1] : memref<2x!qc.qubit>
    qc.inv (%arg0 = %q0) {
      qc.rx(%theta) %arg0 : !qc.qubit
      qc.yield
    } : !qc.qubit
    qc.pow(%minus_one) (%arg0 = %q0) {
      qc.s %arg0 : !qc.qubit
      qc.yield
    } : !qc.qubit
    qc.ctrl(%q0) targets (%arg0 = %q1) {
      qc.y %arg0 : !qc.qubit
      qc.yield
    } : {!qc.qubit}, {!qc.qubit}
    memref.dealloc %register : memref<2x!qc.qubit>
    return
  }
}
"""
    )

    circuit = program.to_qiskit()

    assert [instruction.operation.name for instruction in circuit.data] == [
        "rx",
        "sdg",
        "cy",
    ]
    assert float(circuit.data[0].operation.params[0]) == pytest.approx(-0.1234567890123456)


@requires_bridge
def test_qiskit_bridge_exports_dynamic_qc_allocations() -> None:
    """Represent flat dynamic QC qubits as loose QkCircuit bits."""
    program = QCProgram.from_mlir_str(
        """module {
  func.func @main() attributes {passthrough = ["entry_point"]} {
    %q = qc.alloc : !qc.qubit
    qc.h %q : !qc.qubit
    qc.dealloc %q : !qc.qubit
    return
  }
}
"""
    )

    circuit = program.to_qiskit()

    assert circuit.num_qubits == 1
    assert [instruction.operation.name for instruction in circuit.data] == ["h"]


@requires_bridge
def test_qiskit_bridge_preserves_overlapping_register_membership_on_import() -> None:
    """Record loose bits and overlapping named-register views without duplicating bits."""
    qubits = [Qubit() for _ in range(3)]
    first = QuantumRegister(bits=qubits[:2], name="first")
    overlap = QuantumRegister(bits=qubits[1:], name="overlap")
    classical = [Clbit() for _ in range(2)]
    output = ClassicalRegister(bits=classical, name="output")
    circuit = QuantumCircuit(qubits, classical, first, overlap, output)
    circuit.cx(qubits[0], qubits[2])

    program = QCProgram.from_qiskit(circuit)

    assert 'name = "first"' in program.ir
    assert 'name = "overlap"' in program.ir
    assert "bits = array<i32: 0, 1>" in program.ir
    assert "bits = array<i32: 1, 2>" in program.ir
    with pytest.raises(RuntimeError, match="disjoint register membership"):
        program.to_qiskit()


@requires_bridge
def test_qiskit_bridge_imports_nested_structured_control() -> None:
    """Lower nested for/if, while, and switch blocks with mapped resources."""
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

    assert program.ir.count("scf.for") == 1
    assert program.ir.count("scf.if") == 1
    assert program.ir.count("scf.while") == 1
    assert program.ir.count("scf.index_switch") == 1
    assert "qc.ctrl" in program.ir
    with pytest.raises(RuntimeError, match="cannot construct structured control flow"):
        program.to_qiskit()


@requires_bridge
@pytest.mark.parametrize(
    ("variable_type", "condition", "argument_type", "operation"),
    [
        (
            types.Bool(),
            lambda value: expr.logic_and(value, expr.lift(bool(1))),
            "i1",
            "arith.andi",
        ),
        (
            types.Uint(8),
            lambda value: expr.equal(expr.bit_xor(value, 3), 5),
            "i8",
            "arith.xori",
        ),
        (
            types.Uint(8),
            lambda value: expr.equal(expr.bit_not(value), 3),
            "i8",
            "arith.xori",
        ),
        (
            types.Uint(8),
            lambda value: expr.less(expr.add(value, 1), 8),
            "i8",
            "arith.addi",
        ),
        (
            types.Uint(8),
            lambda value: expr.equal(expr.shift_left(value, 1), 4),
            "i8",
            "arith.shli",
        ),
        (types.Uint(8), lambda value: expr.index(value, 2), "i8", "arith.shrui"),
        (
            types.Uint(8),
            lambda value: expr.greater(expr.cast(value, types.Float()), 0.5),
            "i8",
            "arith.uitofp",
        ),
        (
            types.Float(),
            lambda value: expr.greater(expr.negate(value), -1.0),
            "f64",
            "arith.negf",
        ),
    ],
)
def test_qiskit_bridge_imports_supported_classical_expressions(
    variable_type: types.Type,
    condition: Callable[[expr.Var], expr.Expr],
    argument_type: str,
    operation: str,
) -> None:
    """Lower typed, C-API-inspectable expression trees to arith operations."""
    value = expr.Var.new("value", variable_type)
    circuit = QuantumCircuit(1, inputs=[value])
    with circuit.if_test(condition(value)):
        circuit.x(0)

    program = QCProgram.from_qiskit(circuit)

    assert f"%arg0: {argument_type}" in program.ir
    assert operation in program.ir


@requires_bridge
def test_qiskit_bridge_rejects_unsupported_constructs() -> None:
    """Reject custom operations, delays, and composite parameter expressions explicitly."""
    custom = QuantumCircuit(1)
    custom.append(Gate("custom", 1, []), [0])
    with pytest.raises(RuntimeError, match="unsupported custom Qiskit instruction"):
        QCProgram.from_qiskit(custom)

    delayed = QuantumCircuit(1)
    delayed.delay(10, 0)
    with pytest.raises(RuntimeError, match="delay instructions are not supported"):
        QCProgram.from_qiskit(delayed)

    theta = Parameter("theta")
    composite = QuantumCircuit(1)
    composite.rx(theta + 1, 0)
    with pytest.raises(RuntimeError, match="unsupported expression"):
        QCProgram.from_qiskit(composite)

    boxed = QuantumCircuit(1)
    with boxed.box():
        boxed.x(0)
    with pytest.raises(RuntimeError, match="box instructions are not supported"):
        QCProgram.from_qiskit(boxed)

    for instruction in ("break_loop", "continue_loop"):
        loop_control = QuantumCircuit(1)
        with loop_control.for_loop(range(2), None, None, None, None, label=None):
            getattr(loop_control, instruction)()
        with pytest.raises(RuntimeError, match=instruction.removesuffix("_loop")):
            QCProgram.from_qiskit(loop_control)


@requires_bridge
@pytest.mark.parametrize(
    ("variable_type", "message"),
    [
        (types.Uint(65), "wider than 64 bits"),
        (types.Duration(), "duration expressions are not supported"),
    ],
)
def test_qiskit_bridge_rejects_unsupported_classical_types(variable_type: types.Type, message: str) -> None:
    """Reject unsafe integer widths and non-executable classical types."""
    value = expr.Var.new("value", variable_type)
    circuit = QuantumCircuit(1, inputs=[value])
    with circuit.if_test(expr.equal(value, value)):
        circuit.x(0)

    with pytest.raises(RuntimeError, match=message):
        QCProgram.from_qiskit(circuit)


@requires_bridge
def test_qiskit_bridge_rejects_unidentifiable_expression_variables() -> None:
    """Report the 2.5 C-API gap for bit-backed expression variables safely."""
    circuit = QuantumCircuit(1, 1)
    with circuit.if_test(expr.equal(circuit.clbits[0], expr.lift(bool(1)))):
        circuit.x(0)

    with pytest.raises(RuntimeError, match="cannot identify bit- or register-backed"):
        QCProgram.from_qiskit(circuit)


@requires_bridge
def test_qiskit_bridge_rejects_unknown_minor_before_capsule_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dispatch only an exact registered final Qiskit minor."""
    monkeypatch.setattr(qiskit, "__version__", "2.6.0")
    with pytest.raises(
        RuntimeError,
        match=r"installed version '2\.6\.0'.*>=2\.5\.0,<2\.6\.0",
    ):
        QCProgram.from_qiskit(QuantumCircuit(1))

    legacy = qiskit_to_mqt(QuantumCircuit(1))
    assert legacy.num_qubits == 1


@requires_bridge
def test_qiskit_bridge_rejects_prerelease_before_capsule_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not admit a prerelease through a final-minor shipping adapter."""
    monkeypatch.setattr(qiskit, "__version__", "2.5.2rc1")
    with pytest.raises(
        RuntimeError,
        match=r"prerelease or non-final version '2\.5\.2rc1'.*>=2\.5\.0,<2\.6\.0",
    ):
        QCProgram.from_qiskit(QuantumCircuit(1))


def test_mlir_binding_import_does_not_import_qiskit() -> None:
    """Keep Qiskit entirely lazy at MLIR-binding import time."""
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
