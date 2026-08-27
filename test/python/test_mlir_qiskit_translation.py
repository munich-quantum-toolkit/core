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
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

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
    ParameterExpression,
    ParameterVector,
    PowerModifier,
    Qubit,
    library,
)
from qiskit.circuit.classical import expr, types
from qiskit.circuit.controlflow import CASE_DEFAULT, IfElseOp
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.quantum_info import Operator, random_unitary

from mqt.core.mlir import CompilerTarget, QCProgram, compile_program
from mqt.core.plugins.qiskit import qiskit_to_mqt

if TYPE_CHECKING:
    from collections.abc import Callable

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


def test_native_api_initialization_supports_concurrent_translation() -> None:
    """Initialize and reuse the native Qiskit API from concurrent translations."""

    def _round_trip(_: int) -> str:
        circuit = QuantumCircuit(1)
        circuit.x(0)
        return QCProgram.from_qiskit(circuit).to_qiskit().data[0].operation.name

    with ThreadPoolExecutor(max_workers=8) as executor:
        assert list(executor.map(_round_trip, range(32))) == ["x"] * 32


@pytest.mark.parametrize("gate", STANDARD_GATES, ids=lambda gate: gate.name)
def test_standard_gates_round_trip(gate: Gate) -> None:
    """Translate each supported gate family in both directions."""
    circuit = QuantumCircuit(gate.num_qubits)
    circuit.append(gate, range(gate.num_qubits))

    restored = QCProgram.from_qiskit(circuit).to_qiskit()

    assert np.allclose(Operator(restored).data, Operator(circuit).data)


def test_dense_unitary_round_trip_preserves_qarg_mapping_and_source_data() -> None:
    """Preserve a dense unitary and its qubit mapping without changing source data."""
    local = QuantumCircuit(2)
    local.global_phase = 0.23
    local.h(0)
    local.cx(0, 1)
    local.rz(0.37, 1)

    circuit = QuantumCircuit(3)
    circuit.x(1)
    local_operator = Operator(local)
    circuit.append(library.UnitaryGate(local_operator), [2, 0])
    source_data = list(circuit.data)
    source_operator = Operator(circuit)

    program = QCProgram.from_qiskit(circuit)
    restored = program.to_qiskit()

    assert "qc.unitary" in program.ir
    assert np.allclose(Operator(restored).data, source_operator.data)
    assert np.allclose(Operator(circuit).data, source_operator.data)
    assert list(circuit.data) == source_data
    assert circuit.count_ops() == {"x": 1, "unitary": 1}
    assert restored.count_ops() == {"x": 1, "unitary": 1}
    restored_unitary = next(item for item in restored.data if item.operation.name == "unitary")
    assert [restored.find_bit(qubit).index for qubit in restored_unitary.qubits] == [2, 0]
    assert np.allclose(Operator(restored_unitary.operation).data, local_operator.data)


def test_dense_unitary_import_converts_qiskit_qubit_order() -> None:
    """Convert Qiskit's qubit order by reversing the operation targets."""
    matrix = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0j, 0.0],
        [0.0, 0.0, 0.0, -1.0],
        [-1.0j, 0.0, 0.0, 0.0],
    ])
    circuit = QuantumCircuit(2)
    circuit.append(library.UnitaryGate(matrix), [0, 1])

    ir = QCProgram.from_qiskit(circuit).ir

    dense_text = ir.split("qc.unitary dense<[", 1)[1].split("]>", 1)[0]
    matches = re.findall(r"\(([-+0-9.eE]+),([-+0-9.eE]+)\)", dense_text)
    entries = [complex(float(real), float(imaginary)) for real, imaginary in matches]
    imported = np.asarray(entries).reshape((4, 4))
    assert np.allclose(imported, matrix)
    assert "%1, %0 : !qc.qubit, !qc.qubit" in ir


@pytest.mark.parametrize("num_qubits", [1, 2, 3])
def test_dense_unitary_round_trip(num_qubits: int) -> None:
    """Preserve one-, two-, and three-qubit dense unitaries."""
    circuit = QuantumCircuit(num_qubits)
    matrix = random_unitary(2**num_qubits, seed=100 + num_qubits)
    circuit.append(library.UnitaryGate(matrix), range(num_qubits))

    program = QCProgram.from_qiskit(circuit)
    restored = program.to_qiskit()

    assert "qc.unitary" in program.ir
    assert np.allclose(Operator(restored).data, Operator(circuit).data)
    assert restored.count_ops() == {"unitary": 1}


def test_dense_unitary_import_rejects_more_than_eight_qubits() -> None:
    """Reject oversized matrices before constructing a compiler program."""
    circuit = QuantumCircuit(9)
    circuit.append(
        library.UnitaryGate(np.eye(2**9), check_input=False),
        range(9),
    )

    with pytest.raises(RuntimeError, match=r"supports at most \d+ qubits"):
        QCProgram.from_qiskit(circuit)


def test_quantum_volume_unitaries_remain_dense() -> None:
    """Preserve the dense two-qubit unitaries used by Quantum Volume."""
    circuit = library.quantum_volume(4, depth=3, seed=12345)
    assert circuit.count_ops().get("unitary") == 6

    program = QCProgram.from_qiskit(circuit)
    restored = program.to_qiskit()

    assert program.ir.count("qc.unitary") == 6
    assert np.allclose(Operator(restored).data, Operator(circuit).data)
    assert restored.count_ops().get("unitary") == 6


def test_two_qubit_dense_unitary_compiles_to_target_basis() -> None:
    """Synthesize a dense two-qubit unitary to the target basis."""
    circuit = QuantumCircuit(2)
    circuit.append(library.UnitaryGate(random_unitary(4, seed=2136)), [0, 1])
    target = CompilerTarget(
        2,
        operations=[
            CompilerTarget.Operation("u", 1, 3),
            CompilerTarget.Operation("cx", 2, 0),
        ],
    )
    program = QCProgram.from_qiskit(circuit).to_qco(copy=True)

    program.compile_for_target(target)
    restored = program.to_qc(copy=True).to_qiskit(target=target)

    assert "qco.unitary" not in program.ir
    assert restored.size() > 0
    assert set(restored.count_ops()) <= {"u", "cx"}


def test_controlled_dense_unitary_export_preserves_operation_order() -> None:
    """Export a controlled dense matrix with a Qiskit control annotation."""
    program = QCProgram.from_mlir_str(
        """module {
  func.func @main() attributes {mqt.entry_point} {
    %control = qc.alloc : !qc.qubit
    %target = qc.alloc : !qc.qubit
    qc.x %control : !qc.qubit
    qc.ctrl(%control) targets (%argument = %target) {
      qc.unitary dense<[[(0.0,0.0), (1.0,0.0)],
                        [(1.0,0.0), (0.0,0.0)]]>
          : tensor<2x2xcomplex<f64>> %argument : !qc.qubit
      qc.yield
    } : {!qc.qubit}, {!qc.qubit}
    qc.z %target : !qc.qubit
    qc.dealloc %control : !qc.qubit
    qc.dealloc %target : !qc.qubit
    return
  }
}
"""
    )

    restored = program.to_qiskit()

    assert [item.operation.name for item in restored.data[:1]] == ["x"]
    assert [item.operation.name for item in restored.data[2:]] == ["z"]
    controlled = restored.data[1]
    assert isinstance(controlled.operation, AnnotatedOperation)
    assert len(controlled.operation.modifiers) == 1
    modifier = controlled.operation.modifiers[0]
    assert isinstance(modifier, ControlModifier)
    assert modifier.num_ctrl_qubits == 1
    assert modifier.ctrl_state == 1
    assert [restored.find_bit(qubit).index for qubit in controlled.qubits] == [0, 1]
    expected = QuantumCircuit(2)
    expected.x(0)
    expected.cx(0, 1)
    expected.z(1)
    assert np.allclose(Operator(restored).data, Operator(expected).data)


def test_wrapped_dense_unitary_import_avoids_unsafe_c_accessor() -> None:
    """Import wrapped dense matrices without entering Qiskit's C accessor."""
    script = """
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import (
    AnnotatedOperation,
    ControlModifier,
    InverseModifier,
    PowerModifier,
)
from qiskit.circuit.library import UnitaryGate
from mqt.core.mlir import QCProgram

unitary = UnitaryGate(np.array([[0.0, 1.0], [1.0, 0.0]]))
renamed = UnitaryGate(np.array([[0.0, 1.0], [1.0, 0.0]]))
renamed.name = "renamed_unitary"
operations = [
    unitary.control(1),
    AnnotatedOperation(unitary, []),
    AnnotatedOperation(unitary, InverseModifier()),
    AnnotatedOperation(unitary, PowerModifier(0.5)),
    AnnotatedOperation(unitary, ControlModifier(1)),
    renamed,
]
for operation in operations:
    circuit = QuantumCircuit(operation.num_qubits)
    circuit.append(operation, circuit.qubits)
    program = QCProgram.from_qiskit(circuit)
    assert "qc.unitary" in program.ir
"""

    subprocess.run([sys.executable, "-c", script], check=True)  # ruff: ignore[subprocess-without-shell-equals-true]


@pytest.mark.parametrize(
    ("modifier", "expected"),
    [
        (InverseModifier(), "qc.inv"),
        (PowerModifier(0.5), "qc.pow"),
        (ControlModifier(2), "qc.ctrl"),
    ],
    ids=["inverse", "power", "control"],
)
def test_dense_unitary_modifiers_are_imported(
    modifier: InverseModifier | PowerModifier | ControlModifier, expected: str
) -> None:
    """Preserve supported Qiskit modifiers around dense unitary operations."""
    operation = AnnotatedOperation(
        library.UnitaryGate(np.asarray([[1.0, 0.0], [0.0, 1.0j]])),
        modifier,
    )
    circuit = QuantumCircuit(operation.num_qubits)
    circuit.append(operation, circuit.qubits)

    program = QCProgram.from_qiskit(circuit)

    assert "qc.unitary" in program.ir
    assert expected in program.ir
    if not isinstance(modifier, PowerModifier):
        restored = program.to_qiskit()
        assert np.allclose(Operator(restored).data, Operator(circuit).data)


def test_controlled_dense_unitary_round_trip_preserves_qarg_order() -> None:
    """Preserve controls and target ordering around an asymmetric matrix."""
    operation = library.UnitaryGate(random_unitary(4, seed=2136)).control(1)
    circuit = QuantumCircuit(4)
    circuit.append(operation, [3, 0, 2])

    program = QCProgram.from_qiskit(circuit)
    restored = program.to_qiskit()

    assert "qc.ctrl" in program.ir
    assert "qc.unitary" in program.ir
    controlled = restored.data[0]
    assert [restored.find_bit(qubit).index for qubit in controlled.qubits] == [3, 0, 2]
    assert np.allclose(Operator(restored).data, Operator(circuit).data)


def test_inverse_controlled_dense_unitary_round_trip() -> None:
    """Preserve inverse and control modifiers around one dense unitary."""
    unitary = library.UnitaryGate(np.asarray([[1.0, 0.0], [0.0, 1.0j]]))
    operation = AnnotatedOperation(
        AnnotatedOperation(unitary, ControlModifier(1)),
        InverseModifier(),
    )
    circuit = QuantumCircuit(2)
    circuit.append(operation, circuit.qubits)

    program = QCProgram.from_qiskit(circuit)
    restored = program.to_qiskit()

    assert "qc.ctrl" in program.ir
    assert "qc.inv" in program.ir
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

    assert 'mqt.register_name = "input"' in program.ir
    assert 'cbit.alloc(#cbit.init<zero>) {mqt.register_name = "output"}' in program.ir
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


def test_openqasm2_measurements_export_with_zero_initialized_register() -> None:
    """Export an OpenQASM 2 zero-initialized result register."""
    program = QCProgram.from_qasm_str(
        """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
x q[1];
measure q[1] -> c[0];
measure q[0] -> c[1];
"""
    )

    restored = program.to_qiskit()

    assert [(register.name, len(register)) for register in restored.qregs] == [("q", 2)]
    assert [(register.name, len(register)) for register in restored.cregs] == [("c", 2)]
    assert [item.operation.name for item in restored.data] == ["x", "measure", "measure"]
    measurements = [item for item in restored.data if item.operation.name == "measure"]
    assert [
        (restored.find_bit(item.qubits[0]).index, restored.find_bit(item.clbits[0]).index) for item in measurements
    ] == [(1, 0), (0, 1)]


@pytest.mark.parametrize("late_value", ["false", "true"])
def test_flat_export_rejects_classical_store_after_quantum_work(late_value: str) -> None:
    """Reject constant CBit stores regardless of their position."""
    program = QCProgram.from_mlir_str(
        f"""module {{
  func.func @main() -> !cbit.reg<2> attributes {{mqt.entry_point}} {{
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %initial = arith.constant false
    %late = arith.constant {late_value}
    %q = qc.alloc : !qc.qubit
    %c = cbit.alloc(#cbit.init<undefined>) : !cbit.reg<2>
    cbit.store %initial, %c[%c0] : !cbit.reg<2>
    qc.x %q : !qc.qubit
    cbit.store %late, %c[%c1] : !cbit.reg<2>
    qc.dealloc %q : !qc.qubit
    return %c : !cbit.reg<2>
  }}
}}
"""
    )

    with pytest.raises(RuntimeError, match="does not support non-measurement classical stores"):
        program.to_qiskit()


def test_target_compiled_openqasm2_measurements_export() -> None:
    """Export initialized result registers after target compilation."""
    target = CompilerTarget(5)
    program = QCProgram.from_qasm_str(
        """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
x q[1];
measure q[1] -> c[0];
measure q[0] -> c[1];
"""
    )
    mapped = program.to_qco(copy=True)
    mapped.compile_for_target(target)

    restored = mapped.to_qc(copy=True).to_qiskit(target=target)

    assert restored.num_qubits == 5
    assert [(register.name, len(register)) for register in restored.qregs] == [("q", 5)]
    assert [(register.name, len(register)) for register in restored.cregs] == [("c", 2)]
    assert restored.layout is None
    assert restored.count_ops() == {"measure": 2, "x": 1}


def test_cleanup_forwards_measurement_results_to_qiskit_condition() -> None:
    """Export a condition after cleanup forwards its measurement loads."""
    program = QCProgram.from_qasm_str(
        """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
if (c == 3) x q[2];
"""
    )
    optimized = program.to_qco(copy=True)
    optimized.cleanup()

    restored = optimized.to_qc(copy=True).to_qiskit()

    assert restored.count_ops() == {"measure": 2, "if_else": 1}
    assert restored.data[2].operation.blocks[0].count_ops() == {"x": 1}
    condition = restored.data[2].operation.condition
    assert isinstance(condition, expr.Expr)
    assert expr.structurally_equivalent(condition, expr.logic_and(*restored.clbits))


def test_openqasm_short_circuit_expression_exports_to_qiskit() -> None:
    """Export nested OpenQASM short-circuit logic through canonical scf.if."""
    program = QCProgram.from_qasm_str(
        """OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[2] c;
c[0] = measure q[0];
c[1] = measure q[1];
if (c[0] && (c[1] || !c[0])) x q[2];
"""
    )

    restored = program.to_qiskit()
    condition = restored.data[2].operation.condition
    expected = expr.logic_and(
        restored.clbits[0],
        expr.logic_or(
            restored.clbits[1],
            expr.bit_xor(
                restored.clbits[0],
                True,  # ruff: ignore[boolean-positional-value-in-call] Qiskit expression arguments are positional-only.
            ),
        ),
    )

    assert program.ir.count("scf.if") >= 2
    assert isinstance(condition, expr.Expr)
    assert expr.structurally_equivalent(condition, expected)


def test_openqasm3_measurement_export_uses_undefined_cbit_register() -> None:
    """Represent OpenQASM 3 output initialization without poison values."""
    program = QCProgram.from_qasm_str(
        """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[1] c;
h q[1];
c[0] = measure q[1];
"""
    )

    restored = program.to_qiskit()

    assert "ub.poison" not in program.ir
    assert 'cbit.alloc(#cbit.init<undefined>) {mqt.register_name = "c"}' in program.ir
    assert [(register.name, len(register)) for register in restored.qregs] == [("q", 2)]
    assert [(register.name, len(register)) for register in restored.cregs] == [("c", 1)]
    assert [item.operation.name for item in restored.data] == ["h", "measure"]
    measurement = restored.data[-1]
    assert restored.find_bit(measurement.qubits[0]).index == 1
    assert restored.find_bit(measurement.clbits[0]).index == 0


def test_flat_export_rejects_undefined_returned_bits() -> None:
    """Reject a returned undefined register unless every bit is written."""
    program = QCProgram.from_mlir_str(
        """module {
  func.func @main() -> !cbit.reg<1> attributes {mqt.entry_point} {
    %q = qc.alloc : !qc.qubit
    %c = cbit.alloc(#cbit.init<undefined>) : !cbit.reg<1>
    qc.dealloc %q : !qc.qubit
    return %c : !cbit.reg<1>
  }
}
"""
    )

    with pytest.raises(RuntimeError, match="cannot return undefined classical bits"):
        program.to_qiskit()


def test_qiskit_export_rejects_noncanonical_i64_function_result() -> None:
    """Reject a nonzero i64 result instead of treating it as the output sentinel."""
    program = QCProgram.from_mlir_str(
        """module {
  func.func @main() -> i64 attributes {mqt.entry_point} {
    %q = qc.alloc : !qc.qubit
    %value = arith.constant 1 : i64
    qc.dealloc %q : !qc.qubit
    return %value : i64
  }
}
"""
    )

    with pytest.raises(RuntimeError, match="supports only CBit function return values"):
        program.to_qiskit()


def test_qiskit_export_rejects_mixed_sentinel_and_cbit_results() -> None:
    """Reject the zero sentinel when it is mixed with a public CBit result."""
    program = QCProgram.from_mlir_str(
        """module {
  func.func @main() -> (i64, !cbit.reg<1>) attributes {mqt.entry_point} {
    %q = qc.alloc : !qc.qubit
    %classical = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "c"} : !cbit.reg<1>
    %zero = arith.constant 0 : i64
    qc.dealloc %q : !qc.qubit
    return %zero, %classical : i64, !cbit.reg<1>
  }
}
"""
    )

    with pytest.raises(RuntimeError, match="supports only CBit function return values"):
        program.to_qiskit()


def test_qiskit_round_trip_preserves_anonymous_clbits() -> None:
    """Represent loose Qiskit clbits as one anonymous public CBit register."""
    circuit = QuantumCircuit(1)
    circuit.add_bits([Clbit()])
    circuit.measure(0, 0)

    program = QCProgram.from_qiskit(circuit)
    restored = program.to_qiskit()

    assert "cbit.alloc(#cbit.init<zero>) : !cbit.reg<1>" in program.ir
    assert restored.num_clbits == 1
    assert restored.cregs == []
    assert restored.count_ops() == {"measure": 1}


def test_qiskit_export_excludes_internal_cbit_registers() -> None:
    """Export only CBit registers returned by the entry function."""
    program = QCProgram.from_mlir_str(
        """module {
  func.func @main() -> !cbit.reg<1> attributes {mqt.entry_point} {
    %q = qc.alloc : !qc.qubit
    %output = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "output"} : !cbit.reg<1>
    %internal = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "internal"} : !cbit.reg<2>
    qc.dealloc %q : !qc.qubit
    return %output : !cbit.reg<1>
  }
}
"""
    )

    restored = program.to_qiskit()

    assert restored.num_clbits == 1
    assert [(register.name, len(register)) for register in restored.cregs] == [("output", 1)]


def test_qiskit_export_rejects_duplicate_measurement_destinations() -> None:
    """Reject multiple measurements that write the same public bit."""
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    circuit.measure(0, 0)

    with pytest.raises(RuntimeError, match="duplicate classical destinations"):
        QCProgram.from_qiskit(circuit).to_qiskit()


def test_qiskit_export_rejects_measurement_with_multiple_destinations() -> None:
    """Reject one measurement result stored in more than one public bit."""
    program = QCProgram.from_mlir_str(
        """module {
  func.func @main() -> !cbit.reg<2> attributes {mqt.entry_point} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %q = qc.alloc : !qc.qubit
    %c = cbit.alloc(#cbit.init<undefined>) : !cbit.reg<2>
    %result = qc.measure %q : !qc.qubit -> i1
    cbit.store %result, %c[%c0] : !cbit.reg<2>
    cbit.store %result, %c[%c1] : !cbit.reg<2>
    qc.dealloc %q : !qc.qubit
    return %c : !cbit.reg<2>
  }
}
"""
    )

    with pytest.raises(RuntimeError, match="more than one classical destination"):
        program.to_qiskit()


def test_qiskit_export_rejects_dynamic_measurement_destination() -> None:
    """Require each Qiskit measurement destination to be static."""
    program = QCProgram.from_mlir_str(
        """module {
  func.func @main() -> !cbit.reg<1> attributes {mqt.entry_point} {
    %c0 = arith.constant 0 : index
    %index = arith.addi %c0, %c0 : index
    %q = qc.alloc : !qc.qubit
    %c = cbit.alloc(#cbit.init<undefined>) : !cbit.reg<1>
    %result = qc.measure %q : !qc.qubit -> i1
    cbit.store %result, %c[%index] : !cbit.reg<1>
    qc.dealloc %q : !qc.qubit
    return %c : !cbit.reg<1>
  }
}
"""
    )

    with pytest.raises(RuntimeError, match="dynamic classical destination"):
        program.to_qiskit()


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
    """Reject custom-definition symbols absent from the enclosing circuit."""
    z = Parameter("z")
    a = Parameter("a")
    definition = QuantumCircuit(1)
    definition.rz(z, 0)
    definition.rx(a, 0)
    gate = Gate("ambiguous", 1, [0.1, 0.2])
    gate.definition = definition
    circuit = QuantumCircuit(1)
    circuit.append(gate, [0])

    with pytest.raises(RuntimeError, match="parameter symbol 'z' is not defined"):
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
    """Reject unsupported parameters and inputs without mutation."""
    theta = Parameter("theta")
    symbolic = QuantumCircuit(1)
    symbolic.rx(theta.sign(), 0)
    symbolic_data = list(symbolic.data)
    with pytest.raises(
        RuntimeError,
        match=r"(?i)Qiskit parameter expression operation 'sign' is not supported",
    ):
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


@pytest.mark.parametrize("value", [np.inf, np.nan], ids=["infinity", "nan"])
def test_nonfinite_parameters_fail_closed_without_mutation(value: float) -> None:
    """Reject non-finite scalar parameters before changing the source circuit."""
    circuit = QuantumCircuit(1)
    circuit.rx(value, 0)
    source_data = list(circuit.data)

    with pytest.raises(RuntimeError, match="Qiskit returned a non-finite parameter"):
        QCProgram.from_qiskit(circuit)

    assert len(circuit.data) == len(source_data)
    current = circuit.data[0]
    original = source_data[0]
    assert current.operation.name == original.operation.name == "rx"
    assert current.qubits == original.qubits
    assert current.clbits == original.clbits
    assert len(current.operation.params) == 1
    assert np.isnan(current.operation.params[0]) == np.isnan(value)
    assert np.isinf(current.operation.params[0]) == np.isinf(value)


def test_complex_parameter_expression_fails_closed_without_mutation() -> None:
    """Reject a complex-valued expression before changing the source circuit."""
    theta = Parameter("theta")
    circuit = QuantumCircuit(1)
    circuit.rx(theta + 1j, 0)
    source_data = list(circuit.data)

    with pytest.raises(RuntimeError, match="parameter expressions with complex values are not supported"):
        QCProgram.from_qiskit(circuit)

    assert list(circuit.data) == source_data
    assert circuit.parameters == {theta}


def test_excessively_nested_parameter_expression_fails_closed_without_mutation() -> None:
    """Bound parameter-expression traversal before changing the source circuit."""
    theta = Parameter("theta")
    angle: ParameterExpression = theta
    for _ in range(65):
        angle = angle.sin()
    circuit = QuantumCircuit(1)
    circuit.rz(angle, 0)
    source_data = list(circuit.data)

    with pytest.raises(RuntimeError, match="exceeds the supported 64-level nesting depth"):
        QCProgram.from_qiskit(circuit)

    assert list(circuit.data) == source_data
    assert circuit.parameters == {theta}


def test_oversized_parameter_expression_fails_closed_without_mutation() -> None:
    """Bound a wide parameter expression before changing the source circuit."""
    theta = Parameter("theta")
    level: list[ParameterExpression] = [theta]
    level.extend(theta + float(index) for index in range(1, 2049))
    while len(level) > 1:
        level = [
            level[index] + level[index + 1] if index + 1 < len(level) else level[index]
            for index in range(0, len(level), 2)
        ]
    circuit = QuantumCircuit(1)
    circuit.rz(level[0], 0)
    source_data = list(circuit.data)

    with pytest.raises(RuntimeError, match="exceeds the supported 4096-node size"):
        QCProgram.from_qiskit(circuit)

    assert list(circuit.data) == source_data
    assert circuit.parameters == {theta}


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
    """Round-trip structured control while keeping induction values lexical."""
    circuit = QuantumCircuit(2, 2)
    with circuit.for_loop(range(1, 5, 2), None, None, None, None, label=None) as iteration:
        with circuit.if_test((circuit.clbits[0], False)) as else_:
            circuit.ry(iteration, 1)
        with else_:
            circuit.rz(iteration, 1)
    with circuit.while_loop((circuit.cregs[0], 0), None, None, None, label=None):
        circuit.measure(0, 0)
    with circuit.switch(circuit.cregs[0], None, None, None, label=None) as case:
        with case(0, 1):
            circuit.x(0)
        with case(case.DEFAULT):
            circuit.z(1)

    program = compile_program(circuit)
    source = program.ir

    assert "scf.for" in program.ir
    assert "scf.if" in program.ir
    assert "scf.while" in program.ir
    assert "scf.index_switch" in program.ir
    restored = program.to_qiskit()

    assert program.ir == source
    assert [instruction.operation.name for instruction in restored.data] == [
        "for_loop",
        "while_loop",
        "switch_case",
    ]
    loop = restored.data[0].operation
    loop_parameter = loop.params[1]
    loop_body = loop.blocks[0]
    branch = loop_body.data[0].operation
    assert branch.name == "if_else"
    assert branch.blocks[0].data[0].operation.params[0].uuid == loop_parameter.uuid
    assert branch.blocks[1].data[0].operation.params[0].uuid == loop_parameter.uuid
    switch_cases = list(restored.data[2].operation.cases_specifier())
    assert [labels for labels, _ in switch_cases] == [(0,), (1,), (CASE_DEFAULT,)]
    assert [[instruction.operation.name for instruction in body.data] for _, body in switch_cases] == [
        ["x"],
        ["x"],
        ["z"],
    ]
    QCProgram.from_qiskit(restored)


def test_control_flow_and_controlled_unitary_preserve_instruction_order() -> None:
    """Keep both deferred instruction kinds at their original positions."""
    circuit = QuantumCircuit(2, 1)
    circuit.h(0)
    controlled = library.UnitaryGate(np.asarray([[0.0, 1.0], [1.0, 0.0]])).control(1)
    with circuit.if_test((circuit.clbits[0], True)):
        circuit.append(controlled, [0, 1])
    circuit.z(1)

    restored = QCProgram.from_qiskit(circuit).to_qiskit()

    assert [instruction.operation.name for instruction in restored.data] == ["h", "if_else", "z"]
    body_operation = restored.data[1].operation.blocks[0].data[0].operation
    assert isinstance(body_operation, AnnotatedOperation)
    assert isinstance(body_operation.modifiers[0], ControlModifier)


@pytest.mark.parametrize("num_clbits", [3, 64])
def test_root_register_expression_and_nested_condition_preserve_captures(num_clbits: int) -> None:
    """Keep a root register leaf and pack its nested block-local condition."""
    circuit = QuantumCircuit(1, num_clbits)
    condition = expr.logic_and(expr.equal(circuit.cregs[0], 5), circuit.clbits[0])
    with circuit.if_test(condition), circuit.if_test((circuit.cregs[0], 2)):
        circuit.x(0)

    restored = QCProgram.from_qiskit(circuit).to_qiskit()

    outer = restored.data[0].operation
    assert isinstance(outer.condition, expr.Expr)
    outer_variables = {variable.var for variable in expr.iter_vars(outer.condition)}
    assert outer_variables == {restored.cregs[0], restored.clbits[0]}
    inner = outer.blocks[0].data[0].operation
    assert isinstance(inner.condition, expr.Expr)
    assert {variable.var for variable in expr.iter_vars(inner.condition)} == set(outer.blocks[0].clbits)


def test_repeated_cbit_uint_expression_falls_back_to_expression_tree() -> None:
    """Do not misidentify repeated source bits as a packed classical register."""
    program = _single_qubit_program(
        [
            '%classical = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "c"} : !cbit.reg<1>',
            "%zero = arith.constant 0 : index",
            "%one = arith.constant 1 : i2",
            "%three = arith.constant 3 : i2",
            "%bit = cbit.load %classical[%zero] : !cbit.reg<1>",
            "%wide = arith.extui %bit : i1 to i2",
            "%shifted = arith.shli %wide, %one : i2",
            "%repeated = arith.ori %wide, %shifted : i2",
            "%condition = arith.cmpi eq, %repeated, %three : i2",
            "scf.if %condition {",
            "  qc.x %q : !qc.qubit",
            "}",
        ],
        returns_classical=True,
    )

    restored = program.to_qiskit()

    condition = restored.data[0].operation.condition
    assert isinstance(condition, expr.Expr)
    assert {variable.var for variable in expr.iter_vars(condition)} == {restored.clbits[0]}


def test_free_parameter_identity_is_shared_with_control_flow_blocks() -> None:
    """Canonicalize one scalar Parameter across root and nested writers."""
    theta = Parameter("theta")
    circuit = QuantumCircuit(1, 1, global_phase=theta / 2)
    circuit.rz(theta, 0)
    with circuit.if_test((circuit.clbits[0], True)):
        circuit.rx(theta + 1, 0)

    restored = QCProgram.from_qiskit(circuit).to_qiskit()

    restored_theta = next(iter(restored.parameters))
    assert restored.global_phase.parameters == {restored_theta}
    assert restored.data[0].operation.params[0] == restored_theta
    nested_parameter = restored.data[1].operation.blocks[0].data[0].operation.params[0]
    assert nested_parameter.parameters == {restored_theta}


def test_nested_if_while_switch_preserve_capture_identity() -> None:
    """Map nested control-flow operands through each block-local bit list."""
    circuit = QuantumCircuit(2, 2)
    with (
        circuit.if_test(expr.logic_and(circuit.clbits[0], expr.logic_not(circuit.clbits[1]))),
        circuit.while_loop(expr.logic_not(circuit.clbits[0]), None, None, None, label=None),
        circuit.switch(expr.bit_xor(circuit.cregs[0], 1), None, None, None, label=None) as case,
    ):
        with case(0):
            circuit.x(0)
        with case(case.DEFAULT):
            circuit.cx(0, 1)

    restored = QCProgram.from_qiskit(circuit).to_qiskit()

    outer = restored.data[0]
    assert [restored.find_bit(bit).index for bit in outer.clbits] == [0, 1]
    outer_body = outer.operation.blocks[0]
    while_instruction = outer_body.data[0]
    assert [outer_body.find_bit(bit).index for bit in while_instruction.clbits] == [0, 1]
    assert isinstance(while_instruction.operation.condition, expr.Expr)
    while_body = while_instruction.operation.blocks[0]
    switch_instruction = while_body.data[0]
    assert [while_body.find_bit(bit).index for bit in switch_instruction.clbits] == [0, 1]
    assert switch_instruction.operation.name == "switch_case"
    assert isinstance(switch_instruction.operation.target, expr.Expr)


def test_empty_if_else_branches_round_trip() -> None:
    """Preserve an explicit else branch when both branches are empty."""
    circuit = QuantumCircuit(1, 1)
    with circuit.if_test((circuit.clbits[0], True)) as else_:
        pass
    with else_:
        pass

    restored = QCProgram.from_qiskit(circuit).to_qiskit()

    operation = restored.data[0].operation
    assert operation.name == "if_else"
    assert len(operation.blocks) == 2
    assert all(not block.data for block in operation.blocks)


def test_zero_qubit_cbit_only_control_flow_round_trip() -> None:
    """Round-trip CBit-only structured control without allocating qubits."""
    program = QCProgram.from_mlir_str(
        """module {
  func.func @main() -> !cbit.reg<1> attributes {mqt.entry_point} {
    %classical = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "c"} : !cbit.reg<1>
    %zero = arith.constant 0 : index
    %phase = arith.constant 0.0 : f64
    qc.gphase(%phase)
    %condition = cbit.load %classical[%zero] : !cbit.reg<1>
    scf.if %condition {
    }
    return %classical : !cbit.reg<1>
  }
}
"""
    )

    restored = program.to_qiskit()

    assert restored.num_qubits == 0
    assert restored.num_clbits == 1
    assert len(restored.data) == 1
    instruction = restored.data[0]
    assert instruction.operation.name == "if_else"
    assert instruction.qubits == ()
    assert instruction.clbits == (restored.clbits[0],)
    block = instruction.operation.blocks[0]
    assert block.num_qubits == 0
    assert block.num_clbits == 1
    QCProgram.from_qiskit(restored)


def _single_qubit_program(operations: list[str], *, returns_classical: bool = False) -> QCProgram:
    """Wrap operations in a one-qubit QC entry function.

    Returns:
        The parsed QC program.
    """
    result_type = " -> !cbit.reg<1>" if returns_classical else ""
    return_value = " %classical : !cbit.reg<1>" if returns_classical else ""
    lines = [
        "module {",
        f"  func.func @main(){result_type} attributes {{mqt.entry_point}} {{",
        "    %q = qc.alloc : !qc.qubit",
    ]
    lines.extend(f"    {operation}" for operation in operations)
    lines.extend([
        "    qc.dealloc %q : !qc.qubit",
        f"    return{return_value}",
        "  }",
        "}",
    ])
    return QCProgram.from_mlir_str("\n".join(lines))


@pytest.mark.parametrize(
    ("values", "expected"),
    [(range(5, -2, -2), [5, 3, 1, -1]), (range(3, 3, -1), [])],
    ids=["negative-step", "zero-iterations"],
)
def test_for_loop_range_edges_round_trip(values: range, expected: list[int]) -> None:
    """Preserve descending induction values and empty iteration sets."""
    circuit = QuantumCircuit(1)
    with circuit.for_loop(values, None, None, None, None, label=None) as iteration:
        circuit.rx(iteration, 0)

    restored = QCProgram.from_qiskit(circuit).to_qiskit()

    loop = restored.data[0].operation
    assert loop.name == "for_loop"
    assert list(loop.params[0]) == expected
    assert loop.blocks[0].data[0].operation.params[0].uuid == loop.params[1].uuid


def test_for_loop_affine_projection_checks_the_fused_result() -> None:
    """Accept a fitting affine value whose intermediate product overflows."""
    program = _single_qubit_program([
        "%lower = arith.constant -2 : index",
        "%upper = arith.constant -1 : index",
        "%step = arith.constant 1 : index",
        "%maximum = arith.constant 9223372036854775807 : i64",
        "scf.for %iteration = %lower to %upper step %step {",
        "  %integer = arith.index_cast %iteration : index to i64",
        "  %scaled = arith.muli %integer, %maximum : i64",
        "  %shifted = arith.addi %scaled, %maximum : i64",
        "  %parameter = arith.sitofp %shifted : i64 to f64",
        "  qc.rz(%parameter) %q : !qc.qubit",
        "}",
    ])

    loop = program.to_qiskit().data[0].operation

    assert list(loop.params[0]) == [-9223372036854775807]
    assert loop.blocks[0].data[0].operation.params[0].uuid == loop.params[1].uuid


def test_empty_for_loop_ignores_unrepresentable_projection() -> None:
    """Keep an empty loop even when its unused projection has a zero step."""
    program = _single_qubit_program([
        "%lower = arith.constant 1 : index",
        "%upper = arith.constant 0 : index",
        "%step = arith.constant 1 : index",
        "%zero = arith.constant 0 : i64",
        "scf.for %iteration = %lower to %upper step %step {",
        "  %integer = arith.index_cast %iteration : index to i64",
        "  %scaled = arith.muli %integer, %zero : i64",
        "  %parameter = arith.sitofp %scaled : i64 to f64",
        "  qc.rz(%parameter) %q : !qc.qubit",
        "}",
    ])

    loop = program.to_qiskit().data[0].operation

    assert list(loop.params[0]) == []


def test_nested_for_loop_induction_values_remain_lexically_scoped() -> None:
    """Keep nested induction variables distinct while retaining outer captures."""
    circuit = QuantumCircuit(1)
    with circuit.for_loop(range(2), None, None, None, None, label=None) as outer:
        circuit.rz(outer, 0)
        with circuit.for_loop(range(4, 0, -2), None, None, None, None, label=None) as inner:
            circuit.rx(inner, 0)
        circuit.ry(outer, 0)

    restored = QCProgram.from_qiskit(circuit).to_qiskit()

    outer_loop = restored.data[0].operation
    outer_body = outer_loop.blocks[0]
    inner_loop = outer_body.data[1].operation
    outer_parameter = outer_loop.params[1]
    inner_parameter = inner_loop.params[1]
    assert outer_parameter.uuid != inner_parameter.uuid
    assert outer_body.data[0].operation.params[0].uuid == outer_parameter.uuid
    assert outer_body.data[2].operation.params[0].uuid == outer_parameter.uuid
    assert inner_loop.blocks[0].data[0].operation.params[0].uuid == inner_parameter.uuid


def test_generated_loop_parameter_name_avoids_free_symbol_collision() -> None:
    """Choose a loop symbol name distinct from every free program input."""
    free = Parameter("_mqt_loop_0")
    circuit = QuantumCircuit(1)
    circuit.rz(free, 0)
    with circuit.for_loop(range(2), None, None, None, None, label=None) as iteration:
        circuit.rx(iteration, 0)

    restored = compile_program(circuit).to_qiskit()

    assert restored.data[0].operation.params[0].name == "_mqt_loop_0"
    loop = restored.data[1].operation
    assert loop.params[1].name == "_mqt_loop_1"
    assert loop.blocks[0].data[0].operation.params[0].uuid == loop.params[1].uuid


@pytest.mark.parametrize(
    "dead_use",
    ["", "%unused = math.sin %parameter : f64"],
    ids=["direct", "transitive"],
)
def test_dead_for_loop_parameter_projection_is_ignored(dead_use: str) -> None:
    """Omit a loop symbol whose projection has no emitted parameter use."""
    program = _single_qubit_program([
        "%lower = arith.constant 0 : index",
        "%upper = arith.constant 2 : index",
        "%step = arith.constant 1 : index",
        "scf.for %iteration = %lower to %upper step %step {",
        "  %integer = arith.index_cast %iteration : index to i64",
        "  %parameter = arith.sitofp %integer : i64 to f64",
        f"  {dead_use}",
        "  qc.x %q : !qc.qubit",
        "}",
    ])

    restored = program.to_qiskit()

    loop = restored.data[0].operation
    assert loop.name == "for_loop"
    assert loop.params[1] is None
    assert loop.blocks[0].count_ops() == {"x": 1}


def test_switch_case_label_width_is_preflighted() -> None:
    """Reject a switch label that cannot fit its one-bit target."""
    program = _single_qubit_program(
        [
            '%classical = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "c"} : !cbit.reg<1>',
            "%zero = arith.constant 0 : index",
            "%bit = cbit.load %classical[%zero] : !cbit.reg<1>",
            "%index = arith.index_castui %bit : i1 to index",
            "scf.index_switch %index",
            "case 2 {",
            "  qc.x %q : !qc.qubit",
            "  scf.yield",
            "}",
            "default {",
            "  scf.yield",
            "}",
        ],
        returns_classical=True,
    )
    with pytest.raises(RuntimeError, match="case label 2 does not fit the 1-bit target"):
        program.to_qiskit()


def test_constant_index_switch_exports() -> None:
    """Lift a direct constant index selector into a Qiskit Uint expression."""
    program = _single_qubit_program([
        "%selector = arith.constant 0 : index",
        "scf.index_switch %selector",
        "case 0 {",
        "  qc.x %q : !qc.qubit",
        "  scf.yield",
        "}",
        "default {",
        "  qc.z %q : !qc.qubit",
        "  scf.yield",
        "}",
    ])

    restored = program.to_qiskit()
    switch = restored.data[0].operation
    expected = expr.lift(0, types.Uint(64))
    assert isinstance(switch.target, expr.Expr)
    assert expr.structurally_equivalent(switch.target, expected)
    assert [labels for labels, _ in switch.cases_specifier()] == [(0,), (CASE_DEFAULT,)]


def test_shared_expression_dag_expansion_is_bounded() -> None:
    """Bound tree expansion when both operands reuse the same SSA value."""
    operations = [
        '%classical = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "c"} : !cbit.reg<1>',
        "%zero = arith.constant 0 : index",
        "%value0 = cbit.load %classical[%zero] : !cbit.reg<1>",
    ]
    operations.extend(f"%value{index} = arith.andi %value{index - 1}, %value{index - 1} : i1" for index in range(1, 14))
    operations.extend(["scf.if %value13 {", "  qc.x %q : !qc.qubit", "}"])
    program = _single_qubit_program(operations, returns_classical=True)
    with pytest.raises(RuntimeError, match="size limit of 4096 nodes"):
        program.to_qiskit()


def test_shared_packed_register_candidate_expansion_is_bounded() -> None:
    """Bound speculative packed-register matching on a shared SSA DAG."""
    operations = ["%value0 = arith.constant 0 : i64"]
    operations.extend(f"%value{index} = arith.ori %value{index - 1}, %value{index - 1} : i64" for index in range(1, 31))
    operations.extend([
        "%condition = arith.cmpi eq, %value30, %value0 : i64",
        "scf.if %condition {",
        "  qc.x %q : !qc.qubit",
        "}",
    ])
    program = _single_qubit_program(operations)
    with pytest.raises(RuntimeError, match="size limit of 4096 nodes"):
        program.to_qiskit()


def test_classical_snapshot_walk_is_bounded() -> None:
    """Bound snapshot discovery before recursive expression export."""
    operations = [
        '%classical = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "c"} : !cbit.reg<1>',
        "%zero = arith.constant 0 : index",
        "%value0 = cbit.load %classical[%zero] : !cbit.reg<1>",
    ]
    operations.extend(f"%value{index} = arith.andi %value{index - 1}, %value0 : i1" for index in range(1, 4097))
    operations.extend(["scf.if %value4096 {", "  qc.x %q : !qc.qubit", "}"])
    program = _single_qubit_program(operations, returns_classical=True)
    with pytest.raises(RuntimeError, match="size limit of 4096 nodes"):
        program.to_qiskit()


def test_export_expression_depth_is_bounded() -> None:
    """Reject a classical expression deeper than 64 levels during export."""
    operations = [
        '%classical = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "c"} : !cbit.reg<1>',
        "%zero = arith.constant 0 : index",
        "%true = arith.constant true",
        "%value0 = cbit.load %classical[%zero] : !cbit.reg<1>",
    ]
    operations.extend(f"%value{index} = arith.andi %value{index - 1}, %true : i1" for index in range(1, 65))
    operations.extend(["scf.if %value64 {", "  qc.x %q : !qc.qubit", "}"])
    program = _single_qubit_program(operations, returns_classical=True)

    with pytest.raises(RuntimeError, match="classical expressions exceed the nesting limit of 64"):
        program.to_qiskit()


def test_general_boolean_select_is_rejected() -> None:
    """Reject a result-bearing scf.if that is not short-circuit logic."""
    program = _single_qubit_program(
        [
            '%classical = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "c"} : !cbit.reg<1>',
            "%zero = arith.constant 0 : index",
            "%condition = cbit.load %classical[%zero] : !cbit.reg<1>",
            "%selected = scf.if %condition -> (i1) {",
            "  %then = cbit.load %classical[%zero] : !cbit.reg<1>",
            "  scf.yield %then : i1",
            "} else {",
            "  %else = cbit.load %classical[%zero] : !cbit.reg<1>",
            "  scf.yield %else : i1",
            "}",
            "scf.if %selected {",
            "  qc.x %q : !qc.qubit",
            "}",
        ],
        returns_classical=True,
    )

    with pytest.raises(RuntimeError, match=r"canonical short-circuit Boolean scf\.if"):
        program.to_qiskit()


def test_export_control_flow_depth_is_bounded() -> None:
    """Reject structured control flow deeper than 64 levels during export."""
    operations = ["%condition = arith.constant true"]
    operations.extend(f"{'  ' * depth}scf.if %condition {{" for depth in range(65))
    operations.append(f"{'  ' * 65}qc.x %q : !qc.qubit")
    operations.extend(f"{'  ' * depth}}}" for depth in reversed(range(65)))
    program = _single_qubit_program(operations)

    with pytest.raises(RuntimeError, match="control flow exceeds the nesting limit of 64"):
        program.to_qiskit()


def test_nonboolean_result_bearing_if_is_rejected() -> None:
    """Reject a result-bearing scf.if whose result is not Boolean."""
    program = _single_qubit_program([
        "%condition = arith.constant true",
        "%result = scf.if %condition -> (i64) {",
        "  %one = arith.constant 1 : i64",
        "  scf.yield %one : i64",
        "} else {",
        "  %zero = arith.constant 0 : i64",
        "  scf.yield %zero : i64",
        "}",
        "qc.x %q : !qc.qubit",
    ])
    with pytest.raises(RuntimeError, match="canonical short-circuit Boolean SSA result"):
        program.to_qiskit()


@pytest.mark.parametrize(
    "write_operations",
    [
        (
            "%measured = qc.measure %q : !qc.qubit -> i1",
            "cbit.store %measured, %classical[%zero] : !cbit.reg<1>",
        ),
        (
            "%always = arith.constant true",
            "scf.if %always {",
            "  %measured = qc.measure %q : !qc.qubit -> i1",
            "  cbit.store %measured, %classical[%zero] : !cbit.reg<1>",
            "}",
        ),
    ],
    ids=["flat-write", "nested-write"],
)
def test_stale_classical_snapshot_is_rejected(write_operations: tuple[str, ...]) -> None:
    """Reject a condition whose classical snapshot crosses a later write."""
    program = _single_qubit_program(
        [
            '%classical = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "c"} : !cbit.reg<1>',
            "%zero = arith.constant 0 : index",
            "%stale = cbit.load %classical[%zero] : !cbit.reg<1>",
            *write_operations,
            "scf.if %stale {",
            "  qc.x %q : !qc.qubit",
            "}",
        ],
        returns_classical=True,
    )
    with pytest.raises(RuntimeError, match=r"cannot preserve a (?:stale )?classical snapshot"):
        program.to_qiskit()


def test_delayed_measurement_store_is_rejected() -> None:
    """Reject a delayed write that would change a captured bit snapshot."""
    program = QCProgram.from_mlir_str(
        """module {
  func.func @main() -> !cbit.reg<1> attributes {mqt.entry_point} {
    %measured_qubit = qc.alloc : !qc.qubit
    %controlled_qubit = qc.alloc : !qc.qubit
    %classical = cbit.alloc(#cbit.init<zero>) {mqt.register_name = "c"} : !cbit.reg<1>
    %zero = arith.constant 0 : index
    %old = cbit.load %classical[%zero] : !cbit.reg<1>
    %measured = qc.measure %measured_qubit : !qc.qubit -> i1
    scf.if %old {
      qc.x %controlled_qubit : !qc.qubit
    }
    cbit.store %measured, %classical[%zero] : !cbit.reg<1>
    qc.dealloc %measured_qubit : !qc.qubit
    qc.dealloc %controlled_qubit : !qc.qubit
    return %classical : !cbit.reg<1>
  }
}
"""
    )
    with pytest.raises(RuntimeError, match="destination must follow the measurement"):
        program.to_qiskit()


def test_multi_result_boolean_select_is_rejected() -> None:
    """Reject multiple results instead of reconstructing Boolean selections."""
    program = _single_qubit_program([
        "%condition = arith.constant true",
        "%first, %second = scf.if %condition -> (i1, i1) {",
        "  %true = arith.constant true",
        "  %false = arith.constant false",
        "  scf.yield %true, %false : i1, i1",
        "} else {",
        "  %true = arith.constant true",
        "  %false = arith.constant false",
        "  scf.yield %false, %true : i1, i1",
        "}",
        "scf.if %first {",
        "  qc.x %q : !qc.qubit",
        "}",
    ])

    with pytest.raises(RuntimeError, match="only one canonical short-circuit Boolean SSA result"):
        program.to_qiskit()


def _undefined_cbit_program(operations: list[str]) -> QCProgram:
    """Build a one-qubit program with one undefined public CBit.

    Returns:
        The parsed QC program.
    """
    return _single_qubit_program(
        [
            '%classical = cbit.alloc(#cbit.init<undefined>) {mqt.register_name = "c"} : !cbit.reg<1>',
            "%zero = arith.constant 0 : index",
            *operations,
        ],
        returns_classical=True,
    )


def test_undefined_cbits_can_be_read_after_unconditional_measurements() -> None:
    """Treat preceding top-level measurement writes as definite initialization."""
    program = _undefined_cbit_program([
        "%measured = qc.measure %q : !qc.qubit -> i1",
        "cbit.store %measured, %classical[%zero] : !cbit.reg<1>",
        "%condition = cbit.load %classical[%zero] : !cbit.reg<1>",
        "scf.if %condition {",
        "  qc.x %q : !qc.qubit",
        "}",
    ])

    restored = program.to_qiskit()

    assert [instruction.operation.name for instruction in restored.data] == ["measure", "if_else"]


def test_undefined_cbit_load_before_measurement_is_rejected() -> None:
    """Reject a read that precedes definite initialization of an output bit."""
    program = _undefined_cbit_program([
        "%condition = cbit.load %classical[%zero] : !cbit.reg<1>",
        "scf.if %condition {",
        "  qc.x %q : !qc.qubit",
        "}",
        "%measured = qc.measure %q : !qc.qubit -> i1",
        "cbit.store %measured, %classical[%zero] : !cbit.reg<1>",
    ])

    with pytest.raises(RuntimeError, match="loads an undefined classical bit"):
        program.to_qiskit()


def test_conditional_measurement_does_not_initialize_returned_cbit() -> None:
    """Do not count a branch-local measurement as a definite output write."""
    program = _undefined_cbit_program([
        "%condition = arith.constant true",
        "scf.if %condition {",
        "  %measured = qc.measure %q : !qc.qubit -> i1",
        "  cbit.store %measured, %classical[%zero] : !cbit.reg<1>",
        "}",
    ])

    with pytest.raises(RuntimeError, match="cannot return undefined classical bits"):
        program.to_qiskit()


@pytest.mark.parametrize(
    ("expression", "error"),
    [
        (
            """%left = arith.constant 5 : i8
    %right = arith.constant 2 : i8
    %remainder = arith.remui %left, %right : i8
    %expected = arith.constant 1 : i8
    %condition = arith.cmpi eq, %remainder, %expected : i8""",
            "unsupported QC classical operation in Qiskit export: arith.remui",
        ),
        (
            """%left = arith.constant 0 : i65
    %right = arith.constant 1 : i65
    %condition = arith.cmpi eq, %left, %right : i65""",
            "unsigned classical values must be between 1 and 64 bits",
        ),
        (
            """%left = arith.constant 0 : i8
    %right = arith.constant 1 : i8
    %condition = arith.cmpi slt, %left, %right : i8""",
            "Uint expressions do not support signed comparisons",
        ),
        (
            """%infinity = arith.constant 0x7FF0000000000000 : f64
    %zero = arith.constant 0.0 : f64
    %condition = arith.cmpf oeq, %infinity, %zero : f64""",
            "floating-point literals must be finite",
        ),
    ],
    ids=["unsupported-op", "width", "signed-compare", "nonfinite"],
)
def test_unsupported_export_expressions_fail_closed(expression: str, error: str) -> None:
    """Reject unsupported expression forms before modifying the source program."""
    program = _single_qubit_program([
        *expression.splitlines(),
        "scf.if %condition {",
        "  qc.x %q : !qc.qubit",
        "}",
    ])
    source = program.ir

    with pytest.raises(RuntimeError, match=error):
        program.to_qiskit()

    assert program.ir == source


def test_qiskit_import_zero_initializes_clbits_before_control_flow() -> None:
    """Initialize Qiskit clbits before a condition reads them."""
    circuit = QuantumCircuit(1, 1)
    with circuit.if_test((circuit.clbits[0], False)):
        circuit.x(0)

    ir = QCProgram.from_qiskit(circuit).ir

    initialization = ir.index("cbit.alloc(#cbit.init<zero>)")
    condition_load = ir.index("cbit.load", initialization)
    assert initialization < condition_load
    assert "memref.store" not in ir


@pytest.mark.parametrize(
    ("condition", "operation"),
    [
        (expr.logic_and(expr.equal(1, 1), expr.equal(0, 1)), "scf.if"),
        (expr.equal(expr.bit_and(expr.lift(2, types.Uint(8)), 3), 2), "arith.andi"),
        (expr.equal(expr.bit_xor(expr.lift(2, types.Uint(8)), 3), 5), "arith.xori"),
        (expr.less(expr.add(expr.lift(2, types.Uint(8)), 1), 8), "arith.addi"),
        (
            expr.greater(expr.cast(expr.lift(2, types.Uint(8)), types.Float()), 0.5),
            "arith.uitofp",
        ),
        (expr.cast(expr.lift(0.5, types.Float()), types.Bool()), "arith.cmpf une"),
        (expr.greater(expr.negate(expr.lift(0.5, types.Float())), -1.0), "arith.negf"),
    ],
)
def test_bool_uint_and_float_expressions(condition: expr.Expr, operation: str) -> None:
    """Round-trip representative Bool, Uint, and Float expressions."""
    circuit = QuantumCircuit(1)
    with circuit.if_test(condition):
        circuit.x(0)

    program = QCProgram.from_qiskit(circuit)
    restored = program.to_qiskit()

    assert operation in program.ir
    assert restored.data[0].operation.name == "if_else"
    restored_condition = restored.data[0].operation.condition
    assert isinstance(restored_condition, expr.Expr)
    if operation == "scf.if":
        expected = expr.logic_and(
            expr.equal(True, True),  # ruff: ignore[boolean-positional-value-in-call] Qiskit expression arguments are positional-only.
            expr.equal(False, True),  # ruff: ignore[boolean-positional-value-in-call] Qiskit expression arguments are positional-only.
        )
    elif operation == "arith.cmpf une":
        expected = expr.not_equal(expr.lift(0.5, types.Float()), 0.0)
    else:
        expected = condition
    assert expr.structurally_equivalent(restored_condition, expected)


def test_index_expression_export_preserves_low_bit() -> None:
    """Export integer truncation as bit indexing instead of a truthiness cast."""
    condition = expr.index(expr.lift(2, types.Uint(3)), expr.lift(0, types.Uint(3)))
    circuit = QuantumCircuit(1)
    with circuit.if_test(condition):
        circuit.x(0)

    program = QCProgram.from_qiskit(circuit)
    assert "arith.trunci" in program.ir

    restored = program.to_qiskit()
    restored_condition = restored.data[0].operation.condition
    assert isinstance(restored_condition, expr.Expr)
    assert expr.structurally_equivalent(restored_condition, condition)


def test_integer_truncation_exports_as_low_bit_index() -> None:
    """Preserve the low-bit semantics of a generic integer truncation."""
    program = _single_qubit_program([
        "%two = arith.constant 2 : i3",
        "%condition = arith.trunci %two : i3 to i1",
        "scf.if %condition {",
        "  qc.x %q : !qc.qubit",
        "}",
    ])

    restored = program.to_qiskit()
    restored_condition = restored.data[0].operation.condition
    expected = expr.index(expr.lift(2, types.Uint(3)), expr.lift(0, types.Uint(3)))
    assert isinstance(restored_condition, expr.Expr)
    assert expr.structurally_equivalent(restored_condition, expected)


def _cbit_load_indices(ir: str) -> list[int]:
    """Extract the constant indices used by CBit loads.

    Args:
        ir: MLIR text to inspect.

    Returns:
        The CBit load indices in occurrence order.
    """
    constants = {
        name: int(value) for name, value in re.findall(r"(?m)^\s*(%[-\w.$]+) = arith\.constant (\d+) : index$", ir)
    }
    return [constants[name] for name in re.findall(r"(?m)^\s*%[-\w.$]+ = cbit\.load [^\[]+\[(%[-\w.$]+)\]", ir)]


def test_boolean_expression_literals_are_imported() -> None:
    """Normalize Qiskit's integer-backed Boolean Value nodes."""
    false_literal = False
    true_literal = True
    circuit = QuantumCircuit(1)
    with circuit.if_test(expr.logic_or(expr.lift(false_literal), expr.lift(true_literal))):
        circuit.x(0)

    ir = QCProgram.from_qiskit(circuit).ir

    assert "arith.constant false" in ir
    assert "arith.constant true" in ir
    assert "scf.if" in ir
    assert "arith.ori" not in ir


def test_uint_register_cast_to_bool_tests_all_bits() -> None:
    """Treat a Uint register as true when any bit is set."""
    circuit = QuantumCircuit(1, 2)
    circuit.x(0)
    circuit.measure(0, 1)
    with circuit.if_test(expr.cast(circuit.cregs[0], types.Bool())):
        circuit.z(0)

    program = QCProgram.from_qiskit(circuit)
    ir = program.ir

    assert "arith.cmpi ne" in ir
    assert "arith.trunci" not in ir

    restored = program.to_qiskit()
    round_trip_ir = QCProgram.from_qiskit(restored).ir
    assert "arith.cmpi ne" in round_trip_ir
    assert "arith.trunci" not in round_trip_ir


def test_public_expression_condition_mutation_is_observed() -> None:
    """Import the current public expression after condition mutation."""
    circuit = QuantumCircuit(1, 2)
    with circuit.if_test(expr.logic_and(circuit.clbits[0], circuit.clbits[1])):
        circuit.x(0)
    operation = circuit.data[0].operation
    assert isinstance(operation, IfElseOp)
    operation.condition = expr.logic_or(circuit.clbits[0], circuit.clbits[1])

    program = QCProgram.from_qiskit(circuit)
    restored = program.to_qiskit()
    condition = restored.data[0].operation.condition

    assert "scf.if" in program.ir
    assert isinstance(condition, expr.Expr)
    assert expr.structurally_equivalent(condition, expr.logic_or(*restored.clbits))


def test_public_tuple_condition_mutation_is_observed() -> None:
    """Import the current bit and value after tuple-condition mutation."""
    body = QuantumCircuit(1)
    body.x(0)
    circuit = QuantumCircuit(1, 2)
    circuit.if_test((circuit.clbits[0], False), body, circuit.qubits, [])
    operation = circuit.data[0].operation
    assert isinstance(operation, IfElseOp)
    operation.condition = (circuit.clbits[1], True)

    ir = QCProgram.from_qiskit(circuit).ir

    assert _cbit_load_indices(ir) == [1]
    assert "arith.constant true" in ir
    assert "arith.constant false" not in ir


def test_narrow_uint_switch_literal_is_rejected() -> None:
    """Reject a Uint literal that does not fit its declared width."""
    circuit = QuantumCircuit(1, 1)
    with circuit.switch(expr.Value(3, types.Uint(1)), None, None, None, label=None) as case, case(0):
        circuit.x(0)

    with pytest.raises(RuntimeError, match=r"Uint literal.*does not fit"):
        QCProgram.from_qiskit(circuit)


def test_malformed_public_expression_type_is_rejected() -> None:
    """Reject a public expression whose declared result type is inconsistent."""
    invalid = expr.Binary(
        expr.Binary.Op.ADD,
        expr.Value(1, types.Uint(1)),
        expr.Value(1, types.Uint(1)),
        types.Bool(),
    )
    circuit = QuantumCircuit(1)
    with circuit.if_test(expr.equal(1, 1)):
        circuit.x(0)
    operation = circuit.data[0].operation
    assert isinstance(operation, IfElseOp)
    operation.condition = invalid

    with pytest.raises(RuntimeError, match="incompatible operator and operand types"):
        QCProgram.from_qiskit(circuit)


def test_classical_expression_clbit_captures_import() -> None:
    """Keep Clbit identity when an expression capture uses a nontrivial order."""
    circuit = QuantumCircuit(1, 2)
    condition = expr.logic_and(circuit.clbits[1], expr.logic_not(circuit.clbits[0]))
    with circuit.if_test(condition):
        circuit.x(0)

    ir = QCProgram.from_qiskit(circuit).ir

    assert _cbit_load_indices(ir) == [1, 0]
    assert "arith.xori" in ir
    assert "scf.if" in ir
    assert "scf.if" in ir


def test_classical_expression_register_captures_round_trip_on_import() -> None:
    """Pack a captured register in Qiskit's little-endian bit order."""
    circuit = QuantumCircuit(1, 3)
    condition = expr.equal(expr.bit_xor(circuit.cregs[0], 1), 5)
    with circuit.if_test(condition):
        circuit.x(0)

    program = QCProgram.from_qiskit(circuit)
    assert QCProgram.from_mlir_str(program.ir).ir == program.ir
    ir = program.ir

    assert _cbit_load_indices(ir) == [0, 1, 2]
    assert ir.count("arith.shli") == 2
    assert "arith.xori" in ir
    assert "arith.cmpi eq" in ir


def test_nested_classical_expression_captures_import() -> None:
    """Compose nested local capture maps without changing root Clbit identity."""
    circuit = QuantumCircuit(1, 3)
    with circuit.if_test(expr.logic_not(circuit.clbits[2])):
        condition = expr.logic_and(circuit.clbits[0], expr.logic_not(circuit.clbits[1]))
        with circuit.while_loop(condition, None, None, None, label=None):
            circuit.x(0)

    ir = QCProgram.from_qiskit(circuit).ir

    assert _cbit_load_indices(ir) == [2, 0, 1]
    assert "scf.if" in ir
    assert "scf.while" in ir


def test_switch_expression_captures_import() -> None:
    """Read an expression switch target through Qiskit's public Python tree."""
    circuit = QuantumCircuit(1, 2)
    with circuit.switch(expr.bit_xor(circuit.cregs[0], 1), None, None, None, label=None) as case:
        with case(0):
            circuit.x(0)
        with case(case.DEFAULT):
            circuit.h(0)

    ir = QCProgram.from_qiskit(circuit).ir

    assert _cbit_load_indices(ir) == [0, 1]
    assert "arith.xori" in ir
    assert "scf.index_switch" in ir


def test_condition_only_clbit_expression_imports() -> None:
    """Resolve a condition bit that no control-flow block uses."""
    body = QuantumCircuit(1)
    body.x(0)
    circuit = QuantumCircuit(1, 1)
    circuit.if_test(expr.logic_not(circuit.clbits[0]), body, [circuit.qubits[0]], [])

    assert len(circuit.data[0].clbits) == 0

    ir = QCProgram.from_qiskit(circuit).ir

    assert _cbit_load_indices(ir) == [0]
    assert "arith.xori" in ir
    assert "scf.if" in ir


def test_condition_only_switch_expression_imports() -> None:
    """Resolve a switch register that no case block uses."""
    zero = QuantumCircuit(1)
    zero.x(0)
    default = QuantumCircuit(1)
    default.h(0)
    circuit = QuantumCircuit(1, 2)
    # Qiskit's overload omits expression targets although its runtime accepts them.
    circuit.switch(  # ty: ignore[no-matching-overload]
        expr.bit_xor(circuit.cregs[0], 1),
        [(0, zero), (CASE_DEFAULT, default)],
        [circuit.qubits[0]],
        [],
    )

    assert len(circuit.data[0].clbits) == 0
    assert all(block.num_clbits == 0 for block in circuit.data[0].operation.blocks)

    ir = QCProgram.from_qiskit(circuit).ir

    assert _cbit_load_indices(ir) == [0, 1]
    assert "arith.xori" in ir
    assert "scf.index_switch" in ir


def test_nested_condition_only_expression_uses_parent_capture_map() -> None:
    """Map a nested condition-only bit through its enclosing block."""
    inner_body = QuantumCircuit(1)
    inner_body.x(0)
    middle = QuantumCircuit(1, 2)
    middle.if_test(expr.logic_not(middle.clbits[0]), inner_body, [middle.qubits[0]], [])
    circuit = QuantumCircuit(1, 2)
    circuit.if_test(
        (circuit.clbits[0], True),
        middle,
        [circuit.qubits[0]],
        [circuit.clbits[1], circuit.clbits[0]],
    )

    ir = QCProgram.from_qiskit(circuit).ir

    assert _cbit_load_indices(ir) == [0, 1]
    assert ir.count("scf.if") == 2


def test_nested_legacy_clbit_condition_uses_root_index() -> None:
    """Resolve a nested tuple condition through its enclosing Clbit map."""
    circuit = QuantumCircuit(2, 2)
    with circuit.for_loop(range(2), None, None, None, None, label=None) as iteration:
        circuit.rx(iteration, 0)
        with circuit.if_test((circuit.clbits[1], True)):
            circuit.x(0)

    ir = QCProgram.from_qiskit(circuit).ir

    assert _cbit_load_indices(ir) == [1]
    assert "scf.for" in ir
    assert "scf.if" in ir


def test_classical_expression_rejects_mismatched_instruction_captures() -> None:
    """Reject an instruction capture list that does not match its block."""
    circuit = QuantumCircuit(1, 1)
    with circuit.if_test(expr.logic_not(circuit.clbits[0])):
        circuit.x(0)
    instruction = circuit.data[0]
    circuit._data[0] = instruction.replace(clbits=())  # ruff: ignore[private-member-access]

    with pytest.raises(RuntimeError, match="incompatible classical-bit captures"):
        QCProgram.from_qiskit(circuit)


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


def test_oversized_classical_expression_is_rejected() -> None:
    """Bound the total size of a balanced classical expression."""
    level = [expr.equal(1, 1) for _ in range(1025)]
    while len(level) > 1:
        level = [
            expr.logic_or(level[index], level[index + 1]) if index + 1 < len(level) else level[index]
            for index in range(0, len(level), 2)
        ]
    circuit = QuantumCircuit(1)
    with circuit.if_test(level[0]):
        circuit.x(0)
    source_data = list(circuit.data)

    with pytest.raises(RuntimeError, match="expressions exceed the node limit of 4096"):
        QCProgram.from_qiskit(circuit)

    assert list(circuit.data) == source_data


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


def test_direct_symbolic_parameters_round_trip_with_shared_identity() -> None:
    """Represent a shared Qiskit parameter as one named f64 input."""
    theta = Parameter("theta")
    circuit = QuantumCircuit(1, global_phase=theta)
    circuit.ry(theta, 0)
    circuit.rz(theta, 0)

    program = QCProgram.from_qiskit(circuit)
    restored = program.to_qiskit()

    assert 'mqt.input_name = "theta"' in program.ir
    assert len(restored.parameters) == 1
    restored_theta = next(iter(restored.parameters))
    assert restored.global_phase == restored_theta
    assert restored.data[0].operation.params[0] == restored_theta
    assert restored.data[1].operation.params[0] == restored_theta
    value = 0.375
    assert np.allclose(
        Operator(restored.assign_parameters({restored_theta: value})).data,
        Operator(circuit.assign_parameters({theta: value})).data,
    )


def test_sparse_parameter_vector_round_trip_preserves_order_and_binding() -> None:
    """Preserve a sparse vector's grouping, size, and numeric element order."""
    vector = ParameterVector("theta", 12)
    circuit = QuantumCircuit(1, global_phase=vector[0])
    circuit.rx(vector[10] + vector[2], 0)

    restored = QCProgram.from_qiskit(circuit).to_qiskit()

    parameters = list(restored.parameters)
    assert [parameter.index for parameter in parameters] == [0, 2, 10]
    restored_vector = parameters[0].vector
    assert len(restored_vector) == len(vector)
    values = [0.01 * index for index in range(12)]
    assert Operator(restored.assign_parameters({restored_vector: values}, strict=False)).equiv(
        Operator(circuit.assign_parameters({vector: values}, strict=False))
    )


def test_parameter_vector_is_shared_across_sibling_blocks() -> None:
    """Restore one vector across parent and sibling control-flow blocks."""
    vector = ParameterVector("theta", 2)
    circuit = QuantumCircuit(1, 1)
    circuit.rz(vector[0], 0)
    with circuit.if_test((circuit.clbits[0], True)) as else_:
        circuit.rx(vector[0], 0)
    with else_:
        circuit.ry(vector[1], 0)

    restored = QCProgram.from_qiskit(circuit).to_qiskit()

    root_parameter = restored.data[0].operation.params[0]
    blocks = restored.data[1].operation.blocks
    parameters = [root_parameter, *(block.data[0].operation.params[0] for block in blocks)]
    assert [parameter.index for parameter in parameters] == [0, 0, 1]
    restored.assign_parameters({parameters[0].vector: [0.25, 0.5]}, inplace=True)
    assert not restored.parameters


def test_distinct_parameter_vectors_with_the_same_name_remain_distinct() -> None:
    """Keep opaque group identity when vector display names coincide."""
    first = ParameterVector("theta", 3)
    second = ParameterVector("theta", 3)
    circuit = QuantumCircuit(1)
    circuit.rx(first[0], 0)
    circuit.ry(second[2], 0)

    restored = QCProgram.from_qiskit(circuit).to_qiskit()

    parameters = list(restored.parameters)
    assert [parameter.index for parameter in parameters] == [0, 2]
    assert parameters[0].vector.uuid != parameters[1].vector.uuid
    assert Operator(restored.assign_parameters([0.2, 0.7])).equiv(Operator(circuit.assign_parameters([0.2, 0.7])))


def test_standalone_bracket_parameter_names_remain_standalone() -> None:
    """Do not infer an input group from a standalone parameter's name."""
    theta_ten = Parameter("theta[10]")
    theta_two = Parameter("theta[2]")
    circuit = QuantumCircuit(1)
    circuit.rx(theta_ten, 0)
    circuit.ry(theta_two, 0)

    restored = QCProgram.from_qiskit(circuit).to_qiskit()

    assert all(not isinstance(parameter, ParameterVectorElement) for parameter in restored.parameters)
    assert {parameter.name for parameter in restored.parameters} == {"theta[2]", "theta[10]"}
    values = [0.1, 0.2]
    assert Operator(restored.assign_parameters(values)).equiv(Operator(circuit.assign_parameters(values)))


def test_parameter_vector_element_is_valid_loop_parameter() -> None:
    """Preserve a vector-element loop symbol as a lexical parameter."""
    iteration = ParameterVector("iteration", 4)[2]
    body = QuantumCircuit(1)
    body.rx(iteration, 0)
    circuit = QuantumCircuit(1)
    circuit.for_loop(range(3), iteration, body, [0], [], label=None)

    program = QCProgram.from_qiskit(circuit)
    assert program.ir.count("mqt.parameter_group") == 1
    restored_circuits = (
        program.to_qiskit(),
        program.to_qco(copy=True).to_qc().to_qiskit(),
    )
    for restored in restored_circuits:
        restored_loop = restored.data[0].operation
        restored_parameter = restored_loop.params[1]
        assert restored_parameter.vector.name == "iteration"
        assert restored_parameter.index == 2
        assert len(restored_parameter.vector) == 4
        assert restored_loop.blocks[0].data[0].operation.params[0] == restored_parameter
        assert not restored.parameters


@pytest.mark.parametrize(("size", "index"), [(0, 0), (1, 1)])
def test_parameter_vector_element_outside_current_size_round_trips(size: int, index: int) -> None:
    """Preserve a vector element outside its vector's current size."""
    vector = ParameterVector("theta", size)
    circuit = QuantumCircuit(1)
    circuit.rx(ParameterVectorElement(vector, index), 0)

    restored = QCProgram.from_qiskit(circuit).to_qiskit()

    restored_element = next(iter(restored.parameters))
    assert restored_element.index == index
    assert len(restored_element.vector) == size


@pytest.mark.parametrize("sizes", [[65_537], [32_769, 32_769]])
def test_parameter_vector_size_limits_on_import(sizes: list[int]) -> None:
    """Bound individual and aggregate vector metadata before MLIR creation."""
    circuit = QuantumCircuit(1)
    for index, size in enumerate(sizes):
        circuit.rx(ParameterVector(f"theta{index}", size)[0], 0)

    with pytest.raises(RuntimeError, match="across all distinct"):
        QCProgram.from_qiskit(circuit)


@pytest.mark.parametrize(
    ("sizes", "shared_group_id", "message"),
    [
        ([65_537], None, "across all distinct"),
        ([32_769, 32_769], None, "across all distinct"),
        ([1, 2], 0, "conflicting metadata"),
    ],
)
def test_parameter_vector_metadata_is_preflighted(sizes: list[int], shared_group_id: int | None, message: str) -> None:
    """Validate vector consistency and resource bounds before allocation."""
    arguments = []
    gates = []
    for index, size in enumerate(sizes):
        arguments.append(
            f'%theta{index}: f64 {{mqt.input_name = "theta{index}[0]", '
            f'mqt.parameter_group = {{identity = "group{index if shared_group_id is None else shared_group_id}", '
            f'name = "theta{index}", index = 0 : i64, size = {size} : i64}}}}'
        )
        gates.append(f"    qc.rx(%theta{index}) %q : !qc.qubit")
    program = QCProgram.from_mlir_str(
        "module {\n"
        f"  func.func @main({', '.join(arguments)}) attributes {{mqt.entry_point}} {{\n"
        "    %q = qc.alloc : !qc.qubit\n" + "\n".join(gates) + "\n    qc.dealloc %q : !qc.qubit\n"
        "    return\n"
        "  }\n"
        "}\n"
    )

    with pytest.raises(RuntimeError, match=message):
        program.to_qiskit()


def _assign_parameter_values(circuit: QuantumCircuit, values: dict[str, float]) -> QuantumCircuit:
    """Bind a circuit using parameter names after an import/export round trip.

    Returns:
        A copy of the circuit with all parameters bound.
    """
    return circuit.assign_parameters({parameter: values[parameter.name] for parameter in circuit.parameters})


def test_nested_symbolic_arithmetic_round_trip_with_shared_global_phase() -> None:
    """Preserve nested arithmetic and shared symbols in gates and global phase."""
    theta = Parameter("theta")
    phi = Parameter("phi")
    angle = -((2 - theta) * (phi.sin() + 0.25) / (theta**2 + 1))
    circuit = QuantumCircuit(1, global_phase=theta + phi)
    circuit.ry(angle, 0)
    circuit.rz(theta + phi, 0)

    program = QCProgram.from_qiskit(circuit)
    restored = program.to_qiskit()

    assert {parameter.name for parameter in restored.parameters} == {"phi", "theta"}
    assert len(restored.parameters) == 2
    values = {"phi": 0.4, "theta": -0.3}
    assert np.allclose(
        Operator(_assign_parameter_values(restored, values)).data,
        Operator(_assign_parameter_values(circuit, values)).data,
    )


@pytest.mark.parametrize(
    ("operation", "value"),
    [
        (lambda parameter: 2 + parameter, 0.25),
        (lambda parameter: 2 - parameter, 0.25),
        (lambda parameter: 2 * parameter, 0.25),
        (lambda parameter: 2 / parameter, 0.75),
        (lambda parameter: 2**parameter, -0.5),
    ],
    ids=["reverse-add", "reverse-subtract", "reverse-multiply", "reverse-divide", "reverse-power"],
)
def test_reverse_symbolic_arithmetic_round_trip(
    operation: Callable[[Parameter], ParameterExpression], value: float
) -> None:
    """Preserve Qiskit's reflected arithmetic operators."""
    theta = Parameter("theta")
    circuit = QuantumCircuit(1)
    circuit.rz(operation(theta), 0)

    restored = QCProgram.from_qiskit(circuit).to_qiskit()

    values = {"theta": value}
    assert np.allclose(
        Operator(_assign_parameter_values(restored, values)).data,
        Operator(_assign_parameter_values(circuit, values)).data,
    )


@pytest.mark.parametrize(
    ("operation", "value"),
    [
        (lambda parameter: parameter.sin(), 0.2),
        (lambda parameter: parameter.cos(), 0.2),
        (lambda parameter: parameter.tan(), 0.2),
        (lambda parameter: parameter.arcsin(), 0.2),
        (lambda parameter: parameter.arccos(), 0.2),
        (lambda parameter: parameter.arctan(), 0.2),
        (lambda parameter: parameter.exp(), 0.2),
        (lambda parameter: parameter.log(), 1.2),
        (abs, -0.2),
        (lambda parameter: parameter.conjugate(), 0.2),
    ],
    ids=["sin", "cos", "tan", "arcsin", "arccos", "arctan", "exp", "log", "abs", "conjugate"],
)
def test_symbolic_unary_function_round_trip(
    operation: Callable[[Parameter], ParameterExpression], value: float
) -> None:
    """Preserve supported unary Qiskit parameter functions."""
    theta = Parameter("theta")
    circuit = QuantumCircuit(1)
    circuit.rx(operation(theta), 0)

    restored = QCProgram.from_qiskit(circuit).to_qiskit()

    values = {"theta": value}
    assert np.allclose(
        Operator(_assign_parameter_values(restored, values)).data,
        Operator(_assign_parameter_values(circuit, values)).data,
    )


def test_partially_bound_symbolic_expression_round_trip() -> None:
    """Keep the unbound parameter after partially binding an expression."""
    theta = Parameter("theta")
    phi = Parameter("phi")
    angle = (theta * phi + phi.sin()).bind({theta: 0.5})
    circuit = QuantumCircuit(1)
    circuit.ry(angle, 0)

    restored = QCProgram.from_qiskit(circuit).to_qiskit()

    assert {parameter.name for parameter in restored.parameters} == {"phi"}
    values = {"phi": 0.4}
    assert np.allclose(
        Operator(_assign_parameter_values(restored, values)).data,
        Operator(_assign_parameter_values(circuit, values)).data,
    )


def test_float_castable_symbolic_expression_keeps_parameter_identity() -> None:
    """Do not collapse a float-castable expression that still tracks a symbol."""
    theta = Parameter("theta")
    angle = (theta - theta) + 2
    assert angle.parameters == {theta}
    assert float(angle) == pytest.approx(2)
    circuit = QuantumCircuit(1)
    circuit.rz(angle, 0)

    program = QCProgram.from_qiskit(circuit)
    restored = program.to_qiskit()

    assert 'mqt.input_name = "theta"' in program.ir
    assert {parameter.name for parameter in restored.parameters} == {"theta"}
    values = {"theta": 0.3}
    assert np.allclose(
        Operator(_assign_parameter_values(restored, values)).data,
        Operator(_assign_parameter_values(circuit, values)).data,
    )


def test_parameterized_custom_definition_round_trip() -> None:
    """Substitute symbolic call parameters while recursively inlining a definition."""
    formal = Parameter("formal")
    definition = QuantumCircuit(1)
    definition.rx(formal + 1, 0)
    custom = definition.to_gate(label="symbolic")
    circuit = QuantumCircuit(1)
    circuit.append(custom, [0])
    theta = Parameter("theta")
    circuit.assign_parameters({formal: theta + 0.25}, inplace=True)

    restored = QCProgram.from_qiskit(circuit).to_qiskit()

    assert {parameter.name for parameter in restored.parameters} == {"theta"}
    values = {"theta": -0.2}
    assert np.allclose(
        Operator(_assign_parameter_values(restored, values)).data,
        Operator(_assign_parameter_values(circuit, values)).data,
    )


def test_manual_arith_and_math_parameter_expression_exports_to_qiskit() -> None:
    """Reconstruct an expression from generic Arith and Math operations."""
    program = QCProgram.from_mlir_str(
        """module {
  func.func @main(%theta: f64 {mqt.input_name = "theta"}) attributes {mqt.entry_point} {
    %q = qc.alloc : !qc.qubit
    %offset = arith.constant 5.000000e-01 : f64
    %sum = arith.addf %theta, %offset : f64
    %angle = math.sin %sum : f64
    qc.rz(%angle) %q : !qc.qubit
    qc.dealloc %q : !qc.qubit
    return
  }
}
"""
    )

    restored = program.to_qiskit()

    theta = next(iter(restored.parameters))
    bound = restored.assign_parameters({theta: 0.25})
    assert bound.data[0].operation.params[0] == pytest.approx(np.sin(0.75))


def _wide_parameter_expression_program(term_count: int) -> QCProgram:
    """Build a program whose gate angle sums ``term_count`` distinct math.sin terms.

    Returns:
        A QC program with a wide scalar parameter-expression graph.
    """
    lines = [
        "module {",
        '  func.func @main(%theta: f64 {mqt.input_name = "theta"}) attributes {mqt.entry_point} {',
        "    %q = qc.alloc : !qc.qubit",
    ]
    values = []
    for index in range(term_count):
        value = f"%term{index}"
        lines.append(f"    {value} = math.sin %theta : f64")
        values.append(value)
    sum_index = 0
    while len(values) > 1:
        next_values = []
        for index in range(0, len(values), 2):
            if index + 1 == len(values):
                next_values.append(values[index])
                continue
            value = f"%sum{sum_index}"
            sum_index += 1
            lines.append(f"    {value} = arith.addf {values[index]}, {values[index + 1]} : f64")
            next_values.append(value)
        values = next_values
    lines.extend([
        f"    qc.rz({values[0]}) %q : !qc.qubit",
        "    qc.dealloc %q : !qc.qubit",
        "    return",
        "  }",
        "}",
    ])
    return QCProgram.from_mlir_str("\n".join(lines))


@pytest.mark.parametrize(
    "term_count",
    [1366, 2049],
    ids=["expanded-tree", "unique-ssa-graph"],
)
def test_oversized_export_parameter_expression_fails_without_mutation(term_count: int) -> None:
    """Bound normalized trees and compiler SSA traversal before Qiskit construction."""
    program = _wide_parameter_expression_program(term_count)
    source_ir = program.ir

    with pytest.raises(RuntimeError, match="exceeds the supported 4096-node size"):
        program.to_qiskit()

    assert program.ir == source_ir


def test_unsupported_scalar_operation_fails_export_without_mutation() -> None:
    """Reject an unsupported f64 producer before changing the source program."""
    program = QCProgram.from_mlir_str(
        """module {
  func.func @main(%theta: f64 {mqt.input_name = "theta"}) attributes {mqt.entry_point} {
    %q = qc.alloc : !qc.qubit
    %angle = math.sqrt %theta : f64
    qc.rz(%angle) %q : !qc.qubit
    qc.dealloc %q : !qc.qubit
    return
  }
}
"""
    )
    source_ir = program.ir

    with pytest.raises(
        RuntimeError,
        match=r"Qiskit circuit export does not support scalar parameter operation 'math\.sqrt'",
    ):
        program.to_qiskit()

    assert program.ir == source_ir


def test_same_name_free_and_bound_parameters_are_rejected() -> None:
    """Require free and lexically bound parameters to have unique names."""
    global_parameter = Parameter("theta")
    loop_parameter = Parameter("theta")
    body = QuantumCircuit(1)
    body.ry(global_parameter, 0)
    circuit = QuantumCircuit(1)
    with pytest.warns(UserWarning, match="loop_parameter was not found"):
        circuit.for_loop(range(2), loop_parameter, body, [0], [], label=None)
    source_data = list(circuit.data)

    with pytest.raises(RuntimeError, match="distinct parameters with the same name"):
        QCProgram.from_qiskit(circuit)

    assert list(circuit.data) == source_data
    assert circuit.parameters == {global_parameter}


def test_bound_parameter_names_are_unique_across_scopes() -> None:
    """Reject same-name binders in separate lexical scopes."""
    first = Parameter("index")
    first_body = QuantumCircuit(1)
    first_body.rx(first, 0)
    second = Parameter("index")
    second_body = QuantumCircuit(1)
    second_body.ry(second, 0)
    circuit = QuantumCircuit(1)
    circuit.for_loop(range(2), first, first_body, [0], [], label=None)
    circuit.for_loop(range(2), second, second_body, [0], [], label=None)
    source_data = list(circuit.data)

    with pytest.raises(RuntimeError, match="distinct parameters with the same name"):
        QCProgram.from_qiskit(circuit)

    assert list(circuit.data) == source_data
    assert not circuit.parameters


def test_duplicate_named_symbolic_inputs_are_invalid_qc_ir() -> None:
    """Reject duplicate program input names when parsing QC IR."""
    with pytest.raises(RuntimeError, match="MLIR operation failed"):
        QCProgram.from_mlir_str(
            """module {
  func.func @main(
      %first: f64 {mqt.input_name = "theta"},
      %second: f64 {mqt.input_name = "theta"}
  ) attributes {mqt.entry_point} {
    %q = qc.alloc : !qc.qubit
    qc.rx(%first) %q : !qc.qubit
    qc.rz(%second) %q : !qc.qubit
    qc.dealloc %q : !qc.qubit
    return
  }
}
"""
        )


def test_parameter_and_register_names_must_be_unique() -> None:
    """Reject a parameter and register that share one program name."""
    theta = Parameter("theta")
    register = QuantumRegister(1, "theta")
    circuit = QuantumCircuit(register)
    circuit.rx(theta, register[0])
    source_data = list(circuit.data)

    with pytest.raises(RuntimeError, match="unique parameter and register names"):
        QCProgram.from_qiskit(circuit)

    assert list(circuit.data) == source_data

    with pytest.raises(RuntimeError, match="MLIR operation failed"):
        QCProgram.from_mlir_str(
            """module {
  func.func @main(%theta: f64 {mqt.input_name = "theta"}) attributes {mqt.entry_point} {
    %q = memref.alloc() {mqt.register_name = "theta"} : memref<1x!qc.qubit>
    memref.dealloc %q : memref<1x!qc.qubit>
    return
  }
}
"""
        )


def test_parameter_names_with_null_characters_fail_closed() -> None:
    """Reject names that the Qiskit C API would silently truncate."""
    parameter = Parameter("before\0after")
    circuit = QuantumCircuit(1)
    circuit.rz(parameter, 0)
    source_data = list(circuit.data)

    with pytest.raises(RuntimeError, match="names cannot contain null characters"):
        QCProgram.from_qiskit(circuit)

    assert list(circuit.data) == source_data

    with pytest.raises(RuntimeError, match="MLIR operation failed"):
        QCProgram.from_mlir_str(
            r"""module {
  func.func @main(%theta: f64 {mqt.input_name = "before\00after"}) attributes {mqt.entry_point} {
    %q = qc.alloc : !qc.qubit
    qc.rz(%theta) %q : !qc.qubit
    qc.dealloc %q : !qc.qubit
    return
  }
}
"""
        )


def test_named_symbolic_input_exports_to_qiskit() -> None:
    """Reconstruct a direct Qiskit parameter from a named f64 input."""
    symbolic = QCProgram.from_mlir_str(
        """module {
  func.func @main(%theta: f64 {mqt.input_name = "theta"}) attributes {mqt.entry_point} {
    %q = qc.alloc : !qc.qubit
    qc.rx(%theta) %q : !qc.qubit
    qc.dealloc %q : !qc.qubit
    return
  }
}
"""
    )
    restored = symbolic.to_qiskit()

    assert [parameter.name for parameter in restored.parameters] == ["theta"]
    assert restored.data[0].operation.params[0] == next(iter(restored.parameters))


def test_unused_named_symbolic_input_fails_export_without_mutation() -> None:
    """Reject a compiler input that would disappear from the Qiskit circuit."""
    program = QCProgram.from_mlir_str(
        """module {
  func.func @main(%theta: f64 {mqt.input_name = "theta"}) attributes {mqt.entry_point} {
    %q = qc.alloc : !qc.qubit
    qc.x %q : !qc.qubit
    qc.dealloc %q : !qc.qubit
    return
  }
}
"""
    )
    source_ir = program.ir

    with pytest.raises(RuntimeError, match="cannot preserve unused named f64 program input 'theta'"):
        program.to_qiskit()

    assert program.ir == source_ir


def test_unnamed_runtime_input_is_rejected_on_export() -> None:
    """Do not infer source semantics for arbitrary runtime inputs."""
    runtime = QCProgram.from_mlir_str(
        """module {
  func.func @main(%theta: f64) attributes {mqt.entry_point} {
    %q = qc.alloc : !qc.qubit
    qc.rx(%theta) %q : !qc.qubit
    qc.dealloc %q : !qc.qubit
    return
  }
}
"""
    )

    with pytest.raises(RuntimeError, match="requires named f64 program inputs"):
        runtime.to_qiskit()


def test_named_non_f64_runtime_input_is_rejected_on_export() -> None:
    """Reject a named compiler input whose type cannot represent a parameter."""
    runtime = QCProgram.from_mlir_str(
        """module {
  func.func @main(%count: i64 {mqt.input_name = "count"}) attributes {mqt.entry_point} {
    %q = qc.alloc : !qc.qubit
    qc.x %q : !qc.qubit
    qc.dealloc %q : !qc.qubit
    return
  }
}
"""
    )
    source_ir = runtime.ir

    with pytest.raises(RuntimeError, match="requires named f64 program inputs"):
        runtime.to_qiskit()

    assert runtime.ir == source_ir


def test_target_aware_qiskit_export_maps_sparse_site_ids() -> None:
    """Map large sparse target site IDs to dense physical-qubit indices."""
    target = CompilerTarget(
        "sparse target",
        [CompilerTarget.Site(10), CompilerTarget.Site(4294967296)],
    )
    program = QCProgram.from_mlir_str(
        """module {
  func.func @main() attributes {mqt.entry_point} {
    %q = qc.static 4294967296 : !qc.qubit
    qc.x %q : !qc.qubit
    return
  }
}
"""
    )

    restored = program.to_qiskit(target=target)

    assert restored.num_qubits == 2
    assert [(register.name, len(register)) for register in restored.qregs] == [("q", 2)]
    assert restored.layout is None
    assert restored.data[0].operation.name == "x"
    assert restored.find_bit(restored.data[0].qubits[0]).index == 1


def test_target_aware_qiskit_export_rejects_unknown_site() -> None:
    """Reject a static site that is absent from the compiler target."""
    target = CompilerTarget(
        "sparse target",
        [CompilerTarget.Site(10), CompilerTarget.Site(20)],
    )
    program = QCProgram.from_mlir_str(
        """module {
  func.func @main() attributes {mqt.entry_point} {
    %q = qc.static 30 : !qc.qubit
    qc.x %q : !qc.qubit
    return
  }
}
"""
    )

    with pytest.raises(RuntimeError, match="QC static qubit is not a site of the supplied compiler target"):
        program.to_qiskit(target=target)


@pytest.mark.parametrize(
    "allocation",
    [
        """%q = qc.alloc : !qc.qubit
    qc.x %q : !qc.qubit
    qc.dealloc %q : !qc.qubit""",
        """%c0 = arith.constant 0 : index
    %q = memref.alloc() : memref<2x!qc.qubit>
    %q0 = memref.load %q[%c0] : memref<2x!qc.qubit>
    qc.x %q0 : !qc.qubit
    memref.dealloc %q : memref<2x!qc.qubit>""",
    ],
    ids=["scalar", "register"],
)
def test_target_aware_qiskit_export_rejects_dynamic_qubits(allocation: str) -> None:
    """Require target-aware export inputs to use static qubits."""
    target = CompilerTarget(2)
    program = QCProgram.from_mlir_str(
        f"""module {{
  func.func @main() attributes {{mqt.entry_point}} {{
    {allocation}
    return
  }}
}}
"""
    )

    with pytest.raises(RuntimeError, match="target-aware Qiskit export requires statically mapped qubits"):
        program.to_qiskit(target=target)


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
