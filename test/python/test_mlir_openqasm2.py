# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""OpenQASM 2 export checked with Qiskit's independent parser and matrices."""

from __future__ import annotations

from typing import assert_type

import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, qasm2
from qiskit.circuit import Gate, Parameter
from qiskit.quantum_info import Operator

from mqt.core.mlir import CompilerTarget, OpenQASMProgram, OutputFormat, QCProgram, compile_program, sample


def test_openqasm2_exact_gate_matrix_and_global_phase() -> None:
    """U parameters and asymmetric operands preserve matrix entries and phase."""
    circuit = QuantumCircuit(3)
    circuit.u(0.31, -0.62, 0.47, 2)
    circuit.sx(0)
    circuit.cx(2, 0)
    circuit.cp(0.19, 0, 1)
    circuit.ry(-0.21, 1)
    circuit.swap(1, 2)
    angle = Parameter("angle")
    definition = QuantumCircuit(1)
    definition.ry(angle + 0.17, 0)
    circuit.append(definition.to_gate(), [1])
    circuit.assign_parameters({angle: 0.23}, inplace=True)
    source = QCProgram.from_qiskit(circuit).to_openqasm2().source
    restored = qasm2.loads(source, strict=True)
    np.testing.assert_allclose(Operator(restored).data, Operator(circuit).data, atol=1e-12)

    circuit.global_phase = 0.37
    with_phase = QCProgram.from_qiskit(circuit).to_openqasm2().source
    actual = Operator(qasm2.loads(with_phase, strict=True)).data
    expected = Operator(circuit).data
    assert not np.allclose(actual, expected)
    np.testing.assert_allclose(actual * np.exp(0.37j), expected, atol=1e-12)


def test_openqasm2_controlled_phase_is_observable() -> None:
    """Removing a phase inside a control would change interference."""
    phase = QuantumCircuit(1, global_phase=0.6)
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.append(phase.to_gate().control(), [0, 1])
    circuit.h(0)
    source = QCProgram.from_qiskit(circuit).to_openqasm2().source
    np.testing.assert_allclose(Operator(qasm2.loads(source, strict=True)).data, Operator(circuit).data, atol=1e-12)


def test_openqasm2_measurement_order_reset_and_native_sites() -> None:
    """Register order, repeated stores, and idle device sites survive export."""
    circuit = QuantumCircuit(QuantumRegister(8, "q"), ClassicalRegister(2, "a"), ClassicalRegister(1, "b"))
    circuit.x(7)
    circuit.measure(7, 1)
    circuit.reset(7)
    circuit.measure(7, 2)
    circuit.measure(7, 0)
    source = QCProgram.from_qiskit(circuit, native=True).to_openqasm2().source
    restored = qasm2.loads(source, strict=True)
    assert restored.num_qubits == 8
    assert [(reg.name, reg.size) for reg in restored.cregs] == [("a", 2), ("b", 1)]
    assert sample(source, shots=32, seed=42) == {"010": 32}


def test_openqasm2_native_opaque_calls_preserve_parameters() -> None:
    """Receiver-defined gates keep their names, parameters, and operands."""
    circuit = QuantumCircuit(4)
    circuit.append(Gate("hop", 2, [0.125]), [3, 1])
    source = QCProgram.from_qiskit(circuit, native=True).to_openqasm2().source
    restored = qasm2.loads(source, strict=True)
    instruction = restored.data[0]
    assert instruction.operation.name == "hop"
    assert instruction.operation.params == [0.125]
    assert [restored.find_bit(qubit).index for qubit in instruction.qubits] == [3, 1]


def test_openqasm2_compilation_unrolls_static_loop() -> None:
    """Explicit compilation lowers finite loops while direct export rejects them."""
    source = 'OPENQASM 3; include "stdgates.inc"; qubit[3] q; for int i in [0:2] { x q[i]; } bit[3] c = measure q;'
    with pytest.raises(RuntimeError, match="lowering"):
        QCProgram.from_openqasm_str(source).to_openqasm2()
    compiled = compile_program(source, output=OutputFormat.OPENQASM2)
    assert_type(compiled, OpenQASMProgram)
    restored = qasm2.loads(compiled.source, strict=True)
    assert restored.num_clbits == 3
    assert sample(compiled, shots=8, seed=1) == {"111": 8}


def test_openqasm2_requires_bound_inputs() -> None:
    """QASM2 has no runtime parameter input declarations."""
    circuit = QuantumCircuit(1)
    circuit.rx(Parameter("theta"), 0)
    with pytest.raises(RuntimeError, match="bound inputs"):
        QCProgram.from_qiskit(circuit).to_openqasm2()


def test_openqasm2_explicit_target_declares_native_calls() -> None:
    """Target compilation emits complete receiver-defined gate signatures."""
    target = CompilerTarget(
        2,
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations([
            CompilerTarget.OperationCapability("u", 1, 3),
            CompilerTarget.OperationCapability("cx", 2, 0),
        ]),
    )
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    compiled = compile_program(circuit, target=target, output=OutputFormat.OPENQASM2)
    restored = qasm2.loads(compiled.source, strict=True)
    assert restored.num_qubits == 2
    assert set(restored.count_ops()) <= {"u", "cx"}
    assert "opaque u(" in compiled.source
    assert 'include "qelib1.inc";' in compiled.source
    assert "opaque cx " not in compiled.source


def test_openqasm2_rejects_conflicting_qelib_signature() -> None:
    """An opaque name cannot override a different included-library signature."""
    circuit = QuantumCircuit(1)
    circuit.append(Gate("cu1", 1, [0.1]), [0])
    with pytest.raises(RuntimeError, match="signature"):
        QCProgram.from_qiskit(circuit, native=True).to_openqasm2()


def test_openqasm2_rejects_unlowered_controlled_rotation() -> None:
    """Do not emit a QASM3 library alias that qelib1 does not define."""
    circuit = QuantumCircuit(2)
    circuit.crx(0.2, 0, 1)
    with pytest.raises(RuntimeError, match="modifiers"):
        QCProgram.from_qiskit(circuit).to_openqasm2()


def test_openqasm2_rejects_unrepresentable_device_index() -> None:
    """A valid QC index must fit an OpenQASM register without overflow."""
    program = QCProgram.from_mlir_str("""
        module {
          func.func @main() attributes {mqt.entry_point} {
            %q = qc.static 9223372036854775807 : !qc.qubit
            qc.x %q : !qc.qubit
            return
          }
        }
    """)
    with pytest.raises(RuntimeError, match="width limit"):
        program.to_openqasm2()


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            ("x q[0]; measure q[0] -> c[0]; if(c==1) x q[1]; if(c==0) reset q[1]; measure q[1] -> c[1];"),
            "11",
        ),
        ("x q; if(c==0) measure q -> c;", "11"),
        ("x q; measure q -> c; if(c==3) reset q; measure q -> c;", "00"),
    ],
)
def test_openqasm2_register_conditions(body: str, expected: str) -> None:
    """Register guards preserve feedback and evaluate register measurement once."""
    source = 'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; creg c[2]; ' + body
    exported = QCProgram.from_openqasm_str(source).to_openqasm2().source
    qasm2.loads(exported, strict=True)
    assert sample(exported, shots=8, seed=7) == {expected: 8}


def test_openqasm2_gate_expression_matrix() -> None:
    """Inlining gate definitions preserves the OpenQASM 2 expression semantics."""
    source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        gate expression(a) q {
            U(sin(a)+cos(a),tan(a)-exp(a),ln(a)+sqrt(a)+a^2) q;
            U(-a^2+1/2,2^3^2,2^-2) q;
            barrier q;
        }
        qreg q[2];
        expression(pi/4) q;
        CX q[1],q[0];
    """
    exported = QCProgram.from_openqasm_str(source).to_openqasm2().source
    np.testing.assert_allclose(
        Operator(qasm2.loads(exported, strict=True)).data,
        Operator(qasm2.loads(source, strict=True)).data,
        atol=1e-12,
    )
