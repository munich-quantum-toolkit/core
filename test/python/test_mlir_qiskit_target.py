# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Contracts for converting Qiskit compiler targets through the native bridge."""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Measure, Parameter, ParameterExpression, Reset
from qiskit.circuit.controlflow import IfElseOp
from qiskit.circuit.library import CXGate, GlobalPhaseGate, RZGate, UGate, XGate
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Operator
from qiskit.transpiler import Target
from qiskit_support import supports_qiskit_translation

from mqt.core.mlir import CompilerTarget, PayloadFormat, PayloadSpecification, QCProgram, TargetEnvironment
from mqt.core.plugins.qiskit import compiler_target_from_qiskit

if not supports_qiskit_translation():
    pytest.skip("Qiskit version has no compiler translation adapter", allow_module_level=True)


def test_directed_sites_and_snapshot() -> None:
    """Undirected routing edges must not broaden native gate applicability."""
    source = Target(num_qubits=3)
    source.add_instruction(RZGate(Parameter("angle")))
    source.add_instruction(XGate(), {(0,): None})
    source.add_instruction(CXGate(), {(1, 0): None, (1, 2): None})
    converted = compiler_target_from_qiskit(source, name="directed")
    source.add_instruction(UGate(Parameter("a"), Parameter("b"), Parameter("c")))

    assert converted.name == "directed"
    assert converted.couplings == [(0, 1), (1, 2)]
    operations = {operation.name: operation for operation in converted.operations}
    assert set(operations) == {"rz", "x", "cx", "gphase"}
    assert not operations["rz"].site_tuples
    assert [placement.sites for placement in operations["x"].site_tuples] == [[0]]
    assert [placement.sites for placement in operations["cx"].site_tuples] == [[1, 0], [1, 2]]


def test_backend_and_global_operations() -> None:
    """A complete graph is all-to-all, without dropping ordered gate sites."""
    backend = GenericBackendV2(2, basis_gates=["sx", "rz", "cx"], control_flow=True)
    converted = compiler_target_from_qiskit(backend)
    assert converted.num_sites == 2
    assert converted.name == backend.name
    assert converted.connectivity_kind == CompilerTarget.ConnectivityKind.ALL_TO_ALL
    assert {operation.name for operation in converted.operations} == {"sx", "rz", "cx", "measure", "reset", "gphase"}

    source = Target(num_qubits=3)
    source.add_instruction(CXGate())
    source.add_instruction(XGate(), {})
    source.add_instruction(RZGate(0.5), {})
    source.add_instruction(IfElseOp, name="if_else")
    converted = compiler_target_from_qiskit(source)
    assert converted.connectivity_kind == CompilerTarget.ConnectivityKind.ALL_TO_ALL
    assert {operation.name for operation in converted.operations} == {"cx", "gphase"}


@pytest.mark.parametrize("angle", [0.5, Parameter("theta") / 2])
def test_parameter_constraints(angle: float | ParameterExpression) -> None:
    """Fixed and derived parameters cannot become arbitrary rotations."""
    source = Target(num_qubits=1)
    source.add_instruction(RZGate(angle))
    with pytest.raises(ValueError, match="parameter constraints for 'rz'"):
        compiler_target_from_qiskit(source, operation_names=["rz"])
    with pytest.warns(UserWarning, match="parameter constraints"), pytest.raises(ValueError, match="no representable"):
        compiler_target_from_qiskit(source)


def test_correlated_parameters_and_custom_gates() -> None:
    """Equal parameter slots and custom gates must not be silently broadened."""
    theta = Parameter("theta")
    source = Target(num_qubits=1)
    source.add_instruction(UGate(theta, theta, theta))
    source.add_instruction(Gate("x", 1, []))
    with pytest.raises(ValueError, match="parameter constraints for 'u'"):
        compiler_target_from_qiskit(source, operation_names=["u"])
    with pytest.raises(ValueError, match=r"custom.*x"):
        compiler_target_from_qiskit(source, operation_names=["x"])
    with pytest.raises(ValueError, match=r"does not expose.*rz"):
        compiler_target_from_qiskit(source, operation_names=["rz"])


def test_restricted_operation_set() -> None:
    """An explicit subset can omit an unrepresentable operation."""
    source = Target(num_qubits=1)
    source.add_instruction(XGate())
    source.add_instruction(RZGate(0.5))
    converted = compiler_target_from_qiskit(source, operation_names=["x"])
    assert {operation.name for operation in converted.operations} == {"x", "gphase"}

    with pytest.warns(UserWarning, match="parameter constraints"):
        converted = compiler_target_from_qiskit(source)
    assert {operation.name for operation in converted.operations} == {"x", "gphase"}


def test_angle_bounds_and_open_controls() -> None:
    """Instruction metadata also constrains the accepted gate semantics."""
    source = Target(num_qubits=2)
    source.add_instruction(CXGate(ctrl_state=0), name="cx")
    with pytest.raises(ValueError, match="open controls"):
        compiler_target_from_qiskit(source, operation_names=["cx"])
    source = Target(num_qubits=1)
    source.add_instruction(RZGate(Parameter("theta")), angle_bounds=[(-1.0, 1.0)])
    with pytest.raises(ValueError, match="parameter constraints"):
        compiler_target_from_qiskit(source, operation_names=["rz"])


def test_standard_aliases_and_custom_operations() -> None:
    """Use gate identity, not target keys or custom gates with standard names."""
    source = Target(num_qubits=1)
    source.add_instruction(XGate(), name="native_x")
    source.add_instruction(Gate("x", 1, []))
    source.add_instruction(GlobalPhaseGate(0.5))
    with pytest.warns(UserWarning, match="custom or unsupported operation for 'x'"):
        converted = compiler_target_from_qiskit(source)
    assert {operation.name: operation.canonical_name for operation in converted.operations} == {
        "native_x": "x",
        "gphase": "gphase",
    }


@pytest.mark.parametrize("qco", [False, True])
def test_export_restores_aliases_on_ordered_sites(*, qco: bool) -> None:
    """Export backend names without changing gate semantics or applicability."""
    theta = Parameter("theta")
    source = Target(num_qubits=2)
    source.add_instruction(XGate(), {(0,): None})
    source.add_instruction(XGate(), {(1,): None}, name="native_x")
    source.add_instruction(RZGate(theta), name="native_rz")
    source.add_instruction(CXGate(), {(0, 1): None}, name="native_cx")
    target = compiler_target_from_qiskit(source)
    circuit = QuantumCircuit(2)
    circuit.x(0)
    circuit.x(1)
    circuit.rz(theta, 1)
    circuit.cx(0, 1)
    program = QCProgram.from_mlir_str("""module {
  func.func @main(%theta: f64 {mqt.input_name = "theta"}) attributes {mqt.entry_point} {
    %a = qc.static 0 : !qc.qubit
    %b = qc.static 1 : !qc.qubit
    qc.x %a : !qc.qubit
    qc.x %b : !qc.qubit
    qc.rz(%theta) %b : !qc.qubit
    qc.ctrl(%a) targets (%t = %b) {
      qc.x %t : !qc.qubit
      qc.yield
    } : {!qc.qubit}, {!qc.qubit}
    return
  }
}
""")
    exported = (program.to_qco() if qco else program).to_qiskit(target=target)
    assert exported.count_ops() == {"x": 1, "native_x": 1, "native_rz": 1, "native_cx": 1}
    assert all(
        source.instruction_supported(
            operation_name=item.operation.name,
            qargs=tuple(exported.find_bit(qubit).index for qubit in item.qubits),
            parameters=item.operation.params,
        )
        for item in exported.data
    )
    assert Operator(exported.assign_parameters({"theta": 0.3})).equiv(Operator(circuit.assign_parameters({theta: 0.3})))
    restored = QCProgram.from_qiskit(exported).to_qiskit()
    assert Operator(restored.assign_parameters({"theta": 0.3})).equiv(Operator(circuit.assign_parameters({theta: 0.3})))

    reverse = QCProgram.from_openqasm_str('OPENQASM 3.0; include "stdgates.inc"; cx $1, $0;')
    assert reverse.to_qiskit(target=target).count_ops() == {"cx": 1}


def test_export_aliases_inside_control_flow() -> None:
    """Keep physical sites and aliased measurement/reset instructions in blocks."""
    source = Target(num_qubits=2)
    source.add_instruction(CXGate())
    source.add_instruction(XGate(), {(1,): None}, name="native_x")
    source.add_instruction(Measure(), name="readout")
    source.add_instruction(Reset(), name="clear")
    program = QCProgram.from_openqasm_str("""OPENQASM 3.0;
include "stdgates.inc";
bit c = measure $1;
if (c) { reset $1; x $1; }
""")
    exported = program.to_qiskit(target=compiler_target_from_qiskit(source))
    assert exported.count_ops() == {"readout": 1, "if_else": 1}
    block = exported.data[-1].operation.blocks[0]
    assert block.count_ops() == {"clear": 1, "native_x": 1}
    assert block.data[-1].operation.definition.count_ops() == {"x": 1}


@pytest.mark.parametrize(
    "basis", [["sx", "x", "rz", "cx"], ["sx", "x", "rz", "ecr"], ["rx", "rz", "cz"], ["u", "cx"], ["r", "rxx"]]
)
def test_backend_bases_compile(basis: list[str]) -> None:
    """Keep usable backend bases despite extra provider-specific operations."""
    backend = GenericBackendV2(2, basis_gates=basis, seed=1)
    backend.target.add_instruction(Gate("provider_gate", 2, []))
    with pytest.warns(UserWarning, match="provider_gate"):
        target = compiler_target_from_qiskit(backend)
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.ry(0.3, 1)
    program = QCProgram.from_qiskit(circuit).to_qco()
    program.compile_for_target(TargetEnvironment(target, PayloadSpecification(PayloadFormat("openqasm", "3.0"))))
    restored = program.to_qiskit(target=target)
    assert restored.data
    assert all(
        backend.target.instruction_supported(
            operation_name=item.operation.name,
            qargs=tuple(restored.find_bit(qubit).index for qubit in item.qubits),
            parameters=item.operation.params,
        )
        for item in restored.data
    )


def test_unknown_width_and_disconnected_topology() -> None:
    """Missing connectivity must not turn into an all-to-all device."""
    with pytest.raises(ValueError, match="positive qubit count"):
        compiler_target_from_qiskit(Target(num_qubits=None))
    source = Target(num_qubits=2)
    source.add_instruction(XGate())
    source.add_instruction(CXGate(), {})
    with pytest.raises(ValueError, match="connected"):
        compiler_target_from_qiskit(source)

    source = Target(num_qubits=3)
    source.add_instruction(CXGate(), {(0, 1): None})
    source.add_instruction(Gate("custom_bridge", 2, []), {(1, 2): None})
    with pytest.warns(UserWarning, match="custom_bridge"), pytest.raises(ValueError, match="connected"):
        compiler_target_from_qiskit(source)
