# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Contracts for converting Qiskit compiler targets through the native bridge."""

from __future__ import annotations

from math import pi
from typing import cast

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Measure, Parameter, Reset
from qiskit.circuit.controlflow import IfElseOp
from qiskit.circuit.library import (
    CCXGate,
    CPhaseGate,
    CUGate,
    CXGate,
    CYGate,
    GlobalPhaseGate,
    PhaseGate,
    RGate,
    RXGate,
    RZGate,
    SXGate,
    U1Gate,
    U3Gate,
    UGate,
    XGate,
)
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Operator
from qiskit.transpiler import Target
from qiskit_support import supports_qiskit_translation

from mqt.core.mlir import (
    CompilerTarget,
    PayloadFormat,
    PayloadSpecification,
    QCProgram,
    TargetEnvironment,
)

if not supports_qiskit_translation():
    pytest.skip("Qiskit version has no compiler translation adapter", allow_module_level=True)


def test_directed_sites_and_snapshot() -> None:
    """Undirected routing edges must not broaden native gate applicability."""
    source = Target(num_qubits=3)
    source.add_instruction(RZGate(Parameter("angle")))
    source.add_instruction(XGate(), {(0,): None})
    source.add_instruction(CXGate(), {(1, 0): None, (1, 2): None})
    converted = CompilerTarget.from_qiskit(source, name="directed")
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
    converted = CompilerTarget.from_qiskit(backend)
    assert converted.num_sites == 2
    assert converted.name == backend.name
    assert CompilerTarget.from_qiskit(backend, name="renamed").name == "renamed"
    assert converted.connectivity_kind == CompilerTarget.ConnectivityKind.ALL_TO_ALL
    assert {operation.name for operation in converted.operations} == {"sx", "rz", "cx", "measure", "reset", "gphase"}

    source = Target(num_qubits=3)
    source.add_instruction(CXGate())
    source.add_instruction(XGate(), {})
    source.add_instruction(RZGate(0.5), {})
    source.add_instruction(IfElseOp, name="if_else")
    converted = CompilerTarget.from_qiskit(source)
    assert converted.connectivity_kind == CompilerTarget.ConnectivityKind.ALL_TO_ALL
    assert {operation.name for operation in converted.operations} == {"cx", "gphase"}
    for name in ("x", "rz", "if_else"):
        with pytest.raises(ValueError, match="no native gate applicability"):
            CompilerTarget.from_qiskit(source, operation_names=["cx", name])


def test_fixed_parameter_constraints() -> None:
    """Named discrete angles remain separate capabilities of one gate."""
    source = Target(num_qubits=1)
    source.add_instruction(RXGate(pi / 2), name="rx_90")
    source.add_instruction(RXGate(pi), name="rx_180")
    source.add_instruction(RZGate(Parameter("angle")))
    target = CompilerTarget.from_qiskit(source)
    operations = {operation.name: operation for operation in target.operations}
    assert operations["rx_90"].canonical_name == operations["rx_180"].canonical_name == "rx"
    assert operations["rx_90"].fixed_parameters == [pi / 2]
    assert operations["rx_180"].fixed_parameters == [pi]
    assert operations["rz"].fixed_parameters == []
    for angle in (pi / 2, pi):
        assert target.supports_operation("rx", 1, 1, parameters=[angle])
    for angle in (None, -pi / 2, 0.3):
        assert not target.supports_operation("rx", 1, 1, parameters=[angle])


def test_partially_fixed_parameter_slots() -> None:
    """Bound expressions become fixed values; remaining symbols are wildcards."""
    angle = Parameter("angle")
    source = Target(num_qubits=1)
    source.add_instruction(UGate(angle.bind({angle: pi / 2}), angle / 2, 0.0), name="native_u")
    target = CompilerTarget.from_qiskit(source)
    operation = next(operation for operation in target.operations if operation.name == "native_u")
    assert operation.canonical_name == "u"
    assert operation.fixed_parameters == [pi / 2, None, 0.0]
    assert target.supports_operation("u", 1, 3, parameters=[pi / 2, None, 0.0])
    assert not target.supports_operation("u", 1, 3, parameters=[None, None, 0.0])


@pytest.mark.filterwarnings("error:Cannot represent.*:UserWarning")
def test_target_warning_as_error() -> None:
    """A native conversion warning respects the caller's warning filters."""
    source = Target(num_qubits=1)
    source.add_instruction(RZGate(Parameter("angle")), angle_bounds=[(-1.0, 1.0)])
    with pytest.raises(UserWarning, match="parameter constraints"):
        CompilerTarget.from_qiskit(source)


def test_target_selection() -> None:
    """Selection accepts iterables and reports invalid sources."""
    source = Target(num_qubits=1)
    source.add_instruction(XGate())
    converted = CompilerTarget.from_qiskit(source, operation_names=iter(["x", "x"]), name="native")
    source.add_instruction(RZGate(Parameter("angle")))
    assert converted.name == "native"
    assert {operation.name for operation in converted.operations} == {"x", "gphase"}
    with pytest.raises(ValueError, match="no representable"):
        CompilerTarget.from_qiskit(source, operation_names=[])
    with pytest.raises(TypeError, match="Expected a Qiskit Target or BackendV2"):
        CompilerTarget.from_qiskit(cast("Target", object()))


@pytest.mark.parametrize("bounds", [None, [None] * 3, [(-float("inf"), float("inf"))] * 3])
def test_unrestricted_parameter_slots(bounds: list[tuple[float, float] | None] | None) -> None:
    """Target symbols are wildcards, not bindings shared between slots."""
    theta = Parameter("theta")
    source = Target(num_qubits=1)
    source.add_instruction(UGate(theta / 2, theta, theta), angle_bounds=bounds)
    assert source.instruction_supported("u", (0,), parameters=[0.1, 0.2, 0.3])
    converted = CompilerTarget.from_qiskit(source, operation_names=["u"])
    assert converted.supports_operation("u", 1, 3)
    with pytest.raises(ValueError, match=r"does not expose.*rz"):
        CompilerTarget.from_qiskit(source, operation_names=["rz"])


def test_restricted_operation_set() -> None:
    """An explicit subset can omit an unrepresentable operation."""
    source = Target(num_qubits=1)
    source.add_instruction(XGate())
    source.add_instruction(RZGate(Parameter("angle")), angle_bounds=[(-1.0, 1.0)])
    converted = CompilerTarget.from_qiskit(source, operation_names=["x"])
    assert {operation.name for operation in converted.operations} == {"x", "gphase"}

    with pytest.warns(UserWarning, match="parameter constraints") as warnings:
        converted = CompilerTarget.from_qiskit(source)
    assert {operation.name for operation in converted.operations} == {"x", "gphase"}
    assert warnings[0].filename == __file__


def test_angle_bounds_and_open_controls() -> None:
    """Instruction metadata also constrains the accepted gate semantics."""
    source = Target(num_qubits=2)
    source.add_instruction(CXGate(ctrl_state=0), name="cx")
    with pytest.raises(ValueError, match="open controls"):
        CompilerTarget.from_qiskit(source, operation_names=["cx"])
    source = Target(num_qubits=1)
    source.add_instruction(RZGate(Parameter("theta")), angle_bounds=[(-1.0, 1.0)])
    with pytest.raises(ValueError, match="parameter constraints"):
        CompilerTarget.from_qiskit(source, operation_names=["rz"])


def test_custom_names_and_operations() -> None:
    """Names do not override gate identity; custom gates remain unsupported."""
    source = Target(num_qubits=1)
    source.add_instruction(XGate(), name="native_x")
    source.add_instruction(Gate("x", 1, []))
    source.add_instruction(RZGate(Parameter("theta")))
    source.add_instruction(GlobalPhaseGate(0.5))
    with pytest.warns(UserWarning, match="custom"):
        converted = CompilerTarget.from_qiskit(source)
    assert {operation.name for operation in converted.operations} == {"native_x", "rz", "gphase"}
    assert converted.supports_operation("x", 1, 0)
    source.add_instruction(Measure(), name="native_measure")
    source.add_instruction(Reset(), name="native_reset")
    for name in ("x", "native_measure", "native_reset"):
        with pytest.raises(ValueError, match="custom"):
            CompilerTarget.from_qiskit(source, operation_names=[name])
    with pytest.raises(ValueError, match="no native gate applicability"):
        CompilerTarget.from_qiskit(source, operation_names=["global_phase"])


@pytest.mark.parametrize("symbolic", [False, True])
def test_compile_named_fixed_rotations(*, symbolic: bool) -> None:
    """Compilation exports executable target names with exact gate phases."""
    source = Target(num_qubits=2)
    source.add_instruction(RXGate(pi / 2), name="quarter_turn")
    source.add_instruction(RXGate(pi), name="half_turn")
    source.add_instruction(RZGate(Parameter("angle")))
    source.add_instruction(CXGate())
    target = CompilerTarget.from_qiskit(source)
    circuit = QuantumCircuit(2)
    circuit.rx(pi, 0)
    circuit.ry(Parameter("theta") if symbolic else 0.3, 1)
    circuit.cx(0, 1)
    program = QCProgram.from_qiskit(circuit).to_qco()
    program.compile_for_target(TargetEnvironment(target, PayloadSpecification(PayloadFormat("openqasm", "3.0"))))
    exported = program.to_qiskit(target=target)
    assert any(item.operation.name == "quarter_turn" for item in exported.data)
    assert all(
        source.instruction_supported(
            item.operation.name,
            tuple(exported.find_bit(qubit).index for qubit in item.qubits),
            parameters=item.operation.params,
        )
        for item in exported.data
    )
    if symbolic:
        exported = exported.assign_parameters({"theta": 0.7})
        circuit = circuit.assign_parameters({"theta": 0.7})
    assert np.allclose(Operator(exported).data, Operator(circuit).data)
    restored = QCProgram.from_qiskit(exported).to_qiskit()
    assert np.allclose(Operator(restored).data, Operator(circuit).data)


def test_export_fixed_rotations_by_parameters_and_sites() -> None:
    """Select aliases by values and placement, including colliding gate names."""
    source = Target(num_qubits=2)
    source.add_instruction(RXGate(pi / 2), {(0,): None}, name="a_quarter")
    source.add_instruction(RXGate(pi), {(0,): None}, name="b_half")
    source.add_instruction(RXGate(Parameter("angle")), {(0,): None}, name="z_variable")
    source.add_instruction(RXGate(pi / 2), {(1,): None}, name="ry")
    source.add_instruction(CXGate())
    target = CompilerTarget.from_qiskit(source)
    circuit = QuantumCircuit(2)
    circuit.rx(pi / 2, 0)
    circuit.rx(pi, 0)
    circuit.rx(Parameter("theta"), 0)
    circuit.rx(pi / 2, 1)
    program = QCProgram.from_mlir_str("""module {
  func.func @main(%theta: f64 {mqt.input_name = "theta"}) attributes {mqt.entry_point} {
    %quarter = arith.constant 1.5707963267948966 : f64
    %half = arith.constant 3.141592653589793 : f64
    %a = qc.static 0 : !qc.qubit
    %b = qc.static 1 : !qc.qubit
    qc.rx(%quarter) %a : !qc.qubit
    qc.rx(%half) %a : !qc.qubit
    qc.rx(%theta) %a : !qc.qubit
    qc.rx(%quarter) %b : !qc.qubit
    return
  }
}
""")
    exported = program.to_qiskit(target=target)
    assert [item.operation.name for item in exported.data] == ["a_quarter", "b_half", "z_variable", "ry"]
    assert all(item.operation.base_class is RXGate for item in exported.data)
    restored = QCProgram.from_qiskit(exported).to_qiskit()
    assert np.allclose(
        Operator(restored.assign_parameters({"theta": 0.3})).data,
        Operator(circuit.assign_parameters({"theta": 0.3})).data,
    )


def test_named_sx_and_x_keep_phase() -> None:
    """SX and X keep their exact matrices instead of becoming RX aliases."""
    source = Target(num_qubits=1)
    source.add_instruction(SXGate(), name="native_sx")
    source.add_instruction(XGate(), name="native_x")
    target = CompilerTarget.from_qiskit(source)
    circuit = QuantumCircuit(1)
    circuit.sx(0)
    circuit.x(0)
    program = QCProgram.from_openqasm_str('OPENQASM 3.0; include "stdgates.inc"; sx $0; x $0;')
    exported = program.to_qiskit(target=target)
    assert [item.operation.name for item in exported.data] == ["native_sx", "native_x"]
    assert np.allclose(Operator(exported).data, Operator(circuit).data)


@pytest.mark.parametrize("qco", [False, True])
@pytest.mark.parametrize("symbolic", [False, True])
def test_export_standard_aliases_on_ordered_sites(*, qco: bool, symbolic: bool) -> None:
    """Keep legacy spellings as native Qiskit gates on their applicable sites."""
    theta = Parameter("theta")
    source = Target(num_qubits=2)
    source.add_instruction(PhaseGate(theta), {(0,): None})
    source.add_instruction(U1Gate(theta), {(1,): None})
    source.add_instruction(CXGate(), {(0, 1): None})
    target = CompilerTarget.from_qiskit(source)
    circuit = QuantumCircuit(2)
    circuit.p(theta if symbolic else 0.3, 0)
    circuit.p(theta if symbolic else 0.3, 1)
    argument = '%theta: f64 {mqt.input_name = "theta"}' if symbolic else ""
    definition = "" if symbolic else "%theta = arith.constant 0.3 : f64"
    program = QCProgram.from_mlir_str(f"""module {{
  func.func @main({argument}) attributes {{mqt.entry_point}} {{
    {definition}
    %a = qc.static 0 : !qc.qubit
    %b = qc.static 1 : !qc.qubit
    qc.p(%theta) %a : !qc.qubit
    qc.p(%theta) %b : !qc.qubit
    return
  }}
}}
""")
    exported = (program.to_qco() if qco else program).to_qiskit(target=target)
    assert exported.count_ops() == {"p": 1, "u1": 1}
    assert exported.data[1].operation.base_class is U1Gate
    assert all(
        source.instruction_supported(
            operation_name=item.operation.name,
            qargs=tuple(exported.find_bit(qubit).index for qubit in item.qubits),
            parameters=item.operation.params,
        )
        for item in exported.data
    )
    if symbolic:
        exported = exported.assign_parameters({"theta": 0.3})
        circuit = circuit.assign_parameters({theta: 0.3})
    assert Operator(exported).equiv(Operator(circuit))
    restored = QCProgram.from_qiskit(exported).to_qiskit()
    assert Operator(restored).equiv(Operator(circuit))

    reverse = QCProgram.from_openqasm_str('OPENQASM 3.0; include "stdgates.inc"; cx $1, $0;')
    assert reverse.to_qiskit(target=target).count_ops() == {"cx": 1}


def test_export_aliases_inside_control_flow() -> None:
    """Select legacy gates using physical sites inside nested blocks."""
    source = Target(num_qubits=2)
    source.add_instruction(CXGate())
    source.add_instruction(U1Gate(Parameter("theta")), {(1,): None})
    source.add_instruction(Measure())
    source.add_instruction(Reset())
    program = QCProgram.from_openqasm_str("""OPENQASM 3.0;
include "stdgates.inc";
bit c = measure $1;
if (c) { reset $1; p(0.3) $1; }
""")
    exported = program.to_qiskit(target=CompilerTarget.from_qiskit(source))
    assert exported.count_ops() == {"measure": 1, "if_else": 1}
    block = exported.data[-1].operation.blocks[0]
    assert block.count_ops() == {"reset": 1, "u1": 1}
    assert block.data[-1].operation.base_class is U1Gate


@pytest.mark.parametrize(
    "basis",
    [
        ["sx", "x", "rz", "cx"],
        ["sx", "x", "rz", "ecr"],
        ["rx", "rz", "cz"],
        ["u", "cx"],
        ["r", "rxx"],
        ["u1", "u2", "u3", "cx"],
    ],
)
def test_backend_bases_compile(basis: list[str]) -> None:
    """Keep usable backend bases despite extra provider-specific operations."""
    backend = GenericBackendV2(2, basis_gates=basis, seed=1)
    backend.target.add_instruction(Gate("provider_gate", 2, []))
    with pytest.warns(UserWarning, match="provider_gate"):
        target = CompilerTarget.from_qiskit(backend)
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.ry(0.3, 1)
    program = QCProgram.from_qiskit(circuit).to_qco()
    program.compile_for_target(TargetEnvironment(target, PayloadSpecification(PayloadFormat("openqasm", "3.0"))))
    restored = program.to_qiskit(target=target)
    assert restored.data
    if "u3" in basis:
        assert any(item.operation.base_class is U3Gate for item in restored.data)
    assert all(
        backend.target.instruction_supported(
            operation_name=item.operation.name,
            qargs=tuple(restored.find_bit(qubit).index for qubit in item.qubits),
            parameters=item.operation.params,
        )
        for item in restored.data
    )


@pytest.mark.parametrize("gate", [CPhaseGate(Parameter("a")), CCXGate(), CUGate(*[Parameter("a")] * 4)])
def test_unsupported_controlled_gate_does_not_add_routing_edges(gate: Gate) -> None:
    """Circuit import support must not create unusable native routing edges."""
    source = Target(num_qubits=3)
    theta = Parameter("theta")
    source.add_instruction(UGate(theta, theta, theta))
    source.add_instruction(CXGate(), {(0, 1): None, (1, 2): None})
    source.add_instruction(gate, {(0, 2) if gate.num_qubits == 2 else (0, 1, 2): None})
    with pytest.warns(UserWarning, match=f"unsupported operation for '{gate.name}'"):
        target = CompilerTarget.from_qiskit(source)
    assert target.couplings == [(0, 1), (1, 2)]
    assert gate.name not in {operation.name for operation in target.operations}
    with pytest.raises(ValueError, match=f"unsupported operation for '{gate.name}'"):
        CompilerTarget.from_qiskit(source, operation_names=[gate.name])
    program = QCProgram.from_openqasm_str('OPENQASM 3.0; include "stdgates.inc"; qubit[3] q; cz q[0], q[1];').to_qco()
    program.compile_for_target(TargetEnvironment(target, PayloadSpecification(PayloadFormat("openqasm", "3.0"))))
    assert program.to_qiskit(target=target).count_ops()["cx"] == 1


def test_unknown_width_and_disconnected_topology() -> None:
    """Missing connectivity must not turn into an all-to-all device."""
    with pytest.raises(ValueError, match="positive qubit count"):
        CompilerTarget.from_qiskit(Target(num_qubits=None))
    source = Target(num_qubits=2)
    source.add_instruction(XGate())
    source.add_instruction(CXGate(), {})
    with pytest.raises(ValueError, match="connected"):
        CompilerTarget.from_qiskit(source)

    source = Target(num_qubits=3)
    source.add_instruction(CXGate(), {(0, 1): None})
    source.add_instruction(Gate("custom_bridge", 2, []), {(1, 2): None})
    with pytest.warns(UserWarning, match="custom_bridge"), pytest.raises(ValueError, match="connected"):
        CompilerTarget.from_qiskit(source)


@pytest.mark.parametrize("name", ["gpi", "gpi2"])
@pytest.mark.parametrize("symbolic", [False, True])
def test_native_r_capabilities(name: str, *, symbolic: bool) -> None:
    """Native names project fixed-angle R gates without changing circuit phase."""
    phi = Parameter("phi") if symbolic else 0.13
    definition = QuantumCircuit(1)
    definition.r(pi if name == "gpi" else pi / 2, phi, 0)
    if name == "gpi":
        definition.global_phase = pi / 2
    gate = Gate(name, 1, [phi])
    gate.definition = definition
    source = Target(num_qubits=2)
    source.add_instruction(gate)
    unrestricted = UGate(*map(Parameter, ("theta", "lambda", "beta")))
    source.add_instruction(unrestricted)
    source.add_instruction(CXGate())
    converted = CompilerTarget.from_qiskit(source)
    operation = next(operation for operation in converted.operations if operation.name == name)
    assert operation.canonical_name == "r"
    assert operation.num_parameters == 2
    assert operation.fixed_parameters == [pi if name == "gpi" else pi / 2, None if symbolic else phi]
    circuit = QuantumCircuit(2, global_phase=0.29)
    circuit.append(gate, [0])
    imported = QCProgram.from_qiskit(circuit).to_qco()
    imported.compile_for_target(TargetEnvironment(converted, PayloadSpecification(PayloadFormat("openqasm", "3.0"))))
    assert "qco.r(" in imported.ir
    exported = imported.to_qiskit(target=converted)
    for _ in range(2):
        assert exported.data[0].operation.name == name
        assert exported.data[0].operation.params == [phi]
        assert source.instruction_supported(name, (0,), parameters=exported.data[0].operation.params)
        rebound = Target(num_qubits=2)
        rebound.add_instruction(exported.data[0].operation)
        rebound.add_instruction(unrestricted)
        rebound.add_instruction(CXGate())
        target = CompilerTarget.from_qiskit(rebound)
        unmapped = QuantumCircuit(2)
        unmapped.compose(exported, inplace=True)
        program = QCProgram.from_qiskit(unmapped).to_qco()
        environment = TargetEnvironment(target, PayloadSpecification(PayloadFormat("openqasm", "3.0")))
        program.compile_for_target(environment)
        exported = program.to_qiskit(target=target)
        actual = exported.assign_parameters({phi: 0.37}) if symbolic else exported
        expected = circuit.assign_parameters({phi: 0.37}) if symbolic else circuit
        assert np.allclose(Operator(actual).data, Operator(expected).data)
        assert np.allclose(Operator(actual.to_gate().control()).data, Operator(expected.to_gate().control()).data)
    gate.definition.global_phase += 0.37
    with pytest.raises(ValueError, match="custom"):
        CompilerTarget.from_qiskit(source, operation_names=[name])


@pytest.mark.parametrize("name", ["gpi", "gpi2"])
def test_native_r_names_require_exact_definitions(name: str) -> None:
    """Reserved native names cannot relabel another gate or change its arity."""
    phi = Parameter("phi")
    source = Target(num_qubits=1)
    source.add_instruction(RGate(pi if name == "gpi" else pi / 2, phi), name=name)
    with pytest.raises(ValueError, match="custom"):
        CompilerTarget.from_qiskit(source, operation_names=[name])

    definition = QuantumCircuit(1, global_phase=pi / 2 if name == "gpi" else 0)
    definition.r(pi if name == "gpi" else pi / 2, phi, 0)
    gate = Gate(name, 1, [phi])
    gate.definition = definition
    source = Target(num_qubits=1)
    source.add_instruction(gate, name="renamed")
    with pytest.raises(ValueError, match="custom"):
        CompilerTarget.from_qiskit(source, operation_names=["renamed"])


@pytest.mark.parametrize("angle", [0.3, Parameter("theta")])
def test_native_r_export_rejects_incompatible_target(angle: float | Parameter) -> None:
    """Direct target construction must not bypass native alias semantics."""
    target = CompilerTarget(
        1,
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations([
            CompilerTarget.OperationCapability("gpi", 1, 2, canonical_name="r"),
        ]),
    )
    circuit = QuantumCircuit(1)
    circuit.r(angle, 0.2, 0)
    program = QCProgram.from_qiskit(circuit).to_qco()
    program.compile_for_target(TargetEnvironment(target, PayloadSpecification(PayloadFormat("openqasm", "3.0"))))
    with pytest.raises(RuntimeError, match="fixed-angle R definition"):
        program.to_qiskit(target=target)


@pytest.mark.parametrize("name", ["cy", "controlled_y"])
def test_native_cy_preserves_capability_without_synthesis_entangler(name: str) -> None:
    """Native CY remains available without claiming a general two-qubit basis."""
    source = Target(num_qubits=2)
    source.add_instruction(UGate(*map(Parameter, ("theta", "phi", "lambda"))))
    source.add_instruction(CYGate(), {(0, 1): None}, name=name)
    target = CompilerTarget.from_qiskit(source, operation_names=["u", name])
    assert target.supports_operation("cy", 2, 0, [0, 1])
    assert not target.supports_operation("cy", 2, 0, [1, 0])
    assert target.synthesis_basis is not None
    assert target.synthesis_basis.entangler is None
    circuit = QuantumCircuit(2)
    circuit.cy(0, 1)
    program = QCProgram.from_qiskit(circuit).to_qco()
    program.compile_for_target(TargetEnvironment(target, PayloadSpecification(PayloadFormat("openqasm", "3.0"))))
    exported = program.to_qiskit(target=target)
    assert [item.operation.name for item in exported.data] == [name]
    restored = QCProgram.from_qiskit(exported).to_qiskit()
    assert np.allclose(Operator(exported).data, Operator(circuit).data)
    assert np.allclose(Operator(restored).data, Operator(circuit).data)
