# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Contracts for converting Qiskit compiler targets."""

from __future__ import annotations

import pytest
from qiskit.circuit import Gate, Parameter, ParameterExpression
from qiskit.circuit.controlflow import IfElseOp
from qiskit.circuit.library import CXGate, RZGate, UGate, XGate
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import Target

from mqt.core.mlir import CompilerTarget
from mqt.core.plugins.qiskit import compiler_target_from_qiskit


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
    with pytest.raises(ValueError, match="parameter constraints for rz"):
        compiler_target_from_qiskit(source)


def test_correlated_parameters_and_custom_gates() -> None:
    """Equal parameter slots and custom gates must not be silently broadened."""
    theta = Parameter("theta")
    source = Target(num_qubits=1)
    source.add_instruction(UGate(theta, theta, theta))
    source.add_instruction(Gate("x", 1, []))
    with pytest.raises(ValueError, match="parameter constraints for u"):
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


def test_angle_bounds_and_open_controls() -> None:
    """Instruction metadata also constrains the accepted gate semantics."""
    source = Target(num_qubits=2)
    source.add_instruction(CXGate(ctrl_state=0), name="cx")
    with pytest.raises(ValueError, match="open controls"):
        compiler_target_from_qiskit(source)
    if not hasattr(Target, "gate_has_angle_bounds"):
        pytest.skip("Target angle bounds require Qiskit 2.5")
    source = Target(num_qubits=1)
    source.add_instruction(RZGate(Parameter("theta")), angle_bounds=[(-1.0, 1.0)])
    with pytest.raises(ValueError, match="parameter constraints"):
        compiler_target_from_qiskit(source)


def test_unknown_width_and_disconnected_topology() -> None:
    """Missing connectivity must not turn into an all-to-all device."""
    with pytest.raises(ValueError, match="positive qubit count"):
        compiler_target_from_qiskit(Target(num_qubits=None))
    source = Target(num_qubits=2)
    source.add_instruction(XGate())
    source.add_instruction(CXGate(), {})
    with pytest.raises(ValueError, match="connected"):
        compiler_target_from_qiskit(source)
