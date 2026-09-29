# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Qiskit target conversion for the MQT Compiler Collection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qiskit.circuit import Barrier, ControlFlowOp, ControlledGate, Delay, Parameter, Store
from qiskit.circuit.controlflow import BreakLoopOp, ContinueLoopOp
from qiskit.circuit.library import get_standard_gate_name_mapping
from qiskit.providers import BackendV2
from qiskit.transpiler import Target

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ...mlir import CompilerTarget

__all__ = ["compiler_target_from_qiskit"]


def compiler_target_from_qiskit(
    source: Target | BackendV2,
    *,
    operation_names: Iterable[str] | None = None,
    name: str | None = None,
) -> CompilerTarget:
    """Snapshot standard operations and connectivity from a Qiskit target.

    Preserve ordered operation sites, including directional two-qubit gates.
    Connectivity comes from two-qubit operation sites and is undirected.
    Global phase is circuit metadata and is always permitted.
    Delay, barrier, and control-flow instructions do not
    describe native gates and are omitted. Timing, calibration, and scheduling
    properties are not transferred. This does not assert device support for
    classical control flow or guarantee that a program can be compiled.

    Args:
        source: A target with a known positive qubit count, or a BackendV2.
        operation_names: Restrict the snapshot to these target operation names.
            By default, inspect every operation.
        name: Optional name of the compiler target; defaults to the backend name.

    Returns:
        An immutable compiler target independent of subsequent source changes.

    Raises:
        TypeError: The source is not a Target or BackendV2.
        ValueError: The target has custom gates, constrained parameters (including
            explicit angle bounds), open controls, an unknown width, or
            connectivity that Core cannot represent.
    """
    from ...mlir import CompilerTarget  # ruff: ignore[import-outside-top-level] Keep MLIR optional for QDMI-only use.

    if isinstance(source, BackendV2):
        if name is None:
            name = source.name
        source = source.target
    if not isinstance(source, Target):
        msg = "Expected a Qiskit Target or BackendV2"
        raise TypeError(msg)
    if source.num_qubits is None or source.num_qubits <= 0:
        msg = "Qiskit target must have a known positive qubit count"
        raise ValueError(msg)

    standards = get_standard_gate_name_mapping()
    # Qiskit versions before 2.5 do not support target angle bounds.
    has_angle_bounds = getattr(source, "gate_has_angle_bounds", None)
    operations = []
    couplings: set[tuple[int, int]] = set()
    all_to_all = source.num_qubits == 1
    names = source.operation_names if operation_names is None else operation_names
    for operation_name in sorted(set(names)):
        if operation_name not in source.operation_names:
            msg = f"Qiskit target does not expose operation {operation_name!r}"
            raise ValueError(msg)
        instruction = source.operation_from_name(operation_name)
        qargs = source.qargs_for_operation_name(operation_name)
        if qargs is not None and not qargs:
            continue
        instruction_type = instruction if isinstance(instruction, type) else instruction.base_class
        if issubclass(instruction_type, (Barrier, Delay, ControlFlowOp, BreakLoopOp, ContinueLoopOp, Store)):
            continue
        standard = standards.get(operation_name)
        if isinstance(instruction, type) or standard is None or instruction_type is not standard.base_class:
            msg = f"Cannot represent custom Qiskit target operation {operation_name!r}"
            raise ValueError(msg)
        if isinstance(instruction, ControlledGate) and instruction.ctrl_state != (1 << instruction.num_ctrl_qubits) - 1:
            msg = f"Cannot represent open controls for {operation_name}"
            raise ValueError(msg)
        parameters = instruction.params
        if (
            any(not isinstance(parameter, Parameter) for parameter in parameters)
            or len(set(parameters)) != len(parameters)
            or (has_angle_bounds is not None and has_angle_bounds(operation_name))
        ):
            msg = f"Cannot represent parameter constraints for {operation_name}"
            raise ValueError(msg)
        sites = None if qargs is None else sorted(qargs)
        arity = instruction.num_qubits
        if arity == 2:
            if sites is None:
                all_to_all = True
            else:
                couplings.update((min(a, b), max(a, b)) for a, b in sites)
        if operation_name != "global_phase":
            operations.append(
                CompilerTarget.OperationCapability(operation_name, arity, len(parameters), site_tuples=sites)
            )

    operations.append(CompilerTarget.OperationCapability("gphase", CompilerTarget.OperationArity.fixed(0), 1))
    if len(couplings) == source.num_qubits * (source.num_qubits - 1) // 2:
        all_to_all = True
    connectivity = (
        CompilerTarget.Connectivity.all_to_all() if all_to_all else CompilerTarget.Connectivity(sorted(couplings))
    )
    native_operations = CompilerTarget.NativeOperations(operations)
    if name is not None:
        return CompilerTarget(name, source.num_qubits, connectivity=connectivity, native_operations=native_operations)
    return CompilerTarget(source.num_qubits, connectivity=connectivity, native_operations=native_operations)
