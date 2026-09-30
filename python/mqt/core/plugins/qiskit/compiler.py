# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Qiskit target conversion for the MQT Compiler Collection."""

from __future__ import annotations

from math import inf
from typing import TYPE_CHECKING
from warnings import warn

from qiskit.circuit import Barrier, ControlFlowOp, ControlledGate, Delay, Measure, Parameter, Reset, Store
from qiskit.circuit.controlflow import BreakLoopOp, ContinueLoopOp
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

    Preserve standard gate names and ordered sites, including directional
    two-qubit gates. Target-aware Qiskit export also supports legacy u1/u3 names.
    Connectivity comes from two-qubit operation sites and is undirected.
    Global phase is circuit metadata and is always permitted.
    Gate recognition uses the same adapter and Qiskit versions as circuit import/export.
    Custom names, unsupported gates, fixed parameters, and restricted angles
    are omitted with a warning unless explicitly selected. Parameter expressions
    are unrestricted slots, following Qiskit's target-matching semantics.
    Delay, barrier, control flow, and empty applicability are ignored by default
    and rejected when explicitly selected. Timing, calibration, and scheduling
    properties are not transferred. This does not assert device support for
    classical control flow or guarantee that a program can be compiled.

    Args:
        source: A target with a known positive qubit count, or a BackendV2.
        operation_names: Restrict the snapshot to these target operation names.
            By default, retain every representable operation. Explicitly selected
            operations must all be representable.
        name: Optional name of the compiler target; defaults to the backend name.

    Returns:
        An immutable compiler target independent of subsequent source changes.

    Raises:
        TypeError: The source is not a Target or BackendV2.
        ValueError: An explicitly selected operation is unsupported, no native
            operations remain, or the width or connectivity cannot be represented.
    """
    from ...mlir import (  # ruff: ignore[import-outside-top-level] Keep MLIR optional for QDMI-only use.
        CompilerTarget,
        _qiskit_native_gate_name,
    )

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

    operations = []
    couplings: set[tuple[int, int]] = set()
    all_to_all = source.num_qubits == 1
    names = source.operation_names if operation_names is None else operation_names
    parameter = Parameter("_mqt_target_parameter")
    for operation_name in sorted(set(names)):
        if operation_name not in source.operation_names:
            msg = f"Qiskit target does not expose operation {operation_name!r}"
            raise ValueError(msg)
        instruction = source.operation_from_name(operation_name)
        qargs = source.qargs_for_operation_name(operation_name)
        instruction_type = instruction if isinstance(instruction, type) else type(instruction)
        if isinstance(instruction, Measure):
            native_name = "measure"
        elif isinstance(instruction, Reset):
            native_name = "reset"
        else:
            native_name = _qiskit_native_gate_name(instruction)
        if (
            (qargs is not None and not qargs)
            or issubclass(instruction_type, (Barrier, Delay, ControlFlowOp, BreakLoopOp, ContinueLoopOp, Store))
            or native_name == "gphase"
        ):
            if operation_names is not None:
                msg = f"Qiskit target operation {operation_name!r} has no native gate applicability"
                raise ValueError(msg)
            continue
        reason = None
        if isinstance(instruction, ControlledGate) and instruction.ctrl_state != (1 << instruction.num_ctrl_qubits) - 1:
            reason = "open controls"
        elif native_name is None:
            reason = "custom or unsupported operation"
        elif operation_name != native_name:
            reason = "custom operation name"
        elif not source.instruction_supported(operation_name, parameters=[parameter] * len(instruction.params)) or (
            source.gate_has_angle_bounds(operation_name)
            and any(
                not source.supported_angle_bound(operation_name, [bound] * len(instruction.params))
                for bound in (-inf, inf)
            )
        ):
            reason = "parameter constraints"
        if reason is not None:
            msg = f"Cannot represent {reason} for {operation_name!r}"
            if operation_names is not None:
                raise ValueError(msg)
            warn(f"{msg}; omitting it from the compiler target", UserWarning, stacklevel=2)
            continue
        assert native_name is not None
        sites = None if qargs is None else sorted(qargs)
        arity = instruction.num_qubits
        if arity == 2:
            if sites is None:
                all_to_all = True
            else:
                couplings.update((min(a, b), max(a, b)) for a, b in sites)
        operations.append(
            CompilerTarget.OperationCapability(operation_name, arity, len(instruction.params), site_tuples=sites)
        )

    if not operations:
        msg = "Qiskit target has no representable native operations"
        raise ValueError(msg)
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
