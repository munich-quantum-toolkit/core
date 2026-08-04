# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Convert preprocessed PennyLane programs to QDMI program formats."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import pennylane as qp

from mqt.core import fomac

from .exceptions import (
    PennyLaneTranslationError as TranslationError,
    PennyLaneUnsupportedFormatError as UnsupportedFormatError,
    PennyLaneUnsupportedOperationError as UnsupportedOperationError,
    PennyLaneValidationError as ValidationError,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    from pennylane.operation import Operator
    from pennylane.tape import QuantumScript
    from pennylane.wires import Wires

__all__ = ["ConvertedProgram", "convert_program", "supports_operation"]


@dataclass(frozen=True)
class ConvertedProgram:
    """A QDMI payload plus the information required to decode its measurements."""

    payload: str
    program_format: fomac.ProgramFormat
    wire_map: Mapping[Hashable, int]
    measurement_order: tuple[int, ...]


@dataclass(frozen=True)
class _OperationSpec:
    aliases: tuple[str, ...]
    wires: int
    parameters: int


_QASM3_OPERATIONS: Mapping[str, _OperationSpec] = MappingProxyType({
    "Identity": _OperationSpec(("i", "id"), 1, 0),
    "PauliX": _OperationSpec(("x",), 1, 0),
    "PauliY": _OperationSpec(("y",), 1, 0),
    "PauliZ": _OperationSpec(("z",), 1, 0),
    "Hadamard": _OperationSpec(("h",), 1, 0),
    "S": _OperationSpec(("s",), 1, 0),
    "Adjoint(S)": _OperationSpec(("sdg", "si"), 1, 0),
    "T": _OperationSpec(("t",), 1, 0),
    "Adjoint(T)": _OperationSpec(("tdg", "ti"), 1, 0),
    "SX": _OperationSpec(("sx", "v"), 1, 0),
    "Adjoint(SX)": _OperationSpec(("sxdg", "vi"), 1, 0),
    "RX": _OperationSpec(("rx",), 1, 1),
    "RY": _OperationSpec(("ry",), 1, 1),
    "RZ": _OperationSpec(("rz",), 1, 1),
    "PhaseShift": _OperationSpec(("p", "phaseshift"), 1, 1),
    "CNOT": _OperationSpec(("cx", "cnot"), 2, 0),
    "CY": _OperationSpec(("cy",), 2, 0),
    "CZ": _OperationSpec(("cz",), 2, 0),
    "ControlledPhaseShift": _OperationSpec(("cp", "cphaseshift"), 2, 1),
    "CPhaseShift00": _OperationSpec(("cphaseshift00",), 2, 1),
    "CPhaseShift01": _OperationSpec(("cphaseshift01",), 2, 1),
    "CPhaseShift10": _OperationSpec(("cphaseshift10",), 2, 1),
    "Toffoli": _OperationSpec(("ccx", "ccnot"), 3, 0),
    "SWAP": _OperationSpec(("swap",), 2, 0),
    "CSWAP": _OperationSpec(("cswap",), 3, 0),
    "ISWAP": _OperationSpec(("iswap",), 2, 0),
    "PSWAP": _OperationSpec(("pswap",), 2, 1),
    "ECR": _OperationSpec(("ecr",), 2, 0),
    "IsingXX": _OperationSpec(("rxx", "xx"), 2, 1),
    "IsingXY": _OperationSpec(("rxy", "xy"), 2, 1),
    "IsingYY": _OperationSpec(("ryy", "yy"), 2, 1),
    "IsingZZ": _OperationSpec(("rzz", "zz"), 2, 1),
})

# PennyLane's OpenQASM 2 serializer emits these exact qelib1 gate names.
_QASM2_OPERATIONS: Mapping[str, str] = MappingProxyType({
    "CNOT": "cx",
    "CZ": "cz",
    "U3": "u3",
    "U2": "u2",
    "U1": "u1",
    "Identity": "id",
    "PauliX": "x",
    "PauliY": "y",
    "PauliZ": "z",
    "Hadamard": "h",
    "S": "s",
    "Adjoint(S)": "sdg",
    "T": "t",
    "Adjoint(T)": "tdg",
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "CRX": "crx",
    "CRY": "cry",
    "CRZ": "crz",
    "SWAP": "swap",
    "Toffoli": "ccx",
    "CSWAP": "cswap",
    "PhaseShift": "u1",
})


def _device_operations(device: fomac.Device) -> dict[str, fomac.Device.Operation]:
    """Return advertised operations keyed by their lower-case spelling."""
    return {operation.name().lower(): operation for operation in device.operations()}


def _preferred_format(device: fomac.Device) -> fomac.ProgramFormat:
    """Choose the required QASM3-first exchange format.

    Returns:
        The selected QDMI program format.

    Raises:
        UnsupportedFormatError: If neither OpenQASM version is advertised.
    """
    formats = set(device.supported_program_formats())
    if fomac.ProgramFormat.QASM3 in formats:
        return fomac.ProgramFormat.QASM3
    if fomac.ProgramFormat.QASM2 in formats:
        return fomac.ProgramFormat.QASM2
    msg = f"QDMI device '{device.name()}' supports none of the required program formats: OpenQASM 3 or OpenQASM 2."
    raise UnsupportedFormatError(msg)


def _resolve_qasm3_operation(
    operation: Operator, advertised: Mapping[str, fomac.Device.Operation]
) -> tuple[str, _OperationSpec, fomac.Device.Operation] | None:
    """Resolve a PennyLane operation to one advertised QDMI spelling.

    Returns:
        The spelling, operation specification, and QDMI operation, or ``None``.
    """
    spec = _QASM3_OPERATIONS.get(operation.name)
    if spec is None:
        return None
    for alias in spec.aliases:
        if alias in advertised:
            return alias, spec, advertised[alias]
    return None


def supports_operation(operation: Operator, device: fomac.Device, program_format: fomac.ProgramFormat) -> bool:
    """Return whether an operation can stop PennyLane decomposition.

    For OpenQASM 3, support requires an operation-table entry and one matching
    semantic spelling advertised by QDMI. For OpenQASM 2, support additionally
    requires the exact gate spelling produced by PennyLane's serializer.
    """
    advertised = _device_operations(device)
    if program_format == fomac.ProgramFormat.QASM3:
        return _resolve_qasm3_operation(operation, advertised) is not None
    if program_format == fomac.ProgramFormat.QASM2:
        spelling = _QASM2_OPERATIONS.get(operation.name)
        return spelling is not None and spelling in advertised
    return False


def _wire_mapping(device_wires: Wires) -> Mapping[Hashable, int]:
    """Build an immutable, deterministic wire-label mapping.

    Returns:
        The mapping from wire labels to contiguous QASM indices.
    """
    return MappingProxyType({wire: index for index, wire in enumerate(device_wires)})


def _measurement_order(tape: QuantumScript, wire_map: Mapping[Hashable, int]) -> tuple[int, ...]:
    """Return sample columns in the order requested by the transformed tape."""
    if not tape.measurements:
        return tuple(wire_map.values())
    measured_wires = tape.measurements[0].wires
    if len(measured_wires) == 0:
        return tuple(wire_map.values())
    return tuple(wire_map[wire] for wire in measured_wires)


def _finite_parameter(parameter: object, operation_name: str) -> float:
    """Convert one bound scalar parameter to a finite Python float.

    Returns:
        The finite scalar parameter.

    Raises:
        ValidationError: If the value is unbound, non-scalar, or non-finite.
    """
    try:
        value = float(qp.math.toarray(parameter))
    except (TypeError, ValueError) as exc:
        msg = f"Operation '{operation_name}' has an unbound or non-scalar parameter: {parameter!r}."
        raise ValidationError(msg) from exc
    if not math.isfinite(value):
        msg = f"Operation '{operation_name}' has a non-finite parameter: {value!r}."
        raise ValidationError(msg)
    return value


def _format_parameter(parameter: object, operation_name: str) -> str:
    """Format one QASM parameter without losing double precision.

    Returns:
        The OpenQASM numeric literal.
    """
    return format(_finite_parameter(parameter, operation_name), ".17g")


def _validate_operation_shape(operation: Operator, spec: _OperationSpec) -> None:
    """Validate the operation-table arity and parameter contract.

    Raises:
        ValidationError: If the operation does not match its typed table row.
    """
    if len(operation.wires) != spec.wires:
        msg = f"Operation '{operation.name}' requires {spec.wires} wires, but received {len(operation.wires)}."
        raise ValidationError(msg)
    if len(set(operation.wires)) != len(operation.wires):
        msg = f"Operation '{operation.name}' uses the same wire more than once."
        raise ValidationError(msg)
    if len(operation.parameters) != spec.parameters:
        msg = (
            f"Operation '{operation.name}' requires {spec.parameters} parameters, "
            f"but received {len(operation.parameters)}."
        )
        raise ValidationError(msg)


def _validate_qdmi_contract(
    operation: Operator,
    spec: _OperationSpec,
    qdmi_operation: fomac.Device.Operation,
    indices: tuple[int, ...],
    device: fomac.Device,
) -> None:
    """Validate operation metadata and any loci advertised by QDMI.

    Raises:
        ValidationError: If arity, parameters, or topology do not match.
    """
    qdmi_wires = qdmi_operation.qubits_num()
    if qdmi_wires is not None and qdmi_wires != spec.wires:
        msg = (
            f"QDMI operation '{qdmi_operation.name()}' advertises {qdmi_wires} wires, "
            f"but '{operation.name}' requires {spec.wires}."
        )
        raise ValidationError(msg)
    if qdmi_operation.parameters_num() != spec.parameters:
        msg = (
            f"QDMI operation '{qdmi_operation.name()}' advertises "
            f"{qdmi_operation.parameters_num()} parameters, but '{operation.name}' "
            f"requires {spec.parameters}."
        )
        raise ValidationError(msg)

    if spec.wires == 1:
        sites = qdmi_operation.sites()
        if sites is not None and indices[0] not in {site.index() for site in sites}:
            msg = f"Operation '{operation.name}' is not advertised on device wire {indices[0]}."
            raise ValidationError(msg)
        return

    if spec.wires != 2:
        return

    site_pairs = qdmi_operation.site_pairs()
    if site_pairs is not None:
        advertised_pairs = {(first.index(), second.index()) for first, second in site_pairs}
        if indices not in advertised_pairs:
            msg = f"Operation '{operation.name}' is not advertised on device wires {indices}."
            raise ValidationError(msg)
        return

    coupling_map = device.coupling_map()
    if coupling_map is None:
        return
    edges = {(first.index(), second.index()) for first, second in coupling_map}
    if indices not in edges and tuple(reversed(indices)) not in edges:
        msg = f"Device topology does not connect wires {indices} for operation '{operation.name}'."
        raise ValidationError(msg)


def _convert_qasm3(
    tape: QuantumScript,
    device: fomac.Device,
    device_wires: Wires,
) -> ConvertedProgram:
    """Emit a minimal capability-driven OpenQASM 3 program.

    Returns:
        The converted QDMI program.

    Raises:
        UnsupportedOperationError: If no advertised spelling exists.
        ValidationError: If parameters, wires, or topology are invalid.
    """
    wire_map = _wire_mapping(device_wires)
    advertised = _device_operations(device)
    lines = [
        "OPENQASM 3.0;",
        f"qubit[{len(device_wires)}] q;",
        f"bit[{len(device_wires)}] c;",
    ]

    for operation in tape.operations:
        resolved = _resolve_qasm3_operation(operation, advertised)
        if resolved is None:
            msg = f"Operation '{operation.name}' has no supported OpenQASM 3 spelling on QDMI device '{device.name()}'."
            raise UnsupportedOperationError(msg)
        spelling, spec, qdmi_operation = resolved
        _validate_operation_shape(operation, spec)
        try:
            indices = tuple(wire_map[wire] for wire in operation.wires)
        except KeyError as exc:
            msg = f"Operation '{operation.name}' uses wire {exc.args[0]!r}, which is not a device wire."
            raise ValidationError(msg) from exc
        _validate_qdmi_contract(operation, spec, qdmi_operation, indices, device)

        parameters = ",".join(_format_parameter(parameter, operation.name) for parameter in operation.parameters)
        parameter_list = f"({parameters})" if parameters else ""
        operands = ",".join(f"q[{index}]" for index in indices)
        lines.append(f"{spelling}{parameter_list} {operands};")

    lines.append("c = measure q;")
    payload = "\n".join(lines) + "\n"
    return ConvertedProgram(
        payload=payload,
        program_format=fomac.ProgramFormat.QASM3,
        wire_map=wire_map,
        measurement_order=_measurement_order(tape, wire_map),
    )


def _convert_qasm2(
    tape: QuantumScript,
    device: fomac.Device,
    device_wires: Wires,
) -> ConvertedProgram:
    """Serialize a QASM2-only program with PennyLane's built-in converter.

    Returns:
        The converted QDMI program.

    Raises:
        UnsupportedOperationError: If the serializer/device intersection is empty.
        TranslationError: If PennyLane cannot serialize the program.
    """
    advertised = _device_operations(device)
    for operation in tape.operations:
        spelling = _QASM2_OPERATIONS.get(operation.name)
        if spelling is None or spelling not in advertised:
            msg = (
                f"Operation '{operation.name}' cannot be serialized to an "
                f"OpenQASM 2 gate advertised by QDMI device '{device.name()}'."
            )
            raise UnsupportedOperationError(msg)

    try:
        payload = qp.to_openqasm(
            tape,
            wires=device_wires,
            rotations=False,
            measure_all=True,
        )
    except Exception as exc:
        msg = f"Failed to translate the PennyLane program to OpenQASM 2: {exc}"
        raise TranslationError(msg) from exc

    wire_map = _wire_mapping(device_wires)
    return ConvertedProgram(
        payload=payload,
        program_format=fomac.ProgramFormat.QASM2,
        wire_map=wire_map,
        measurement_order=_measurement_order(tape, wire_map),
    )


def convert_program(
    tape: QuantumScript,
    device: fomac.Device,
    device_wires: Wires,
) -> ConvertedProgram:
    """Convert one preprocessed tape using QASM3-first negotiation.

    A QASM3 translation error is never retried as QASM2. QASM2 is selected
    only if QASM3 is not advertised at all.

    Returns:
        The converted program and its deterministic measurement metadata.
    """
    program_format = _preferred_format(device)
    if program_format == fomac.ProgramFormat.QASM3:
        return _convert_qasm3(tape, device, device_wires)
    return _convert_qasm2(tape, device, device_wires)
