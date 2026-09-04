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

from mqt.core.qdmi import Device as QDMIDevice
from mqt.core.qdmi import ProgramFormat

from .exceptions import (
    PennyLaneTranslationError as TranslationError,
)
from .exceptions import (
    PennyLaneUnsupportedOperationError as UnsupportedOperationError,
)
from .exceptions import (
    PennyLaneValidationError as ValidationError,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    from pennylane.operation import Operator
    from pennylane.tape import QuantumScript
    from pennylane.wires import Wires


@dataclass(frozen=True)
class _ConvertedProgram:
    """A QDMI payload plus the information required to decode its measurements."""

    payload: str
    program_format: ProgramFormat
    wire_map: Mapping[Hashable, int]
    measurement_order: tuple[int, ...]
    mcm_slot_by_uid: Mapping[str, int]
    output_width: int


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


def _resolve_qasm3_operation(
    operation: Operator, advertised: Mapping[str, QDMIDevice.Operation]
) -> tuple[str, _OperationSpec, QDMIDevice.Operation] | None:
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


def _finite_parameter(parameter: object, operation_name: str) -> float:
    """Convert one bound scalar parameter to a finite Python float.

    Returns:
        The finite scalar parameter.

    Raises:
        PennyLaneValidationError: If the value is unbound, non-scalar, or non-finite.
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


def _validate_operation_shape(operation: Operator, spec: _OperationSpec) -> None:
    """Validate the operation-table arity and parameter contract.

    Raises:
        PennyLaneValidationError: If the operation does not match its typed table row.
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


class _ProgramConverter:
    """Convert preprocessed PennyLane tapes for one opened QDMI device.

    The advertised operation table and the wire mapping do not change while a
    device session is open, so the converter reads them once and reuses them for
    every operation and every tape.

    Args:
        device: Opened QDMI device.
        device_wires: PennyLane wire labels exposed by the device.
        program_format: Program format the device selected.
    """

    def __init__(
        self,
        device: QDMIDevice,
        device_wires: Wires,
        program_format: ProgramFormat,
        capabilities: frozenset[str] = frozenset(),
    ) -> None:
        """Read the advertised capabilities of one opened QDMI device."""
        self._device = device
        self._device_wires = device_wires
        self._program_format = program_format
        self._capabilities = capabilities
        self._advertised = {operation.name().lower(): operation for operation in device.operations()}
        self._wire_map: Mapping[Hashable, int] = MappingProxyType({
            wire: index for index, wire in enumerate(device_wires)
        })
        self._validated_loci: set[tuple[str, tuple[int, ...]]] = set()

    def supports(self, operation: Operator) -> bool:
        """Return whether an operation can stop PennyLane decomposition.

        For OpenQASM 3, support requires an operation-table entry and one matching
        semantic spelling advertised by QDMI. For OpenQASM 2, support additionally
        requires the exact gate spelling produced by PennyLane's serializer.

        Returns:
            Whether the device runs the operation without further decomposition.
        """
        if isinstance(operation, qp.ops.MidMeasure):
            return self._supports_mcm(reset=operation.reset)
        if isinstance(operation, qp.ops.op_math.Conditional):
            return self._supports_mcm() and self.supports(operation.base)
        if self._program_format == ProgramFormat.OPENQASM3:
            return _resolve_qasm3_operation(operation, self._advertised) is not None
        spelling = _QASM2_OPERATIONS.get(operation.name)
        return spelling is not None and spelling in self._advertised

    def convert(self, tape: QuantumScript) -> _ConvertedProgram:
        """Convert one preprocessed tape to the selected program format.

        A QASM3 translation error is never retried as QASM2. QASM2 is selected
        only if QASM3 is not advertised at all.

        Returns:
            The converted program and its deterministic measurement metadata.
        """
        if self._program_format == ProgramFormat.OPENQASM3:
            return self._convert_qasm3(tape)
        return self._convert_qasm2(tape)

    def _program(
        self, tape: QuantumScript, payload: str, mcm_slot_by_uid: Mapping[str, int] | None = None
    ) -> _ConvertedProgram:
        """Attach the measurement-decoding metadata to one converted payload.

        Returns:
            The converted QDMI program.
        """
        return _ConvertedProgram(
            payload=payload,
            program_format=self._program_format,
            wire_map=self._wire_map,
            measurement_order=self._measurement_order(tape),
            mcm_slot_by_uid=MappingProxyType(dict(mcm_slot_by_uid or {})),
            output_width=len(self._device_wires) + len(mcm_slot_by_uid or {}),
        )

    def _supports_mcm(self, *, reset: bool = False) -> bool:
        """Return whether the selected OpenQASM path supports one-shot MCM."""
        required = {
            "mid-circuit-measurement",
            "measured-qubit-reuse",
            "measurement-result-use",
            "boolean-computation",
            "forward-branching",
        }
        operations = self._advertised
        return (
            self._program_format == ProgramFormat.OPENQASM3
            and required <= self._capabilities
            and "measure" in operations
            and (not reset or "reset" in operations)
        )

    def _measurement_order(self, tape: QuantumScript) -> tuple[int, ...]:
        """Return sample columns in the order requested by the transformed tape.

        Returns:
            The device wire indices, one per sample column.
        """
        if not tape.measurements:
            return tuple(self._wire_map.values())
        measured_wires = tape.measurements[0].wires
        if len(measured_wires) == 0:
            return tuple(self._wire_map.values())
        return tuple(self._wire_map[wire] for wire in measured_wires)

    def _validate_qdmi_contract(
        self,
        operation: Operator,
        spec: _OperationSpec,
        qdmi_operation: QDMIDevice.Operation,
        indices: tuple[int, ...],
    ) -> None:
        """Validate operation metadata and any loci advertised by QDMI.

        Raises:
            PennyLaneValidationError: If arity, parameters, or topology do not match.
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

        coupling_map = self._device.coupling_map()
        if coupling_map is None:
            return
        edges = {(first.index(), second.index()) for first, second in coupling_map}
        if indices not in edges and tuple(reversed(indices)) not in edges:
            msg = f"Device topology does not connect wires {indices} for operation '{operation.name}'."
            raise ValidationError(msg)

    def _convert_qasm3(self, tape: QuantumScript) -> _ConvertedProgram:
        """Emit a minimal capability-driven OpenQASM 3 program.

        Returns:
            The converted QDMI program.

        Raises:
            PennyLaneUnsupportedOperationError: If no advertised spelling exists.
            PennyLaneValidationError: If parameters, wires, or topology are invalid.
        """
        mid_measurements = [operation for operation in tape.operations if isinstance(operation, qp.ops.MidMeasure)]
        if any(not operation.meas_uid for operation in mid_measurements):
            msg = "Mid-circuit measurements require unique nonempty measurement IDs."
            raise ValidationError(msg)
        uids = [operation.meas_uid for operation in mid_measurements if operation.meas_uid is not None]
        if len(set(uids)) != len(uids):
            msg = "Mid-circuit measurements require unique nonempty measurement IDs."
            raise ValidationError(msg)
        mcm_slot_by_uid = {uid: len(self._device_wires) + index for index, uid in enumerate(uids)}
        lines = [
            "OPENQASM 3.0;",
            f"qubit[{len(self._device_wires)}] q;",
            f"bit[{len(self._device_wires) + len(uids)}] c;",
        ]
        seen_measurements: set[str] = set()

        for operation in tape.operations:
            if isinstance(operation, qp.ops.MidMeasure):
                if not self._supports_mcm(reset=operation.reset):
                    msg = "The selected payload does not support mid-circuit measurement and reset."
                    raise UnsupportedOperationError(msg)
                wire = self._wire_map[operation.wires[0]]
                uid = operation.meas_uid
                assert uid is not None
                lines.append(f"c[{mcm_slot_by_uid[uid]}] = measure q[{wire}];")
                seen_measurements.add(uid)
                if operation.reset:
                    lines.append(f"reset q[{wire}];")
                continue
            if isinstance(operation, qp.ops.op_math.Conditional):
                if not self._supports_mcm():
                    msg = "The selected payload does not support measurement-conditioned execution."
                    raise UnsupportedOperationError(msg)
                dependencies = operation.meas_val.measurements
                if any(measurement.meas_uid not in seen_measurements for measurement in dependencies):
                    msg = "A conditional uses a measurement that does not precede it."
                    raise ValidationError(msg)
                if len(dependencies) > 16:
                    msg = "A conditional depends on more than 16 measurements."
                    raise ValidationError(msg)
                true_assignments = [assignment for assignment, value in operation.meas_val.items() if bool(value)]
                resolved = _resolve_qasm3_operation(operation.base, self._advertised)
                if resolved is None:
                    msg = f"Conditional operation '{operation.base.name}' is not supported."
                    raise UnsupportedOperationError(msg)
                spelling, spec, qdmi_operation = resolved
                _validate_operation_shape(operation.base, spec)
                indices = tuple(self._wire_map[wire] for wire in operation.base.wires)
                self._validate_qdmi_contract(operation.base, spec, qdmi_operation, indices)
                parameters = ",".join(
                    repr(_finite_parameter(parameter, operation.base.name)) for parameter in operation.base.parameters
                )
                gate = f"{spelling}{f'({parameters})' if parameters else ''} "
                gate += ",".join(f"q[{index}]" for index in indices) + ";"
                for assignment in true_assignments:
                    terms = [
                        f"c[{mcm_slot_by_uid[measurement.meas_uid]}] == {bit}"
                        for measurement, bit in zip(dependencies, assignment, strict=True)
                    ]
                    lines.append(f"if ({' && '.join(terms)}) {{ {gate} }}")
                continue
            resolved = _resolve_qasm3_operation(operation, self._advertised)
            if resolved is None:
                msg = (
                    f"Operation '{operation.name}' has no supported OpenQASM 3 spelling "
                    f"on QDMI device '{self._device.name()}'."
                )
                raise UnsupportedOperationError(msg)
            spelling, spec, qdmi_operation = resolved
            _validate_operation_shape(operation, spec)
            try:
                indices = tuple(self._wire_map[wire] for wire in operation.wires)
            except KeyError as exc:
                msg = f"Operation '{operation.name}' uses wire {exc.args[0]!r}, which is not a device wire."
                raise ValidationError(msg) from exc
            # Capabilities are fixed for this converter's device session. Only
            # cache successful metadata checks; validate each gate's inputs above.
            locus = (operation.name, indices)
            if locus not in self._validated_loci:
                self._validate_qdmi_contract(operation, spec, qdmi_operation, indices)
                self._validated_loci.add(locus)

            # repr gives the shortest literal that reads back as the same double.
            parameters = ",".join(
                repr(_finite_parameter(parameter, operation.name)) for parameter in operation.parameters
            )
            parameter_list = f"({parameters})" if parameters else ""
            operands = ",".join(f"q[{index}]" for index in indices)
            lines.append(f"{spelling}{parameter_list} {operands};")

        lines.extend(f"c[{index}] = measure q[{index}];" for index in range(len(self._device_wires)))
        return self._program(tape, "\n".join(lines) + "\n", mcm_slot_by_uid)

    def _convert_qasm2(self, tape: QuantumScript) -> _ConvertedProgram:
        """Serialize a QASM2-only program with PennyLane's built-in converter.

        Returns:
            The converted QDMI program.

        Raises:
            PennyLaneUnsupportedOperationError: If the serializer/device intersection is empty.
            PennyLaneTranslationError: If PennyLane cannot serialize the program.
        """
        # PennyLane's serializer emits whatever it knows, so the intersection
        # with the advertised gate set has to be checked before serializing.
        for operation in tape.operations:
            spelling = _QASM2_OPERATIONS.get(operation.name)
            if spelling is None or spelling not in self._advertised:
                msg = (
                    f"Operation '{operation.name}' cannot be serialized to an "
                    f"OpenQASM 2 gate advertised by QDMI device '{self._device.name()}'."
                )
                raise UnsupportedOperationError(msg)

        try:
            payload = qp.to_openqasm(
                tape,
                wires=self._device_wires,
                rotations=False,
                measure_all=True,
            )
        except Exception as exc:
            msg = f"Failed to translate the PennyLane program to OpenQASM 2: {exc}"
            raise TranslationError(msg) from exc

        return self._program(tape, payload)
