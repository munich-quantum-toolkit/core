# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""A modern PennyLane device backed by a gate-based QDMI device."""

from __future__ import annotations

import operator
import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pennylane as qp
from pennylane.devices import Device, ExecutionConfig
from pennylane.devices.preprocess import (
    decompose,
    measurements_from_samples,
    validate_device_wires,
    validate_measurements,
)
from pennylane.measurements import CountsMP, ExpectationMP, ProbabilityMP, SampleMP, Shots, VarianceMP
from pennylane.transforms import broadcast_expand, defer_measurements, split_non_commuting
from pennylane.transforms.core import CompilePipeline

from mqt.core import fomac

from .converter import ConvertedProgram, convert_program, supports_operation
from .exceptions import (
    PennyLaneConfigurationError as ConfigurationError,
    PennyLaneExecutionError as ExecutionError,
    PennyLaneUnsupportedFormatError as UnsupportedFormatError,
    PennyLaneUnsupportedOperationError as UnsupportedOperationError,
    PennyLaneValidationError as ValidationError,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    from pennylane.tape import QuantumScript, QuantumScriptOrBatch
    from pennylane.typing import Result, ResultBatch

__all__ = ["DDSIMDevice", "QDMIDevice"]

_SESSION_PARAMETERS = frozenset({
    "base_url",
    "token",
    "auth_file",
    "auth_url",
    "username",
    "password",
    "device_config",
    "device_config_file",
    "custom1",
    "custom2",
    "custom3",
    "custom4",
    "custom5",
})
_JOB_PARAMETERS = frozenset({"custom1", "custom2", "custom3", "custom4", "custom5"})
_SAMPLED_MEASUREMENTS = (SampleMP, CountsMP, ProbabilityMP, ExpectationMP, VarianceMP)


def _validate_parameter_names(parameters: Mapping[str, object], allowed: frozenset[str], kind: str) -> None:
    """Reject unknown FoMaC configuration fields before opening or submission.

    Raises:
        ConfigurationError: If an unknown parameter name is present.
    """
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        msg = f"Unknown QDMI {kind} parameter(s): {', '.join(unknown)}."
        raise ConfigurationError(msg)


@qp.transform
def _validate_finite_shots(tape: QuantumScript) -> tuple[tuple[QuantumScript], Any]:
    """Reject analytic execution before program conversion or submission.

    Returns:
        The unchanged finite-shot tape and its scalar postprocessor.

    Raises:
        ValidationError: If the tape requests analytic execution.
    """
    if not tape.shots:
        msg = "QDMI devices require a finite number of shots."
        raise ValidationError(msg)
    return (tape,), operator.itemgetter(0)


class QDMIDevice(Device):
    """Execute PennyLane programs on a gate-based QDMI device.

    Args:
        device_id: Stable ID from the QDMI device registry.
        wires: PennyLane wire labels or number of wires. By default all QDMI
            qubits are exposed as consecutive integer wires.
        shots: Finite default shot configuration.
        session_parameters: FoMaC device-session keyword arguments.
        job_parameters: FoMaC custom job keyword arguments.
    """

    def __init__(
        self,
        device_id: str,
        wires: int | Sequence[Hashable] | None = None,
        shots: int | Sequence[int | tuple[int, int]] | Shots | None = 1024,
        *,
        session_parameters: Mapping[str, Any] | None = None,
        job_parameters: Mapping[str, str | bool | float | None] | None = None,
    ) -> None:
        """Initialize and open a fresh QDMI device session.

        Raises:
            ConfigurationError: If configuration or requested wires are invalid.
        """
        self._device_id = device_id
        self._session_parameters = dict(session_parameters or {})
        self._job_parameters = dict(job_parameters or {})
        _validate_parameter_names(self._session_parameters, _SESSION_PARAMETERS, "session")
        _validate_parameter_names(self._job_parameters, _JOB_PARAMETERS, "job")

        try:
            self._qdmi_device = fomac.open_device(device_id, **self._session_parameters)
        except (IndexError, RuntimeError, ValueError) as exc:
            msg = f"Failed to open QDMI device '{device_id}': {exc}"
            raise ConfigurationError(msg) from exc

        num_qubits = self._qdmi_device.qubits_num()
        resolved_wires: int | Sequence[Hashable] = num_qubits if wires is None else wires
        requested_wires = resolved_wires if isinstance(resolved_wires, int) else len(resolved_wires)
        if requested_wires <= 0:
            msg = "A QDMI PennyLane device requires at least one wire."
            raise ConfigurationError(msg)
        if requested_wires > num_qubits:
            msg = f"QDMI device '{device_id}' exposes {num_qubits} qubits, but {requested_wires} wires were requested."
            raise ConfigurationError(msg)

        # PennyLane deprecates passing device-level shots to Device.__init__,
        # but still reads Device.shots as the default. Set the validated value
        # after initializing the base class to preserve the finite default
        # without emitting a deprecation warning for every plugin instance.
        super().__init__(wires=resolved_wires, shots=None)
        self._shots = Shots(shots)
        self._program_format = self._select_program_format()
        self._submitted_jobs = 0
        self._execution_time = 0.0

    @property
    def device_id(self) -> str:
        """Stable QDMI device ID."""
        return self._device_id

    @property
    def qdmi_device(self) -> fomac.Device:
        """Opened FoMaC device used for execution."""
        return self._qdmi_device

    @property
    def submitted_jobs(self) -> int:
        """Number of QDMI jobs submitted by this instance."""
        return self._submitted_jobs

    @property
    def execution_time(self) -> float:
        """Cumulative wall-clock time spent submitting and waiting for QDMI jobs."""
        return self._execution_time

    def _select_program_format(self) -> fomac.ProgramFormat:
        """Select QASM3 before QASM2 and reject all other format sets.

        Returns:
            The selected QDMI program format.

        Raises:
            UnsupportedFormatError: If neither OpenQASM version is advertised.
        """
        formats = set(self._qdmi_device.supported_program_formats())
        if fomac.ProgramFormat.QASM3 in formats:
            return fomac.ProgramFormat.QASM3
        if fomac.ProgramFormat.QASM2 in formats:
            return fomac.ProgramFormat.QASM2
        msg = f"QDMI device '{self._device_id}' advertises neither OpenQASM 3 nor OpenQASM 2."
        raise UnsupportedFormatError(msg)

    def preprocess_transforms(self, execution_config: ExecutionConfig | None = None) -> CompilePipeline:
        """Build the PennyLane preprocessing pipeline for sampled QDMI execution.

        Returns:
            The transforms applied before device execution.
        """
        del execution_config
        pipeline = CompilePipeline()
        pipeline.add_transform(_validate_finite_shots)
        pipeline.add_transform(defer_measurements, allow_postselect=False, num_wires=len(self.wires))
        pipeline.add_transform(validate_device_wires, self.wires, name=self.name)
        pipeline.add_transform(
            validate_measurements,
            analytic_measurements=lambda _measurement: False,
            sample_measurements=lambda measurement: isinstance(measurement, _SAMPLED_MEASUREMENTS),
            name=self.name,
        )
        pipeline.add_transform(split_non_commuting, grouping_strategy="qwc")
        pipeline.add_transform(measurements_from_samples)
        pipeline.add_transform(
            decompose,
            stopping_condition=lambda operation: supports_operation(operation, self._qdmi_device, self._program_format),
            stopping_condition_shots=lambda operation: supports_operation(
                operation, self._qdmi_device, self._program_format
            ),
            skip_initial_state_prep=False,
            device_wires=self.wires,
            name=self.name,
            error=UnsupportedOperationError,
        )
        pipeline.add_transform(broadcast_expand)
        return pipeline

    @staticmethod
    def _shot_copies(shots: Shots) -> tuple[int, ...]:
        """Expand a PennyLane shot vector into sequential QDMI job sizes.

        Returns:
            One positive shot count per required QDMI job.

        Raises:
            ValidationError: If execution is analytic.
        """
        if not shots:
            msg = "QDMI devices require a finite number of shots."
            raise ValidationError(msg)
        return tuple(shot_copy.shots for shot_copy in shots.shot_vector for _ in range(shot_copy.copies))

    @staticmethod
    def _shots_or_counts(job: fomac.Job) -> list[str]:
        """Read ordered shots, falling back to an equivalent expansion of counts.

        Returns:
            One QDMI bit string per shot.

        Raises:
            ExecutionError: If the job exposes neither result representation.
        """
        try:
            shots = job.get_shots()
        except RuntimeError:
            shots = []
        if shots:
            return shots

        try:
            counts = job.get_counts()
        except RuntimeError as exc:
            msg = "The QDMI job exposes neither raw shots nor measurement counts."
            raise ExecutionError(msg) from exc
        return [bitstring for bitstring, count in sorted(counts.items()) for _ in range(count)]

    def _samples(self, job: fomac.Job, converted: ConvertedProgram, shots: int) -> np.ndarray:
        """Convert QDMI bit strings to PennyLane sample rows.

        Returns:
            A shot-by-wire array in PennyLane measurement order.

        Raises:
            ExecutionError: If QDMI returns malformed or incomplete results.
        """
        bitstrings = self._shots_or_counts(job)
        if len(bitstrings) != shots:
            msg = f"QDMI returned {len(bitstrings)} samples for a {shots}-shot job."
            raise ExecutionError(msg)

        rows: list[list[int]] = []
        width = len(converted.wire_map)
        for bitstring in bitstrings:
            clean = bitstring.replace(" ", "")
            if len(clean) != width or any(bit not in "01" for bit in clean):
                msg = f"QDMI returned an invalid {width}-wire shot: {bitstring!r}."
                raise ExecutionError(msg)
            # QDMI bit strings use the conventional basis-state spelling with
            # the highest-index site on the left. PennyLane sample columns use
            # the declared wire order, starting with wire zero.
            wire_order = clean[::-1]
            rows.append([int(wire_order[index]) for index in converted.measurement_order])
        return np.asarray(rows, dtype=np.int8)

    @staticmethod
    def _require_done(job: fomac.Job) -> None:
        """Require successful QDMI completion.

        Raises:
            ExecutionError: If the terminal QDMI status is not ``DONE``.
        """
        status = job.check()
        if status != fomac.Job.Status.DONE:
            msg = f"QDMI job '{job.id}' finished with status {status.name}."
            raise ExecutionError(msg)

    def _submit(self, converted: ConvertedProgram, shots: int) -> fomac.Job:
        """Submit and wait for one QDMI job.

        Returns:
            The successfully completed job.

        Raises:
            ExecutionError: If submission, waiting, or execution fails.
        """
        try:
            job = self._qdmi_device.submit_job(
                converted.payload,
                converted.program_format,
                shots,
                **self._job_parameters,
            )
            self._submitted_jobs += 1
            job.wait()
        except (RuntimeError, ValueError) as exc:
            msg = f"QDMI execution on '{self._device_id}' failed: {exc}"
            raise ExecutionError(msg) from exc
        self._require_done(job)
        return job

    def _execute_tape(self, tape: QuantumScript) -> np.ndarray | tuple[np.ndarray, ...]:
        """Execute one preprocessed tape, including every shot-vector partition.

        Returns:
            Raw samples, partitioned when a shot vector was requested.
        """
        converted = convert_program(tape, self._qdmi_device, self.wires)
        results: list[np.ndarray] = []
        for shots in self._shot_copies(tape.shots):
            started = time.monotonic()
            try:
                results.append(self._samples(self._submit(converted, shots), converted, shots))
            finally:
                self._execution_time += time.monotonic() - started

        if tape.shots.has_partitioned_shots:
            return tuple(results)
        return results[0]

    def execute(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig | None = None,
    ) -> Result | ResultBatch:
        """Execute a batch sequentially through QDMI.

        Returns:
            One result for every preprocessed input tape.
        """
        del execution_config
        if isinstance(circuits, qp.tape.QuantumScript):
            return cast("Result", self._execute_tape(circuits))
        return cast("ResultBatch", tuple(self._execute_tape(tape) for tape in circuits))


class DDSIMDevice(QDMIDevice):
    """PennyLane entry point for Core's local DDSIM QDMI device."""

    def __init__(
        self,
        wires: int | Sequence[Hashable] | None = None,
        shots: int | Sequence[int | tuple[int, int]] | Shots | None = 1024,
        *,
        session_parameters: Mapping[str, object] | None = None,
        job_parameters: Mapping[str, str | bool | float | None] | None = None,
    ) -> None:
        """Open the built-in DDSIM device by its stable QDMI ID."""
        super().__init__(
            "mqt.ddsim.default",
            wires=wires,
            shots=shots,
            session_parameters=session_parameters,
            job_parameters=job_parameters,
        )
