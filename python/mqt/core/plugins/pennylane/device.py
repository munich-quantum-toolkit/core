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
from contextlib import suppress
from time import monotonic
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pennylane as qp
from pennylane.devices import Device, DeviceCapabilities, ExecutionConfig
from pennylane.devices.preprocess import (
    decompose,
    measurements_from_samples,
    validate_device_wires,
    validate_measurements,
)
from pennylane.measurements import CountsMP, ExpectationMP, ProbabilityMP, SampleMP, Shots, VarianceMP
from pennylane.transforms import broadcast_expand, defer_measurements, split_non_commuting
from pennylane.transforms.core import CompilePipeline

from mqt.core.qdmi import Device as QDMIDeviceHandle
from mqt.core.qdmi import Job as QDMIJobHandle
from mqt.core.qdmi import ProgramFormat
from mqt.core.qdmi.driver import open_device

from .converter import _ConvertedProgram, _ProgramConverter
from .exceptions import (
    PennyLaneConfigurationError as ConfigurationError,
)
from .exceptions import (
    PennyLaneExecutionError as ExecutionError,
)
from .exceptions import (
    PennyLaneUnsupportedFormatError as UnsupportedFormatError,
)
from .exceptions import (
    PennyLaneUnsupportedOperationError as UnsupportedOperationError,
)
from .exceptions import (
    PennyLaneValidationError as ValidationError,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    from pennylane.tape import QuantumScript, QuantumScriptOrBatch
    from pennylane.typing import Result, ResultBatch
    from pennylane.wires import Wires

    from mqt.core.typing import QDMIJobParameters, QDMISessionParameters

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
    """Reject unknown QDMI configuration fields before opening or submission.

    Raises:
        PennyLaneConfigurationError: If an unknown parameter name is present.
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
        PennyLaneValidationError: If the tape requests analytic execution.
    """
    if not tape.shots:
        msg = "QDMI devices require a finite number of shots."
        raise ValidationError(msg)
    return (tape,), operator.itemgetter(0)


@qp.transform
def _defer_on_device_wires(tape: QuantumScript, wires: Wires) -> tuple[tuple[QuantumScript], Any]:
    """Defer measurements using unused device wires, including custom labels.

    Returns:
        The transformed tape and PennyLane's result postprocessor.

    Raises:
        PennyLaneValidationError: If deferral requires more wires than the device exposes.
    """
    if not any(isinstance(operation, qp.ops.MidMeasure) for operation in tape.operations):
        return (tape,), operator.itemgetter(0)
    ordered_wires = [*tape.wires, *(wire for wire in wires if wire not in tape.wires)]
    wire_map = {wire: index for index, wire in enumerate(ordered_wires)}
    (mapped,), _ = qp.map_wires(tape, wire_map)
    (deferred,), postprocess = defer_measurements(mapped, allow_postselect=False)
    if any(wire >= len(wires) for wire in deferred.wires):
        msg = "Deferred measurements require more wires than the QDMI device exposes."
        raise ValidationError(msg)
    (restored,), _ = qp.map_wires(deferred, dict(enumerate(ordered_wires)))
    return (restored,), postprocess


class QDMIDevice(Device):
    """Execute PennyLane programs on a gate-based QDMI device.

    Args:
        device_id: Stable ID from the QDMI device registry. Use either this
            argument or ``device``.
        wires: PennyLane wire labels or number of wires. By default all QDMI
            qubits are exposed as consecutive integer wires.
        device: An already-open QDMI device. Use this for a session selected by
            an integration such as Slurm.
        session_parameters: QDMI device-session keyword arguments.
        job_parameters: QDMI custom job keyword arguments.
    """

    capabilities = DeviceCapabilities(supported_mcm_methods=[])
    """Backend capabilities described by :class:`~pennylane.devices.capabilities.DeviceCapabilities`."""

    def __init__(
        self,
        device_id: str | None = None,
        wires: int | Sequence[Hashable] | None = None,
        *,
        device: QDMIDeviceHandle | None = None,
        session_parameters: QDMISessionParameters | None = None,
        job_parameters: QDMIJobParameters | None = None,
    ) -> None:
        """Initialize from a stable ID or an open QDMI device.

        Raises:
            PennyLaneConfigurationError: If configuration or requested wires are invalid.
        """
        self._session_parameters = dict(session_parameters or {})
        self._job_parameters = dict(job_parameters or {})
        _validate_parameter_names(self._session_parameters, _SESSION_PARAMETERS, "session")
        _validate_parameter_names(self._job_parameters, _JOB_PARAMETERS, "job")

        if (device_id is None) == (device is None):
            msg = "Specify exactly one of device_id and device."
            raise ConfigurationError(msg)
        if device is not None:
            if self._session_parameters:
                msg = "session_parameters cannot be used with an already-open QDMI device."
                raise ConfigurationError(msg)
            self._qdmi_device = device
        else:
            assert device_id is not None
            try:
                self._qdmi_device = open_device(device_id, **self._session_parameters)
            except (IndexError, RuntimeError, ValueError) as exc:
                msg = f"Failed to open QDMI device '{device_id}': {exc}"
                raise ConfigurationError(msg) from exc
        self._device_id = device_id
        self._device_name = device_id or self._qdmi_device.name()

        num_qubits = self._qdmi_device.qubits_num()
        resolved_wires: int | Sequence[Hashable] = num_qubits if wires is None else wires
        requested_wires = resolved_wires if isinstance(resolved_wires, int) else len(resolved_wires)
        if requested_wires <= 0:
            msg = "A QDMI PennyLane device requires at least one wire."
            raise ConfigurationError(msg)
        if requested_wires > num_qubits:
            msg = (
                f"QDMI device '{self._device_name}' exposes {num_qubits} qubits, "
                f"but {requested_wires} wires were requested."
            )
            raise ConfigurationError(msg)

        super().__init__(wires=resolved_wires)
        self._program_format = self._select_program_format()
        self._converter = _ProgramConverter(self._qdmi_device, self.wires, self._program_format)
        self._submitted_jobs = 0
        self._execution_time = 0.0

    @property
    def device_id(self) -> str | None:
        """Stable QDMI device ID, if the device was opened by ID."""
        return self._device_id

    @property
    def qdmi_device(self) -> QDMIDeviceHandle:
        """Opened QDMI device used for execution."""
        return self._qdmi_device

    @property
    def submitted_jobs(self) -> int:
        """Number of QDMI jobs submitted by this instance."""
        return self._submitted_jobs

    @property
    def execution_time(self) -> float:
        """Cumulative wall-clock time spent submitting and waiting for QDMI jobs."""
        return self._execution_time

    def _select_program_format(self) -> ProgramFormat:
        """Select QASM3 before QASM2 and reject all other format sets.

        Returns:
            The selected QDMI program format.

        Raises:
            PennyLaneUnsupportedFormatError: If neither OpenQASM version is advertised.
        """
        formats = set(self._qdmi_device.supported_program_formats())
        if ProgramFormat.QASM3 in formats:
            return ProgramFormat.QASM3
        if ProgramFormat.QASM2 in formats:
            return ProgramFormat.QASM2
        msg = f"QDMI device '{self._device_name}' advertises neither OpenQASM 3 nor OpenQASM 2."
        raise UnsupportedFormatError(msg)

    def preprocess_transforms(self, execution_config: ExecutionConfig | None = None) -> CompilePipeline:
        """Build the PennyLane preprocessing pipeline for sampled QDMI execution.

        Returns:
            The transforms applied before device execution.
        """
        del execution_config
        pipeline = CompilePipeline()
        pipeline.add_transform(_validate_finite_shots)
        pipeline.add_transform(validate_device_wires, self.wires, name=self.name)
        pipeline.add_transform(_defer_on_device_wires, self.wires)
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
            stopping_condition=self._converter.supports,
            target_gates=self._converter.target_gates,
            skip_initial_state_prep=False,
            device_wires=self.wires,
            name=self.name,
            error=UnsupportedOperationError,
        )
        pipeline.add_transform(broadcast_expand)
        return pipeline

    @staticmethod
    def _shot_copies(shots: Shots) -> tuple[int, ...]:
        """Expand a PennyLane shot vector into individual QDMI job sizes.

        Returns:
            One positive shot count per required QDMI job.

        Raises:
            PennyLaneValidationError: If execution is analytic.
        """
        if not shots:
            msg = "QDMI devices require a finite number of shots."
            raise ValidationError(msg)
        return tuple(shot_copy.shots for shot_copy in shots.shot_vector for _ in range(shot_copy.copies))

    @staticmethod
    def _shots_or_counts(job: QDMIJobHandle) -> list[str]:
        """Read ordered shots, falling back to an equivalent expansion of counts.

        Returns:
            One QDMI bit string per shot.

        Raises:
            PennyLaneExecutionError: If the job exposes neither result representation.
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

    def _samples(self, job: QDMIJobHandle, converted: _ConvertedProgram, shots: int) -> np.ndarray:
        """Convert QDMI bit strings to PennyLane sample rows.

        Returns:
            A shot-by-wire array in PennyLane measurement order.

        Raises:
            PennyLaneExecutionError: If QDMI returns malformed or incomplete results.
        """
        bitstrings = self._shots_or_counts(job)
        if len(bitstrings) != shots:
            msg = f"QDMI returned {len(bitstrings)} samples for a {shots}-shot job."
            raise ExecutionError(msg)

        width = len(converted.wire_map)
        cleaned: list[str] = []
        for bitstring in bitstrings:
            clean = bitstring.replace(" ", "")
            if len(clean) != width or clean.strip("01"):
                msg = f"QDMI returned an invalid {width}-wire shot: {bitstring!r}."
                raise ExecutionError(msg)
            cleaned.append(clean)
        if not bitstrings:
            return np.asarray([], dtype=np.int8)
        packed = np.frombuffer("".join(cleaned).encode("ascii"), dtype=np.int8).reshape(shots, width)
        # QDMI spells the highest-index site first; PennyLane starts with wire zero.
        return packed[:, ::-1][:, converted.measurement_order] - ord("0")

    @staticmethod
    def _require_done(job: QDMIJobHandle) -> None:
        """Require successful QDMI completion.

        Raises:
            PennyLaneExecutionError: If the terminal QDMI status is not ``DONE``.
        """
        status = job.check()
        if status != QDMIJobHandle.Status.DONE:
            msg = f"QDMI job '{job.id}' finished with status {status.name}."
            raise ExecutionError(msg)

    def _submit(self, converted: _ConvertedProgram, shots: int) -> QDMIJobHandle:
        """Submit one QDMI job.

        Returns:
            The submitted job.

        Raises:
            PennyLaneExecutionError: If submission fails.
        """
        try:
            job = self._qdmi_device.submit_job(
                converted.payload,
                converted.program_format,
                shots,
                **self._job_parameters,
            )
            self._submitted_jobs += 1
        except (RuntimeError, ValueError) as exc:
            msg = f"QDMI execution on '{self._device_name}' failed: {exc}"
            raise ExecutionError(msg) from exc
        return job

    def _result(self, job: QDMIJobHandle, converted: _ConvertedProgram, shots: int) -> np.ndarray:
        """Wait for and decode one QDMI job.

        Returns:
            Raw samples from the completed job.

        Raises:
            PennyLaneExecutionError: If waiting or execution fails.
        """
        try:
            job.wait()
        except (RuntimeError, ValueError) as exc:
            msg = f"QDMI execution on '{self._device_name}' failed: {exc}"
            raise ExecutionError(msg) from exc
        self._require_done(job)
        return self._samples(job, converted, shots)

    def execute(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig | None = None,
    ) -> Result | ResultBatch:
        """Submit a batch to QDMI before collecting its ordered results.

        Returns:
            One result for every preprocessed input tape.
        """
        del execution_config
        single = isinstance(circuits, qp.tape.QuantumScript)
        tapes = (circuits,) if single else tuple(circuits)
        prepared = tuple((self._converter.convert(tape), self._shot_copies(tape.shots)) for tape in tapes)
        if not prepared:
            return cast("ResultBatch", ())

        if self.tracker.active:
            self.tracker.update(batches=1, batch_len=len(tapes))
            self.tracker.record()

        submitted: list[tuple[int, _ConvertedProgram, int, QDMIJobHandle]] = []
        tape_results: list[list[np.ndarray]] = [[] for _ in tapes]
        started = monotonic()
        try:
            for index, (converted, shot_copies) in enumerate(prepared):
                for shots in shot_copies:
                    submitted.append((index, converted, shots, self._submit(converted, shots)))
                    if self.tracker.active:
                        self.tracker.update(executions=1, shots=shots)
                        self.tracker.record()

            for index, converted, shots, job in submitted:
                tape_results[index].append(self._result(job, converted, shots))
        except BaseException:
            for *_unused, job in submitted:
                with suppress(BaseException):
                    job.cancel()
            raise
        finally:
            self._execution_time += monotonic() - started

        results = tuple(
            tuple(samples) if tape.shots.has_partitioned_shots else samples[0]
            for tape, samples in zip(tapes, tape_results, strict=True)
        )
        if single:
            return cast("Result", results[0])
        return cast("ResultBatch", results)


class DDSIMDevice(QDMIDevice):
    """PennyLane entry point for MQT Core's local DDSIM QDMI device."""

    def __init__(
        self,
        wires: int | Sequence[Hashable] | None = None,
        *,
        session_parameters: QDMISessionParameters | None = None,
        job_parameters: QDMIJobParameters | None = None,
    ) -> None:
        """Open the built-in DDSIM device by its stable QDMI ID."""
        super().__init__(
            "mqt.ddsim.default",
            wires=wires,
            session_parameters=session_parameters,
            job_parameters=job_parameters,
        )
