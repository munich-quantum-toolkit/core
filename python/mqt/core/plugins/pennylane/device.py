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
from time import monotonic
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pennylane as qp
from pennylane.devices import Device, DeviceCapabilities, ExecutionConfig, MCMConfig
from pennylane.devices.preprocess import (
    decompose,
    measurements_from_samples,
    validate_device_wires,
    validate_measurements,
)
from pennylane.measurements import (
    CountsMP,
    ExpectationMP,
    MeasurementValue,
    ProbabilityMP,
    SampleMeasurement,
    SampleMP,
    Shots,
    VarianceMP,
)
from pennylane.transforms import (
    broadcast_expand,
    defer_measurements,
    diagonalize_measurements,
    dynamic_one_shot,
    split_non_commuting,
)
from pennylane.transforms.core import CompilePipeline, transform

from mqt.core.qdmi import Device as QDMIDeviceHandle
from mqt.core.qdmi import Job as QDMIJobHandle
from mqt.core.qdmi import PayloadDescriptor, ProgramFormat
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
    from collections.abc import Callable, Hashable, Mapping, Sequence

    from pennylane.tape import QuantumScript, QuantumScriptOrBatch
    from pennylane.typing import Result, ResultBatch

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


@transform
def _dynamic_or_samples(
    tape: QuantumScript, postselect_mode: str | None = None
) -> tuple[Sequence[QuantumScript], Callable[[Sequence[Result]], Result]]:
    """Use dynamic one-shot only for tapes that contain mid-circuit measurements.

    Returns:
        The transformed tapes and their result postprocessor.
    """
    if any(isinstance(operation, qp.ops.MidMeasure) for operation in tape.operations):
        return dynamic_one_shot(tape, postselect_mode=postselect_mode)
    return measurements_from_samples(tape)


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


class QDMIDevice(Device):
    """Execute PennyLane programs on a gate-based QDMI device.

    Args:
        device_id: Stable ID from the QDMI device registry.
        wires: PennyLane wire labels or number of wires. By default all QDMI
            qubits are exposed as consecutive integer wires.
        shots: Finite default shot configuration.
        session_parameters: QDMI device-session keyword arguments.
        job_parameters: QDMI custom job keyword arguments.
    """

    def __init__(
        self,
        device_id: str,
        wires: int | Sequence[Hashable] | None = None,
        shots: int | Sequence[int | tuple[int, int]] | Shots | None = 1024,
        *,
        session_parameters: QDMISessionParameters | None = None,
        job_parameters: QDMIJobParameters | None = None,
    ) -> None:
        """Initialize and open a fresh QDMI device session.

        Raises:
            PennyLaneConfigurationError: If configuration or requested wires are invalid.
        """
        self._device_id = device_id
        self._session_parameters = dict(session_parameters or {})
        self._job_parameters = dict(job_parameters or {})
        _validate_parameter_names(self._session_parameters, _SESSION_PARAMETERS, "session")
        _validate_parameter_names(self._job_parameters, _JOB_PARAMETERS, "job")

        try:
            self._qdmi_device = open_device(device_id, **self._session_parameters)
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
        reported = self._qdmi_device.try_program_features(self._program_format)
        self._program_capabilities = frozenset(feature.id for feature in reported or ())
        operations = {operation.name().lower() for operation in self._qdmi_device.operations()}
        mcm = {
            "mid-circuit-measurement",
            "measured-qubit-reuse",
            "measurement-result-use",
            "boolean-computation",
            "forward-branching",
        }
        supports_one_shot = (
            reported is not None
            and self._program_format == ProgramFormat.QASM3
            and mcm <= self._program_capabilities
            and {"measure", "reset"} <= operations
        )
        self._capabilities = DeviceCapabilities(
            qjit_compatible=False,
            runtime_code_generation=False,
            dynamic_qubit_management=False,
            supported_mcm_methods=["one-shot"] if supports_one_shot else [],
        )
        self._converter = _ProgramConverter(
            self._qdmi_device, self.wires, self._program_format, self._program_capabilities
        )
        self._submitted_jobs = 0
        self._execution_time = 0.0

    @property
    def capabilities(self) -> DeviceCapabilities:
        """Capabilities projected from the selected QDMI payload."""
        return self._capabilities

    @property
    def device_id(self) -> str:
        """Stable QDMI device ID."""
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

    def _select_program_format(self) -> PayloadDescriptor:
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
        msg = f"QDMI device '{self._device_id}' advertises neither OpenQASM 3 nor OpenQASM 2."
        raise UnsupportedFormatError(msg)

    def preprocess_transforms(self, execution_config: ExecutionConfig | None = None) -> CompilePipeline:
        """Build the PennyLane preprocessing pipeline for sampled QDMI execution.

        Returns:
            The transforms applied before device execution.
        """
        config = execution_config or ExecutionConfig()
        mcm_config = MCMConfig(**config.mcm_config) if isinstance(config.mcm_config, dict) else config.mcm_config
        one_shot = mcm_config.mcm_method == "one-shot"
        pipeline = CompilePipeline()
        pipeline.add_transform(_validate_finite_shots)
        pipeline.add_transform(validate_device_wires, self.wires, name=self.name)
        pipeline.add_transform(
            validate_measurements,
            analytic_measurements=lambda _measurement: False,
            sample_measurements=lambda measurement: isinstance(measurement, _SAMPLED_MEASUREMENTS),
            name=self.name,
        )
        pipeline.add_transform(split_non_commuting, grouping_strategy="qwc")
        if one_shot:
            pipeline.add_transform(_dynamic_or_samples, postselect_mode=mcm_config.postselect_mode)
            pipeline.add_transform(diagonalize_measurements)
        else:
            pipeline.add_transform(defer_measurements, allow_postselect=False, num_wires=len(self.wires))
            pipeline.add_transform(measurements_from_samples)
        pipeline.add_transform(
            decompose,
            stopping_condition=self._converter.supports,
            stopping_condition_shots=self._converter.supports,
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

        rows: list[list[int]] = []
        width = converted.output_width
        for bitstring in bitstrings:
            clean = bitstring.replace(" ", "")
            if len(clean) != width or any(bit not in "01" for bit in clean):
                msg = f"QDMI returned an invalid {width}-wire shot: {bitstring!r}."
                raise ExecutionError(msg)
            # QDMI bit strings use the conventional basis-state spelling with
            # the highest-index site on the left. PennyLane sample columns use
            # the declared wire order, starting with wire zero.
            wire_order = clean[::-1]
            rows.append([int(bit) for bit in wire_order])
        return np.asarray(rows, dtype=np.int8)

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
        """Submit and wait for one QDMI job.

        Returns:
            The successfully completed job.

        Raises:
            PennyLaneExecutionError: If submission, waiting, or execution fails.
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
        converted = self._converter.convert(tape)
        results: list[Any] = []
        for shots in self._shot_copies(tape.shots):
            started = monotonic()
            try:
                samples = self._samples(self._submit(converted, shots), converted, shots)
                if converted.mcm_slot_by_uid:
                    terminal = samples[:, : len(self.wires)]
                    measurement_results = []
                    for measurement in tape.measurements:
                        if isinstance(measurement.mv, MeasurementValue):
                            source = measurement.mv.measurements[0]
                            uid = source.meas_uid
                            assert uid is not None
                            measurement_results.append(samples[0, converted.mcm_slot_by_uid[uid]])
                        else:
                            measurement_results.append(
                                cast("SampleMeasurement", measurement).process_samples(terminal, self.wires)
                            )
                    results.append(tuple(measurement_results))
                else:
                    results.append(samples[:, converted.measurement_order])
            finally:
                self._execution_time += monotonic() - started

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
    """PennyLane entry point for MQT Core's local DDSIM QDMI device."""

    def __init__(
        self,
        wires: int | Sequence[Hashable] | None = None,
        shots: int | Sequence[int | tuple[int, int]] | Shots | None = 1024,
        *,
        session_parameters: QDMISessionParameters | None = None,
        job_parameters: QDMIJobParameters | None = None,
    ) -> None:
        """Open the built-in DDSIM device by its stable QDMI ID."""
        super().__init__(
            "mqt.ddsim.default",
            wires=wires,
            shots=shots,
            session_parameters=session_parameters,
            job_parameters=job_parameters,
        )
