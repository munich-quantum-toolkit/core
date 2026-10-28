# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""PennyLane execution on gate-based QDMI devices."""

from __future__ import annotations

__lazy_modules__ = {
    "importlib.metadata",
    "mqt.core",
    "mqt.core.mlir",
    "mqt.core.mlir._pennylane",
    "mqt.core.plugins.pennylane.exceptions",
    "mqt.core.plugins.pennylane.job",
    "mqt.core.qdmi",
    "mqt.core.qdmi.builtin_driver",
    "numbers",
    "operator",
    "packaging.specifiers",
    "pennylane.transforms",
    "pennylane.transforms.core",
}

import operator
from importlib.metadata import version
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pennylane as qp
from packaging.specifiers import SpecifierSet
from pennylane.devices import Device, ExecutionConfig
from pennylane.transforms import defer_measurements
from pennylane.transforms.core import CompilePipeline

from mqt.core import mlir
from mqt.core.mlir import QCProgram, SerializationProfile, compile_program
from mqt.core.mlir._pennylane import Context
from mqt.core.plugins.pennylane.exceptions import PennyLaneConfigurationError as ConfigurationError
from mqt.core.plugins.pennylane.exceptions import PennyLaneTranslationError as TranslationError
from mqt.core.plugins.pennylane.exceptions import PennyLaneValidationError as ValidationError
from mqt.core.plugins.pennylane.job import PennyLaneJob
from mqt.core.qdmi import ProgramFormat
from mqt.core.qdmi.builtin_driver import open_device

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    from pennylane.measurements import Shots
    from pennylane.tape import QuantumScript, QuantumScriptOrBatch
    from pennylane.typing import Result, ResultBatch
    from pennylane.wires import Wires

    from mqt.core.mlir import CompiledProgram
    from mqt.core.qdmi import Device as QDMIDeviceHandle
    from mqt.core.typing import QDMIJobParameters, QDMISessionParameters

__all__ = ["DDSIMDevice", "QDMIDevice"]

_SESSION_PARAMETERS = frozenset({
    "driver_path",
    "token",
    "auth_file",
    "auth_url",
    "username",
    "password",
    "base_url",
    "device_config",
    "device_config_file",
    "custom1",
    "custom2",
    "custom3",
    "custom4",
    "custom5",
})
_JOB_PARAMETERS = frozenset({"custom1", "custom2", "custom3", "custom4", "custom5"})


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
        device_id: Stable ID reported by the QDMI driver. Use either this
            argument or ``device``.
        wires: PennyLane wire labels or number of wires. By default all QDMI
            qubits are exposed as consecutive integer wires.
        device: An already-open QDMI device.
        session_parameters: Device-session parameters for the MQT Core QDMI driver.
        job_parameters: QDMI custom job keyword arguments.
        max_retries: Maximum automatic replacements per failed entry; disabled by default.
        profile: Explicit receiver syntax and readout requirements for compilation.
            Nondefault profiles select OpenQASM 3.
    """

    config_filepath = str(Path(__file__).with_name("capabilities.toml"))

    @staticmethod
    def get_c_interface() -> tuple[str, str]:
        """Return the Catalyst device factory name and library path.

        Raises:
            PennyLaneConfigurationError: If Catalyst uses an unsupported device ABI.
        """
        installed = version("pennylane-catalyst")
        supported = SpecifierSet(">=0.15,<0.16.0a0")
        if not supported.contains(installed, prereleases=True):
            msg = f"Catalyst {installed} has an unsupported device ABI; requires {supported}."
            raise ConfigurationError(msg)
        return "MQTQDMI", mlir.__file__

    def __init__(
        self,
        device_id: str | None = None,
        wires: int | Sequence[Hashable] | None = None,
        *,
        device: QDMIDeviceHandle | None = None,
        session_parameters: QDMISessionParameters | None = None,
        job_parameters: QDMIJobParameters | None = None,
        max_retries: int = 0,
        profile: SerializationProfile | None = None,
    ) -> None:
        """Initialize from a stable ID or an open QDMI device.

        Raises:
            PennyLaneConfigurationError: If configuration or requested wires are invalid.
        """
        if isinstance(max_retries, bool) or not isinstance(max_retries, Integral) or max_retries < 0:
            msg = f"max_retries must be a nonnegative integer, got {max_retries!r}."
            raise ConfigurationError(msg)
        self._max_retries = int(max_retries)
        self._profile = SerializationProfile.DEFAULT if profile is None else profile
        self.last_job: PennyLaneJob | None = None
        session_parameters = session_parameters.copy() if session_parameters else {}
        job_parameters = job_parameters.copy() if job_parameters else {}
        _validate_parameter_names(session_parameters, _SESSION_PARAMETERS, "session")
        _validate_parameter_names(job_parameters, _JOB_PARAMETERS, "job")

        if (device_id is None) == (device is None):
            msg = "Specify exactly one of device_id and device."
            raise ConfigurationError(msg)
        if device is not None:
            if session_parameters:
                msg = "session_parameters cannot be used with an already-open QDMI device."
                raise ConfigurationError(msg)
            self._qdmi_device = device
        else:
            assert device_id is not None
            try:
                self._qdmi_device = open_device(device_id, **session_parameters)
            except (IndexError, RuntimeError, ValueError) as exc:
                msg = f"Failed to open QDMI device '{device_id}': {exc}"
                raise ConfigurationError(msg) from exc
        self._device_id = device_id
        device_name = device_id or self._qdmi_device.name()

        num_qubits = self._qdmi_device.qubits_num()
        resolved_wires: int | Sequence[Hashable] = num_qubits if wires is None else wires
        requested_wires = resolved_wires if isinstance(resolved_wires, int) else len(resolved_wires)
        if requested_wires <= 0:
            msg = "A QDMI PennyLane device requires at least one wire."
            raise ConfigurationError(msg)
        if requested_wires > num_qubits:
            msg = (
                f"QDMI device '{device_name}' exposes {num_qubits} qubits, but {requested_wires} wires were requested."
            )
            raise ConfigurationError(msg)

        super().__init__(wires=resolved_wires)
        try:
            self._context = Context(self._qdmi_device, profile=self._profile, **job_parameters)
        except (RuntimeError, ValueError) as exc:
            raise ConfigurationError(str(exc)) from exc
        self._job_parameters = job_parameters
        self.device_kwargs = {"context": self._context.id}
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

    def preprocess_transforms(self, execution_config: ExecutionConfig | None = None) -> CompilePipeline:
        """Return transforms for finite-shot execution and deferred measurements."""
        pipeline = CompilePipeline()
        pipeline.add_transform(_validate_finite_shots)
        pipeline.add_transform(_defer_on_device_wires, self.wires)
        pipeline += super().preprocess_transforms(execution_config)
        return pipeline

    def _prepare(self, tapes: Sequence[QuantumScript]) -> list[CompiledProgram]:
        """Compile preprocessed tapes using one snapshot of the QDMI target.

        Returns:
            Programs ready for submission to this device, in input order.

        Raises:
            PennyLaneTranslationError: If an operation or target cannot be compiled.
        """
        try:
            return compile_program(
                [QCProgram.from_pennylane(tape, wires=self.wires) for tape in tapes],
                target=self._qdmi_device,
                profile=self._profile,
                program_format=ProgramFormat.QASM3 if self._profile != SerializationProfile.DEFAULT else None,
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise TranslationError(str(exc)) from exc

    @staticmethod
    def _shot_copies(shots: Shots) -> tuple[int, ...]:
        """Return one shot count for each shot-vector copy.

        Raises:
            PennyLaneValidationError: If execution is analytic.
        """
        if not shots:
            msg = "QDMI devices require a finite number of shots."
            raise ValidationError(msg)
        return tuple(shot_copy.shots for shot_copy in shots.shot_vector for _ in range(shot_copy.copies))

    def execute(
        self,
        circuits: QuantumScriptOrBatch,
        execution_config: ExecutionConfig | None = None,
    ) -> Result | ResultBatch:
        """Submit a batch to QDMI before collecting its ordered results.

        Returns:
            One result for every preprocessed input tape.
        """
        self.last_job = None
        del execution_config
        single = isinstance(circuits, qp.tape.QuantumScript)
        tapes = (circuits,) if single else tuple(circuits)
        shot_copies = tuple(self._shot_copies(tape.shots) for tape in tapes)
        if not tapes:
            return cast("ResultBatch", ())
        prepared = tuple(zip(self._prepare(tapes), shot_copies, strict=True))

        if self.tracker.active:
            self.tracker.update(batches=1, batch_len=len(tapes))
            self.tracker.record()

        job = PennyLaneJob(
            self,
            prepared,
            tapes,
            single=single,
            max_retries=self._max_retries,
        )
        self.last_job = job
        job.submit()
        return job.result()


class DDSIMDevice(QDMIDevice):
    """PennyLane entry point for MQT Core's local DDSIM QDMI device."""

    def __init__(
        self,
        wires: int | Sequence[Hashable] | None = None,
        *,
        session_parameters: QDMISessionParameters | None = None,
        job_parameters: QDMIJobParameters | None = None,
        max_retries: int = 0,
    ) -> None:
        """Open the built-in DDSIM device by its stable QDMI ID."""
        super().__init__(
            "mqt.ddsim.default",
            wires=wires,
            session_parameters=session_parameters,
            job_parameters=job_parameters,
            max_retries=max_retries,
        )
