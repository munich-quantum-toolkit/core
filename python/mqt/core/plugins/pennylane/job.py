# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""In-process recovery of a preprocessed PennyLane execution batch."""

from __future__ import annotations

__lazy_modules__ = {
    "mqt.core.mlir",
    "mqt.core.mlir._pennylane",
    "mqt.core.plugins.pennylane.exceptions",
    "mqt.core.plugins.qdmi_batch",
}

from contextlib import contextmanager
from time import monotonic
from typing import TYPE_CHECKING, cast

from mqt.core.mlir import submit_program, try_submit_programs
from mqt.core.mlir._pennylane import samples
from mqt.core.plugins.pennylane.exceptions import PennyLaneExecutionError
from mqt.core.plugins.qdmi_batch import Batch, BatchEntry

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    import numpy as np
    from pennylane.measurements import SampleMeasurement
    from pennylane.tape import QuantumScript
    from pennylane.typing import Result, ResultBatch

    from mqt.core.mlir import CompiledProgram
    from mqt.core.plugins.pennylane.device import QDMIDevice
    from mqt.core.qdmi import Job

__all__ = ["PennyLaneJob"]


class PennyLaneJob:
    """Retain results and attempts for one preprocessed device batch.

    Obtain this handle through ``device.last_job`` or an execution error's
    ``job`` attribute. Results have the device's preprocessed tape shape; they
    do not reconstruct an interrupted QNode or gradient calculation. Automatic
    retries use the device's setting at creation, applied over each entry's
    lifetime. Cancelled or uncertain jobs are never retried automatically.
    """

    def __init__(
        self,
        device: QDMIDevice,
        prepared: Sequence[tuple[CompiledProgram, tuple[int, ...]]],
        tapes: Sequence[QuantumScript],
        *,
        single: bool,
        max_retries: int,
    ) -> None:
        """Snapshot prepared programs and shot partitions before the first submission."""
        self._device = device
        self._tapes = tuple(tapes)
        self._single = single
        self._prepared = tuple((program, shots) for program, copies in prepared for shots in copies)
        self._batch: Batch[np.ndarray] = Batch(
            [BatchEntry(index, copy) for index, (_, copies) in enumerate(prepared) for copy in range(len(copies))],
            submit=self._submit,
            decode=self._samples,
            submit_programs=self._submit_programs,
            group_by=[(program.program_format, shots) for program, shots in self._prepared],
            submission_error=lambda msg: PennyLaneExecutionError(msg, job=self),
            execution_error=lambda msg: PennyLaneExecutionError(msg, job=self),
            max_retries=max_retries,
            on_submit=self._record_submission,
        )

    @property
    def entries(self) -> tuple[BatchEntry[np.ndarray], ...]:
        """Ordered snapshots of inputs, submission attempts, results, and failures."""
        return self._batch.entries

    def _submit(self, index: int) -> Job:
        program, shots = self._prepared[index]
        return submit_program(
            program,
            target=self._device.qdmi_device,
            num_shots=shots,
            **self._device._job_parameters,  # ruff:ignore[private-member-access] Retained device submission configuration.
        )

    def _submit_programs(self, indices: Sequence[int]) -> Job | None:
        _, shots = self._prepared[indices[0]]
        return try_submit_programs(
            [self._prepared[index][0] for index in indices],
            target=self._device.qdmi_device,
            num_shots=shots,
            **self._device._job_parameters,  # ruff:ignore[private-member-access] Retained device submission configuration.
        )

    def _samples(self, index: int, job: Job, program_index: int) -> np.ndarray:
        """Return a shot-by-wire sample array in the device's logical wire order."""
        program, shots = self._prepared[index]
        return samples(job, len(self._device.wires), shots, program_index, program.result_layout)

    def _record_submission(self, indices: Sequence[int]) -> None:
        self._device._submitted_jobs += 1  # ruff:ignore[private-member-access] Update the device's read-only total.
        if self._device.tracker.active:
            self._device.tracker.update(
                executions=len(indices), shots=sum(self._prepared[index][1] for index in indices)
            )
            self._device.tracker.record()

    @contextmanager
    def _record_time(self) -> Generator[None, None, None]:
        started = monotonic()
        try:
            yield
        finally:
            self._device._execution_time += monotonic() - started  # ruff:ignore[private-member-access] Include recovery in the device's read-only total.

    def submit(self, indices: Sequence[int] | None = None) -> None:
        """Submit selected untouched entries, or all remaining untouched entries.

        Previously attempted entries require :meth:`resubmit`.
        """
        with self._record_time():
            self._batch.submit(indices)

    def collect(self) -> tuple[BatchEntry[np.ndarray], ...]:
        """Read existing jobs without replacement executions or aggregate errors.

        Returns:
            Entry snapshots; successful results are cached.
        """
        with self._record_time():
            return self._batch.collect()

    def result(self) -> Result | ResultBatch:
        """Collect, retry confirmed failures, and assemble the device's usual output.

        Submission or collection failures propagate
        :class:`~mqt.core.plugins.pennylane.exceptions.PennyLaneExecutionError`
        with this batch handle.

        Returns:
            Measurement results for the preprocessed input tapes.
        """
        with self._record_time():
            self._batch.complete()
            tape_results: list[list[object]] = [[] for _ in self._tapes]
            for entry in self.entries:
                assert entry.result is not None
                measurements = self._tapes[entry.input_index].measurements
                measured = tuple(
                    cast("SampleMeasurement", measurement).process_samples(entry.result, self._device.wires)
                    for measurement in measurements
                )
                tape_results[entry.input_index].append(measured[0] if len(measured) == 1 else measured)
            results = tuple(
                tuple(samples) if tape.shots.has_partitioned_shots else samples[0]
                for tape, samples in zip(self._tapes, tape_results, strict=True)
            )
            if self._single:
                return cast("Result", results[0])
            return cast("ResultBatch", results)

    def resubmit(self, indices: Sequence[int], *, allow_unknown: bool = False) -> PennyLaneJob:
        """Replace selected failed or cancelled entries, retaining earlier attempts.

        Unknown outcomes require ``allow_unknown=True`` and may duplicate work.
        Running or completed jobs cannot be replaced. Use :meth:`submit` for
        entries that have never been submitted.

        Returns:
            This batch handle, with previous attempts retained.
        """
        with self._record_time():
            self._batch.resubmit(indices, allow_unknown=allow_unknown)
        return self

    def cancel(self) -> bool:
        """Disable automatic retries and explicitly cancel known attempts.

        Returns:
            Whether all cancellation requests succeeded without uncertain admissions.
        """
        return self._batch.cancel()
