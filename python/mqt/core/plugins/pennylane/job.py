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
    "mqt.core.mlir._pennylane",
    "mqt.core.plugins.pennylane.exceptions",
    "mqt.core.plugins.qdmi_batch",
    "mqt.core.qdmi",
    "numpy",
    "pennylane.measurements",
}

from contextlib import contextmanager
from time import monotonic
from typing import TYPE_CHECKING, cast

import numpy as np
from pennylane.measurements import CountsMP, ExpectationMP, ProbabilityMP, VarianceMP

from mqt.core.mlir._pennylane import counts, samples
from mqt.core.plugins.pennylane.exceptions import PennyLaneExecutionError
from mqt.core.plugins.qdmi_batch import Batch, BatchEntry, JobAttempt
from mqt.core.qdmi import Job

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from pennylane.measurements import SampleMeasurement
    from pennylane.tape import QuantumScript
    from pennylane.typing import Result, ResultBatch

    from mqt.core.mlir import CompiledProgram
    from mqt.core.plugins.pennylane.device import QDMIDevice

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
        prepared: Sequence[tuple[CompiledProgram | None, tuple[int, ...]]],
        tapes: Sequence[QuantumScript],
        *,
        single: bool,
        max_retries: int,
    ) -> None:
        """Snapshot prepared programs and shot partitions before the first submission."""
        self._device = device
        self._tapes = tuple(tapes)
        self._wires = tuple(device._tape_wires(tape) for tape in tapes)  # ruff:ignore[private-member-access] Match the compiled tape wire order.
        self._single = single
        self._prepared = tuple(
            (program, shots, len(wires), tape_index)
            for tape_index, ((program, copies), wires) in enumerate(zip(prepared, self._wires, strict=True))
            for shots in copies
        )
        self._batch: Batch[np.ndarray | dict[str, int]] = Batch(
            [
                BatchEntry(
                    index,
                    copy,
                    attempts=(
                        JobAttempt(
                            status=Job.Status.DONE, result=np.zeros((shots, len(self._wires[index])), dtype=np.int8)
                        ),
                    )
                    if program is None
                    else (),
                )
                for index, (program, copies) in enumerate(prepared)
                for copy, shots in enumerate(copies)
            ],
            submit=self._submit,
            decode=self._readout,
            submit_programs=self._submit_programs,
            group_by=[(program.program_format if program else None, shots) for program, shots, _, _ in self._prepared],
            submission_error=lambda msg: PennyLaneExecutionError(msg, job=self),
            execution_error=lambda msg: PennyLaneExecutionError(msg, job=self),
            max_retries=max_retries,
            on_submit=self._record_submission,
        )

    @property
    def entries(self) -> tuple[BatchEntry[np.ndarray | dict[str, int]], ...]:
        """Ordered attempts with count dictionaries or shot-by-wire sample arrays."""
        return self._batch.entries

    def _program(self, index: int) -> CompiledProgram:
        program = self._prepared[index][0]
        assert program is not None
        return program

    def _submit(self, index: int) -> Job:
        _, shots, _, _ = self._prepared[index]
        return self._device._context.submit(self._program(index), shots)  # ruff:ignore[private-member-access] Retained session and target snapshot.

    def _submit_programs(self, indices: Sequence[int]) -> Job | None:
        _, shots, _, _ = self._prepared[indices[0]]
        return self._device._context.try_submit(  # ruff:ignore[private-member-access] Retained session and target snapshot.
            [self._program(index) for index in indices], shots
        )

    def _samples(self, index: int, job: Job, program_index: int) -> np.ndarray:
        """Return a shot-by-wire sample array in the device's logical wire order."""
        _, shots, width, _ = self._prepared[index]
        return samples(job, width, shots, program_index, self._program(index).result_layout)

    def _readout(self, index: int, job: Job, program_index: int) -> np.ndarray | dict[str, int]:
        tape = self._tapes[self._prepared[index][3]]
        if all(
            type(m) in {CountsMP, ProbabilityMP} or (type(m) in {ExpectationMP, VarianceMP} and bool(m.wires))
            for m in tape.measurements
        ):
            _, shots, width, _ = self._prepared[index]
            return counts(job, width, shots, program_index, self._program(index).result_layout)
        return self._samples(index, job, program_index)

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

    def collect(self) -> tuple[BatchEntry[np.ndarray | dict[str, int]], ...]:
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
                wires = self._wires[entry.input_index]
                measured = []
                for measurement in measurements:
                    if isinstance(entry.result, dict):
                        processor = measurement
                        if isinstance(measurement, CountsMP) and not measurement.wires:
                            processor = CountsMP(wires=wires, all_outcomes=measurement.all_outcomes)
                        elif isinstance(measurement, ProbabilityMP) and not measurement.wires:
                            processor = ProbabilityMP(wires=wires)
                        measured.append(cast("SampleMeasurement", processor).process_counts(entry.result, wires))
                    else:
                        measured.append(cast("SampleMeasurement", measurement).process_samples(entry.result, wires))
                tape_results[entry.input_index].append(measured[0] if len(measured) == 1 else tuple(measured))
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
