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
    "pennylane",
    "pennylane.measurements",
}

from contextlib import contextmanager
from time import monotonic
from typing import TYPE_CHECKING, cast

import numpy as np
from pennylane import math
from pennylane.measurements import CountsMP, ExpectationMP, ProbabilityMP, VarianceMP

from mqt.core.mlir._pennylane import aggregate, samples
from mqt.core.plugins.pennylane.exceptions import PennyLaneExecutionError
from mqt.core.plugins.qdmi_batch import Batch, BatchEntry, JobAttempt
from mqt.core.qdmi import Job

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

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
        submit_programs: Callable[[Sequence[int], Sequence[CompiledProgram], int], Job | None],
    ) -> None:
        """Snapshot prepared programs and shot partitions before the first submission."""
        self._device = device
        self._submit_group = submit_programs
        self._tapes = tuple(tapes)
        self._wires = tuple(device._tape_wires(tape) for tape in tapes)  # ruff:ignore[private-member-access] Match the compiled tape wire order.
        self._single = single
        self._prepared = tuple(
            (program, shots, len(wires), tape_index)
            for tape_index, ((program, copies), wires) in enumerate(zip(prepared, self._wires, strict=True))
            for shots in copies
        )
        self._batch: Batch[object] = Batch(
            [
                BatchEntry(
                    index,
                    copy,
                    attempts=(
                        JobAttempt(
                            status=Job.Status.DONE,
                            result=self._process_samples(
                                index, np.zeros((shots, len(self._wires[index])), dtype=np.int8)
                            ),
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
    def entries(self) -> tuple[BatchEntry[object], ...]:
        """Ordered attempts with completed per-tape measurement results."""
        return self._batch.entries

    def _program(self, index: int) -> CompiledProgram:
        program = self._prepared[index][0]
        assert program is not None
        return program

    def _submit(self, index: int) -> Job:
        job = self._submit_programs([index])
        if job is None:
            msg = "Device declined the compiled program before submission."
            raise RuntimeError(msg)
        return job

    def _submit_programs(self, indices: Sequence[int]) -> Job | None:
        _, shots, _, _ = self._prepared[indices[0]]
        return self._submit_group(indices, [self._program(index) for index in indices], shots)

    def _process_samples(self, tape_index: int, values: np.ndarray) -> object:
        measured = tuple(
            cast("SampleMeasurement", measurement).process_samples(values, self._wires[tape_index])
            for measurement in self._tapes[tape_index].measurements
        )
        return measured[0] if len(measured) == 1 else measured

    def _readout(self, index: int, job: Job, program_index: int) -> object:
        _, shots, width, tape_index = self._prepared[index]
        tape = self._tapes[tape_index]
        if not all(
            type(m) in {CountsMP, ProbabilityMP} or (type(m) in {ExpectationMP, VarianceMP} and bool(m.wires))
            for m in tape.measurements
        ):
            return self._process_samples(
                tape_index, samples(job, width, shots, program_index, self._program(index).result_layout)
            )
        wires = self._wires[tape_index]
        selected = [measurement.wires or wires for measurement in tape.measurements]
        marginals = aggregate(
            job,
            width,
            shots,
            [
                ([wires.index(wire) for wire in selection], type(measurement) is CountsMP)
                for measurement, selection in zip(tape.measurements, selected, strict=True)
            ],
            program_index,
            self._program(index).result_layout,
        )
        measured = []
        for measurement, selection, value in zip(tape.measurements, selected, marginals, strict=True):
            if type(measurement) is CountsMP:
                processor = (
                    measurement
                    if measurement.wires
                    else CountsMP(wires=selection, all_outcomes=measurement.all_outcomes)
                )
                measured.append(processor.process_counts(cast("dict[str, int]", value), selection))
            elif type(measurement) is ProbabilityMP:
                measured.append(value)
            else:
                eigvals = measurement.eigvals()
                if type(measurement) is VarianceMP:
                    eigvals = math.asarray(eigvals, dtype="float64")
                mean = math.dot(value, eigvals)
                measured.append(mean if type(measurement) is ExpectationMP else math.dot(value, eigvals**2) - mean**2)
        return measured[0] if len(measured) == 1 else tuple(measured)

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

    def collect(self) -> tuple[BatchEntry[object], ...]:
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
                tape_results[entry.input_index].append(entry.result)
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
