# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""In-process recovery of a preprocessed PennyLane execution batch."""

from __future__ import annotations

# ruff: file-ignore[private-member-access] Companion handle owns the device execution bookkeeping.
from time import monotonic
from typing import TYPE_CHECKING, cast

from ..qdmi_batch import BatchEntry, _Batch
from .exceptions import PennyLaneExecutionError

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from pennylane.typing import Result, ResultBatch

    from mqt.core.qdmi import Job

    from .converter import _ConvertedProgram
    from .device import QDMIDevice

__all__ = ["PennyLaneJob"]


class PennyLaneJob:
    """Retain results and attempts for one preprocessed device batch.

    Obtain this handle through ``device.last_job`` or an execution error's
    ``job`` attribute. Results have the device's preprocessed tape shape; they
    do not reconstruct an interrupted QNode or gradient calculation.
    """

    def __init__(
        self,
        device: QDMIDevice,
        prepared: Sequence[tuple[_ConvertedProgram, tuple[int, ...]]],
        partitioned: tuple[bool, ...],
        *,
        single: bool,
    ) -> None:
        """Snapshot prepared programs and shot partitions before the first submission."""
        self._device = device
        self._partitioned = partitioned
        self._single = single
        self._prepared = tuple((program, shots) for program, copies in prepared for shots in copies)
        self._parameters = device._job_parameters.copy()
        self._batch: _Batch[np.ndarray] = _Batch(
            [BatchEntry(index, copy) for index, (_, copies) in enumerate(prepared) for copy in range(len(copies))],
            submit=self._submit,
            decode=lambda index, job: device._samples(job, *self._prepared[index]),
            submission_error=lambda msg: PennyLaneExecutionError(msg, job=self),
            execution_error=lambda msg: PennyLaneExecutionError(msg, job=self),
            max_retries=device._max_retries,
            on_submit=self._record_submission,
        )

    @property
    def entries(self) -> tuple[BatchEntry[np.ndarray], ...]:
        """Ordered snapshots of inputs, submission attempts, results, and failures."""
        return self._batch.entries

    def _submit(self, index: int) -> Job:
        converted, shots = self._prepared[index]
        return self._device.qdmi_device.submit_job(
            converted.payload, converted.program_format, shots, **self._parameters
        )

    def _record_submission(self, index: int) -> None:
        _, shots = self._prepared[index]
        self._device._submitted_jobs += 1
        if self._device.tracker.active:
            self._device.tracker.update(executions=1, shots=shots)
            self._device.tracker.record()

    def collect(self) -> tuple[BatchEntry[np.ndarray], ...]:
        """Read existing jobs without replacement executions or aggregate errors.

        Returns:
            Entry snapshots; successful results are cached.
        """
        started = monotonic()
        try:
            return self._batch.collect()
        finally:
            self._device._execution_time += monotonic() - started

    def result(self) -> Result | ResultBatch:
        """Collect, retry confirmed failures, and assemble the device's usual output.

        Returns:
            Raw samples for the preprocessed input tapes.

        Raises:
            PennyLaneExecutionError: If submission fails or entries remain unsuccessful.
        """  # ruff:ignore[docstring-extraneous-exception] The shared collector raises the adapter error.
        started = monotonic()
        try:
            self._batch.complete()
            tape_results: list[list[np.ndarray]] = [[] for _ in self._partitioned]
            for entry in self.entries:
                assert entry.result is not None
                tape_results[entry.input_index].append(entry.result)
            results = tuple(
                tuple(samples) if partitioned else samples[0]
                for partitioned, samples in zip(self._partitioned, tape_results, strict=True)
            )
            return cast("Result | ResultBatch", results[0] if self._single else results)
        finally:
            self._device._execution_time += monotonic() - started

    def resubmit(self, indices: Sequence[int], *, allow_unknown: bool = False) -> PennyLaneJob:
        """Explicitly submit selected entries; unknown outcomes may duplicate work.

        Returns:
            This batch handle, with previous attempts retained.
        """
        started = monotonic()
        try:
            self._batch.resubmit(indices, allow_unknown=allow_unknown)
        finally:
            self._device._execution_time += monotonic() - started
        return self

    def cancel(self) -> bool:
        """Disable automatic retries and explicitly cancel known attempts.

        Returns:
            Whether all cancellation requests succeeded without uncertain admissions.
        """
        return self._batch.cancel()
