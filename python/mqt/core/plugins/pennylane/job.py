# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""In-process recovery of a preprocessed PennyLane execution batch."""

from __future__ import annotations

__lazy_modules__ = {"mqt.core.plugins.pennylane.exceptions", "mqt.core.plugins.qdmi_batch", "numpy"}

from contextlib import contextmanager
from time import monotonic
from typing import TYPE_CHECKING, cast

import numpy as np

from mqt.core.plugins.pennylane.exceptions import PennyLaneExecutionError
from mqt.core.plugins.qdmi_batch import Batch, BatchEntry

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from pennylane.typing import Result, ResultBatch

    from mqt.core.plugins.pennylane.converter import _ConvertedProgram
    from mqt.core.plugins.pennylane.device import QDMIDevice
    from mqt.core.qdmi import Job
    from mqt.core.typing import QDMIJobParameters

__all__ = ["PennyLaneJob"]


class PennyLaneJob:
    """Retain results and attempts for one preprocessed device batch.

    Obtain this handle through ``device.last_job`` or an execution error's
    ``job`` attribute. Results have the device's preprocessed tape shape; they
    do not reconstruct an interrupted QNode or gradient calculation. Automatic
    retries use the device's setting captured at creation and its lifetime limit
    per failed entry. Cancelled or uncertain jobs are never retried.
    """

    def __init__(
        self,
        device: QDMIDevice,
        prepared: Sequence[tuple[_ConvertedProgram, tuple[int, ...]]],
        partitioned: tuple[bool, ...],
        *,
        single: bool,
        parameters: QDMIJobParameters,
        max_retries: int,
    ) -> None:
        """Snapshot prepared programs and shot partitions before the first submission."""
        self._device = device
        self._partitioned = partitioned
        self._single = single
        self._prepared = tuple((program, shots) for program, copies in prepared for shots in copies)
        self._parameters = parameters.copy()
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
        converted, shots = self._prepared[index]
        return self._device.qdmi_device.submit_job(
            converted.payload, converted.program_format, shots, **self._parameters
        )

    def _submit_programs(self, indices: Sequence[int]) -> Job | None:
        converted, shots = self._prepared[indices[0]]
        return self._device.qdmi_device.try_submit_job(
            [self._prepared[index][0].payload for index in indices],
            converted.program_format,
            shots,
            **self._parameters,
        )

    @staticmethod
    def _shots_or_counts(job: Job, program_index: int = 0) -> tuple[list[str], list[int] | None]:
        """Read individual shots or distinct outcomes with their multiplicities.

        Returns:
            Bit strings and optional repetition counts for histogram results.

        Raises:
            PennyLaneExecutionError: If the job exposes neither result representation.
        """
        shots_error = None
        try:
            if shots := job.get_shots(program_index):
                return shots, None
        except RuntimeError as exc:
            shots_error = exc

        try:
            counts = job.get_counts(program_index)
        except RuntimeError as exc:
            msg = f"Could not read QDMI samples: shots: {shots_error}; counts: {exc}"
            causes = [cause for cause in (shots_error, exc) if cause is not None]
            raise PennyLaneExecutionError(msg) from ExceptionGroup("QDMI result retrieval failed", causes)
        bitstrings = sorted(bitstring for bitstring, count in counts.items() if count)
        return bitstrings, [counts[bitstring] for bitstring in bitstrings]

    def _samples(self, index: int, job: Job, program_index: int) -> np.ndarray:
        """Convert QDMI bit strings to PennyLane sample rows.

        Returns:
            A shot-by-wire array in PennyLane measurement order.

        Raises:
            PennyLaneExecutionError: If QDMI returns malformed or incomplete results.
        """
        converted, shots = self._prepared[index]
        bitstrings, repeats = self._shots_or_counts(job, program_index)
        num_samples = len(bitstrings) if repeats is None else sum(repeats)
        if num_samples != shots:
            msg = f"QDMI returned {num_samples} samples for a {shots}-shot job."
            raise PennyLaneExecutionError(msg)

        width = len(converted.wire_map)
        cleaned: list[str] = []
        for bitstring in bitstrings:
            clean = bitstring.replace(" ", "")
            if len(clean) != width or clean.strip("01"):
                msg = f"QDMI returned an invalid {width}-wire shot: {bitstring!r}."
                raise PennyLaneExecutionError(msg)
            cleaned.append(clean)
        if not bitstrings:
            return np.asarray([], dtype=np.int8)
        packed = np.frombuffer("".join(cleaned).encode("ascii"), dtype=np.int8).reshape(len(bitstrings), width)
        # QDMI spells the highest-index site first; PennyLane starts with wire zero.
        samples = packed[:, ::-1][:, converted.measurement_order] - ord("0")
        return samples if repeats is None else np.repeat(samples, repeats, axis=0)

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
            Raw samples for the preprocessed input tapes.
        """
        with self._record_time():
            self._batch.complete()
            tape_results: list[list[np.ndarray]] = [[] for _ in self._partitioned]
            for entry in self.entries:
                assert entry.result is not None
                tape_results[entry.input_index].append(entry.result)
            results = tuple(
                tuple(samples) if partitioned else samples[0]
                for partitioned, samples in zip(self._partitioned, tape_results, strict=True)
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
