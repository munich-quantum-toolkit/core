# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""QDMI Qiskit Job implementation.

Provides a Qiskit JobV1-compatible wrapper for QDMI job execution and results.
"""

from __future__ import annotations

import datetime
from collections import Counter
from copy import deepcopy
from numbers import Integral
from typing import TYPE_CHECKING, Any, cast

from qiskit.providers import JobError, JobStatus, JobV1
from qiskit.result import Result
from qiskit.result.models import ExperimentResult

from mqt.core.qdmi import Job as QDMIJobHandle

from ..qdmi_batch import Batch, BatchEntry, JobAttempt
from .exceptions import JobExecutionError, JobSubmissionError

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Self

    from qiskit.circuit import QuantumCircuit

    from mqt.core.qdmi import ProgramFormat

    from .backend import QDMIBackend

__all__ = ["QDMIJob"]


def __dir__() -> list[str]:
    return __all__


def _encode_bits(bits: str, width: int) -> str:
    """Validate a QDMI bitstring and encode it as Qiskit result data.

    Returns:
        The hexadecimal Qiskit memory value.

    Raises:
        JobError: If the bitstring has the wrong width or contains nonbinary digits.
    """
    if len(bits) != width or any(bit not in "01" for bit in bits):
        msg = f"Invalid QDMI bitstring {bits!r}: expected {width} binary digits in classical-bit order."
        raise JobError(msg)
    return hex(int(bits, 2)) if bits else "0x0"


class QDMIJob(JobV1):
    """Qiskit job wrapping one or more QDMI jobs.

    This class handles both single-circuit and multi-circuit execution,
    using native program lists where supported and independent jobs otherwise. Use
    :meth:`from_circuits` to prepare an unsubmitted batch. Wrapping already
    submitted jobs supports collection and cancellation, without replacements.

    Args:
        backend: The backend this job runs on.
        jobs: Submitted QDMI jobs in circuit order, or None to prepare a new batch.
        circuits: The executed circuits, used to snapshot result headers.
        shots: Requested shots per circuit.
        memory: Whether to collect genuine ordered shots.
        max_retries: Lifetime replacement limit per confirmed failed entry;
            disabled by default. Cancelled or uncertain jobs are never retried.
    """

    def __init__(
        self,
        backend: QDMIBackend,
        jobs: Sequence[QDMIJobHandle] | None,
        circuits: Sequence[QuantumCircuit],
        *,
        shots: int,
        memory: bool,
        max_retries: int = 0,
    ) -> None:
        """Initialize without querying remote job IDs.

        Raises:
            ValueError: If circuits are empty or the supplied jobs differ in length.
        """
        if not circuits or (jobs is not None and len(jobs) != len(circuits)):
            msg = "QDMIJob requires at least one circuit and one submitted job per circuit when jobs are supplied."
            raise ValueError(msg)
        super().__init__(backend=backend, job_id="")
        self._backend: QDMIBackend = backend
        self._headers: list[dict[str, Any]] = [
            {
                "name": circuit.name,
                "memory_slots": circuit.num_clbits,
                "creg_sizes": [[register.name, register.size] for register in circuit.cregs],
                "metadata": deepcopy(circuit.metadata),
            }
            for circuit in circuits
        ]
        self._shots = shots
        self._memory = memory
        self._result: Result | None = None
        self._programs: tuple[tuple[str | bytes, ProgramFormat], ...] | None = None
        if jobs is None:
            formats = backend.device.supported_program_formats()
            self._programs = tuple(
                backend._serialize_circuit(circuit, formats)  # ruff:ignore[private-member-access] Use the backend's serializer selection.
                for circuit in circuits
            )
        self._batch: Batch[ExperimentResult] = Batch(
            [
                BatchEntry(i, attempts=(JobAttempt(handle=jobs[i]),) if jobs is not None else ())
                for i in range(len(circuits))
            ],
            submit=self._submit_entry if self._programs is not None else None,
            decode=lambda index, handle, program_index: self._collect_result(
                handle, self._headers[index], program_index
            ),
            submit_programs=self._submit_programs,
            group_by=[program_format for _, program_format in self._programs] if self._programs is not None else None,
            submission_error=lambda msg: JobSubmissionError(msg, job=self),
            execution_error=lambda msg: JobExecutionError(msg, job=self),
            max_retries=max_retries,
        )

    @classmethod
    def from_circuits(
        cls,
        backend: QDMIBackend,
        circuits: Sequence[QuantumCircuit],
        *,
        shots: int,
        memory: bool,
        max_retries: int = 0,
    ) -> Self:
        """Prepare an unsubmitted batch from bound, backend-ready circuits.

        All circuits are serialized before this method returns. Use
        :meth:`~mqt.core.plugins.qiskit.backend.QDMIBackend.run` for normal execution, including circuit validation
        and parameter binding.

        Returns:
            A batch ready for :meth:`submit`, with programs retained for recovery.
        """
        return cls(backend, None, circuits, shots=shots, memory=memory, max_retries=max_retries)

    @property
    def entries(self) -> tuple[BatchEntry[ExperimentResult], ...]:
        """Ordered snapshots of inputs, submission attempts, results, and failures."""
        return self._batch.entries

    def _submit_entry(self, index: int) -> QDMIJobHandle:
        assert self._programs is not None
        program, program_format = self._programs[index]
        return self._backend.device.submit_job(program=program, program_format=program_format, num_shots=self._shots)

    def _submit_programs(self, indices: Sequence[int]) -> QDMIJobHandle | None:
        assert self._programs is not None
        return self._backend.device.try_submit_programs(
            # A format fixes the payload type for the whole group.
            cast("Sequence[str] | Sequence[bytes]", [self._programs[index][0] for index in indices]),
            self._programs[indices[0]][1],
            self._shots,
        )

    def collect(self) -> tuple[BatchEntry[ExperimentResult], ...]:
        """Read existing jobs without replacement executions or aggregate errors.

        Returns:
            Entry snapshots; successful results are cached.
        """
        return self._batch.collect()

    def resubmit(self, indices: Sequence[int], *, allow_unknown: bool = False) -> QDMIJob:
        """Explicitly replace selected entries of a batch created by ``backend.run``.

        Unknown outcomes require ``allow_unknown=True`` and may duplicate work.
        Running or completed jobs cannot be replaced. Use :meth:`submit` for
        entries that have never been submitted.

        Returns:
            This batch handle, with earlier attempts retained.
        """
        self._batch.resubmit(indices, allow_unknown=allow_unknown)
        return self

    def job_id(self) -> str:
        """Return the first entry's earliest accepted remote job ID.

        The ID is queried on demand and stays unchanged across replacements.

        Raises:
            JobExecutionError: If the first entry has no job handle.
        """
        if not self._job_id:
            handle = next((attempt.handle for attempt in self.entries[0].attempts if attempt.handle is not None), None)
            if handle is None:
                msg = "The first batch entry has no job handle."
                raise JobExecutionError(msg, job=self)
            self._job_id = handle.id
        return self._job_id

    def cancel(self) -> bool:
        """Disable automatic retries and attempt to cancel every job.

        Returns:
            Whether all cancellation requests succeeded.
        """
        return self._batch.cancel()

    def result(self) -> Result:
        """Get the result of the job.

        For multi-circuit jobs, this aggregates results from all submitted circuits.
        An automatic replacement can propagate :class:`~mqt.core.plugins.qiskit.exceptions.JobSubmissionError` with
        this batch handle if submission fails.

        Returns:
            The result of the job with one ExperimentResult per circuit.

        Raises:
            JobExecutionError: If collection or result assembly fails.
        """
        if self._result is not None:
            return self._result
        self._batch.complete()
        try:
            self._result = Result(
                backend_name=self._backend.name,
                backend_version=self._backend.backend_version,
                job_id=self.job_id(),
                success=True,
                date=datetime.datetime.now(datetime.UTC).isoformat(),
                results=[entry.result for entry in self.entries],
            )
        except BaseException as exc:
            self._batch.record_failure(0, "assembly", exc)
            if not isinstance(exc, Exception):
                raise
            msg = f"Failed to assemble Qiskit results: {exc}"
            raise JobExecutionError(msg, job=self) from exc
        return self._result

    def _collect_result(self, job: QDMIJobHandle, header: dict[str, Any], program_index: int = 0) -> ExperimentResult:
        """Collect and validate one circuit's result.

        Returns:
            A Qiskit experiment with counts and, if requested, ordered memory.

        Raises:
            JobError: If execution failed or the result violates the circuit's output contract.
        """
        width = header["memory_slots"]
        if self._memory:
            try:
                shots = job.get_shots(program_index)
            except Exception as exc:
                exc.add_note("memory=True and BackendSamplerV2 require valid QDMI SHOTS results.")
                raise
            if len(shots) != self._shots:
                msg = f"Invalid QDMI SHOTS result: expected {self._shots} shots, got {len(shots)}."
                raise JobError(msg)
            memory = [_encode_bits(bits, width) for bits in shots]
            data = {"memory": memory, "counts": dict(Counter(memory))}
        elif not width:
            data = {"counts": {}}
        else:
            counts = job.get_counts(program_index)
            if any(not isinstance(count, Integral) or count < 0 for count in counts.values()):
                msg = "Invalid QDMI histogram: counts must be nonnegative integers."
                raise JobError(msg)
            if sum(counts.values()) != self._shots:
                msg = f"Invalid QDMI histogram: expected {self._shots} total shots."
                raise JobError(msg)
            data = {"counts": {_encode_bits(bits, width): count for bits, count in counts.items()}}
        return ExperimentResult.from_dict({
            "success": True,
            "shots": self._shots,
            "data": data,
            "header": header,
        })

    def status(self) -> JobStatus:
        """Get the status of the job.

        For multi-circuit jobs, returns the most relevant status:
        - ERROR if any job failed
        - CANCELLED if any job was canceled (and none failed)
        - RUNNING if any job is running (and none failed/canceled)
        - QUEUED if any job is queued (and none failed/canceled/running)
        - DONE if all jobs are done

        Returns:
            The aggregated status of the job(s).

        Raises:
            ValueError: If the job status is unknown.
        """
        status_map = {
            QDMIJobHandle.Status.DONE: JobStatus.DONE,
            QDMIJobHandle.Status.RUNNING: JobStatus.RUNNING,
            QDMIJobHandle.Status.CANCELED: JobStatus.CANCELLED,
            QDMIJobHandle.Status.SUBMITTED: JobStatus.QUEUED,
            QDMIJobHandle.Status.QUEUED: JobStatus.QUEUED,
            QDMIJobHandle.Status.CREATED: JobStatus.INITIALIZING,
            QDMIJobHandle.Status.FAILED: JobStatus.ERROR,
        }

        statuses = []
        for entry, qdmi_status in zip(self.entries, self._batch.statuses(), strict=True):
            if not entry.attempts:
                statuses.append(JobStatus.INITIALIZING)
                continue
            if qdmi_status is None:
                statuses.append(JobStatus.ERROR)
                continue
            if qdmi_status not in status_map:
                msg = f"Unknown job status: {qdmi_status}"
                raise ValueError(msg)
            statuses.append(status_map[qdmi_status])

        if JobStatus.ERROR in statuses:
            return JobStatus.ERROR
        if JobStatus.CANCELLED in statuses:
            return JobStatus.CANCELLED
        if JobStatus.RUNNING in statuses:
            return JobStatus.RUNNING
        if JobStatus.QUEUED in statuses:
            return JobStatus.QUEUED
        if JobStatus.INITIALIZING in statuses:
            return JobStatus.INITIALIZING
        return JobStatus.DONE

    def submit(self, indices: Sequence[int] | None = None) -> None:
        """Submit selected untouched entries, or all remaining untouched entries.

        Previously attempted entries require :meth:`resubmit`.
        """
        self._batch.submit(indices)
