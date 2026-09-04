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
from numbers import Integral
from typing import TYPE_CHECKING, Any

from qiskit.providers import JobError, JobStatus, JobV1
from qiskit.result import Result
from qiskit.result.models import ExperimentResult

from mqt.core.qdmi import Job as QDMIJobHandle

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qiskit.circuit import QuantumCircuit

    from .backend import QDMIBackend

__all__ = ["QDMIJob"]


def __dir__() -> list[str]:
    return __all__


def _cancel_jobs(jobs: Sequence[QDMIJobHandle]) -> bool:
    """Attempt every cancellation without masking an execution failure.

    Returns:
        Whether every cancellation succeeded.
    """
    success = True
    for job in jobs:
        try:
            job.cancel()
        except BaseException:  # ruff:ignore[blind-except] Cleanup must preserve the original failure, even on interruption.
            success = False
    return success


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
    aggregating results from multiple QDMI jobs when needed.

    Args:
        backend: The backend this job runs on.
        jobs: Submitted QDMI jobs, in circuit order.
        circuits: The executed circuits, used to snapshot result headers.
        shots: Requested shots per circuit.
        memory: Whether to collect genuine ordered shots.
    """

    def __init__(
        self,
        backend: QDMIBackend,
        jobs: Sequence[QDMIJobHandle],
        circuits: Sequence[QuantumCircuit],
        *,
        shots: int,
        memory: bool,
    ) -> None:
        """Initialize without querying remote job IDs.

        Raises:
            ValueError: If the jobs and circuits are empty or differ in length.
        """
        if not jobs or len(jobs) != len(circuits):
            msg = "QDMIJob requires one submitted job per circuit and at least one circuit."
            raise ValueError(msg)
        super().__init__(backend=backend, job_id="")
        self._backend: QDMIBackend = backend
        self._jobs = list(jobs)
        self._headers: list[dict[str, Any]] = [
            {
                "name": circuit.name,
                "memory_slots": circuit.num_clbits,
                "creg_sizes": [[register.name, register.size] for register in circuit.cregs],
                "metadata": circuit.metadata.copy(),
            }
            for circuit in circuits
        ]
        self._shots = shots
        self._memory = memory
        self._result: Result | None = None

    def job_id(self) -> str:
        """Return the first remote job ID, querying it only when requested."""
        if not self._job_id:
            self._job_id = self._jobs[0].id
        return self._job_id

    def cancel(self) -> bool:
        """Attempt to cancel every job.

        Returns:
            Whether all cancellation requests succeeded.
        """
        return _cancel_jobs(self._jobs)

    def result(self) -> Result:
        """Get the result of the job.

        For multi-circuit jobs, this aggregates results from all submitted circuits.

        Returns:
            The result of the job with one ExperimentResult per circuit.
        """
        if self._result is not None:
            return self._result
        try:
            experiment_results = list(map(self._collect_result, self._jobs, self._headers))
            self._result = Result(
                backend_name=self._backend.name,
                backend_version=self._backend.backend_version,
                job_id=self.job_id(),
                success=True,
                date=datetime.datetime.now(datetime.UTC).isoformat(),
                results=experiment_results,
            )
        except BaseException:
            _cancel_jobs(self._jobs)
            raise
        return self._result

    def _collect_result(self, job: QDMIJobHandle, header: dict[str, Any]) -> ExperimentResult:
        """Collect and validate one circuit's result.

        Returns:
            A Qiskit experiment with counts and, if requested, ordered memory.

        Raises:
            JobError: If execution failed or the result violates the circuit's output contract.
        """
        status = job.check()
        if status not in {
            QDMIJobHandle.Status.DONE,
            QDMIJobHandle.Status.FAILED,
            QDMIJobHandle.Status.CANCELED,
        }:
            job.wait()
            status = job.check()
        if status != QDMIJobHandle.Status.DONE:
            msg = f"QDMI job did not complete successfully: {status.name}."
            raise JobError(msg)

        width = header["memory_slots"]
        if self._memory:
            try:
                shots = job.get_shots()
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
            counts = job.get_counts()
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
        # Map QDMI status to Qiskit JobStatus
        status_map = {
            QDMIJobHandle.Status.DONE: JobStatus.DONE,
            QDMIJobHandle.Status.RUNNING: JobStatus.RUNNING,
            QDMIJobHandle.Status.CANCELED: JobStatus.CANCELLED,
            QDMIJobHandle.Status.SUBMITTED: JobStatus.QUEUED,
            QDMIJobHandle.Status.QUEUED: JobStatus.QUEUED,
            QDMIJobHandle.Status.CREATED: JobStatus.INITIALIZING,
            QDMIJobHandle.Status.FAILED: JobStatus.ERROR,
        }

        # Collect all statuses (self._jobs is guaranteed non-empty by __init__)
        statuses = []
        for job in self._jobs:
            qdmi_status = job.check()
            if qdmi_status not in status_map:
                msg = f"Unknown job status: {qdmi_status}"
                raise ValueError(msg)
            statuses.append(status_map[qdmi_status])

        # Aggregate statuses by priority
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
        # All jobs must be DONE
        return JobStatus.DONE

    def submit(self) -> None:
        """This method should not be called.

        QDMI jobs are submitted via
        :meth:`~mqt.core.plugins.qiskit.QDMIBackend.run`.
        """
        msg = (
            "You should never have to submit jobs by calling this method. "
            "The job instance is only for checking the progress and retrieving the results of the submitted job."
        )
        raise NotImplementedError(msg)
