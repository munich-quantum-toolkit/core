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
from typing import TYPE_CHECKING

from qiskit.providers import JobStatus, JobV1
from qiskit.result import Result
from qiskit.result.models import ExperimentResult

from mqt.core import fomac

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .backend import QDMIBackend

__all__ = ["QDMIJob"]


def __dir__() -> list[str]:
    return __all__


class QDMIJob(JobV1):  # type: ignore[misc]
    """Qiskit job wrapping one or more QDMI/FoMaC jobs.

    This class handles both single-circuit and multi-circuit execution,
    aggregating results from multiple QDMI jobs when needed.

    Args:
        backend: The backend this job runs on.
        jobs: The FoMaC Job object(s). Can be a single job or a list of jobs.
        circuit_names: The name(s) of the circuit(s) being executed. Can be a single name or a list of names.
    """

    def __init__(
        self,
        backend: QDMIBackend,
        jobs: fomac.Job | Sequence[fomac.Job],
        circuit_names: str | Sequence[str],
    ) -> None:
        """Initialize the job.

        Args:
            backend: The backend to use for the job.
            jobs: The FoMaC Job object(s).
            circuit_names: The name(s) of the circuit(s) the job is associated with.

        Raises:
            ValueError: If jobs list is empty or if jobs and circuit_names have mismatched lengths.
        """
        # Normalize to lists
        self._jobs = [jobs] if isinstance(jobs, fomac.Job) else jobs
        self._circuit_names = [circuit_names] if isinstance(circuit_names, str) else circuit_names

        # Validate non-empty jobs list
        if not self._jobs:
            msg = "QDMIJob must be initialized with at least one underlying job."
            raise ValueError(msg)

        # Validate that jobs and circuit_names have matching lengths
        if len(self._jobs) != len(self._circuit_names):
            msg = (
                f"Length mismatch: jobs ({len(self._jobs)}) and circuit_names ({len(self._circuit_names)}) "
                "must have the same length."
            )
            raise ValueError(msg)

        # Use the first job's ID as the primary job ID
        job_id = self._jobs[0].id
        super().__init__(backend=backend, job_id=job_id)
        self._backend: QDMIBackend = backend
        self._counts_cache: list[dict[str, int] | None] = [None] * len(self._jobs)

    def result(self) -> Result:
        """Get the result of the job.

        For multi-circuit jobs, this aggregates results from all submitted circuits.

        Returns:
            The result of the job with one ExperimentResult per circuit.
        """
        experiment_results = []
        overall_success = True

        for idx, (job, circuit_name) in enumerate(zip(self._jobs, self._circuit_names, strict=True)):
            # Wait for job completion if needed
            status = job.check()
            if status not in {fomac.Job.Status.DONE, fomac.Job.Status.FAILED, fomac.Job.Status.CANCELED}:
                job.wait()
                status = job.check()

            success = status == fomac.Job.Status.DONE
            overall_success = overall_success and success

            # Get counts if successful and not cached
            if self._counts_cache[idx] is None and success:
                self._counts_cache[idx] = job.get_counts()

            exp_result = ExperimentResult.from_dict({
                "success": success,
                "shots": job.num_shots,
                "data": {"counts": self._counts_cache[idx], "metadata": {}},
                "header": {"name": circuit_name},
            })
            experiment_results.append(exp_result)

        return Result(
            backend_name=self._backend.name,
            backend_version=self._backend.backend_version,
            qobj_id=self.job_id(),
            job_id=self.job_id(),
            success=overall_success,
            date=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            results=experiment_results,
        )

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
            fomac.Job.Status.DONE: JobStatus.DONE,
            fomac.Job.Status.RUNNING: JobStatus.RUNNING,
            fomac.Job.Status.CANCELED: JobStatus.CANCELLED,
            fomac.Job.Status.SUBMITTED: JobStatus.QUEUED,
            fomac.Job.Status.QUEUED: JobStatus.QUEUED,
            fomac.Job.Status.CREATED: JobStatus.INITIALIZING,
            fomac.Job.Status.FAILED: JobStatus.ERROR,
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

        QDMI jobs are submitted via :meth:`~mqt.core.plugins.qiskit.QDMIBackend.run`.
        """
        msg = (
            "You should never have to submit jobs by calling this method. "
            "The job instance is only for checking the progress and retrieving the results of the submitted job."
        )
        raise NotImplementedError(msg)
