# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
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
    from .backend import QiskitBackend

__all__ = ["QiskitJob"]


class QiskitJob(JobV1):  # type: ignore[misc]
    """Qiskit job wrapping a QDMI/FoMaC job.

    Args:
        backend: The backend this job runs on.
        job: The FoMaC Job object.
        circuit_name: The name of the circuit being executed.
    """

    def __init__(
        self,
        backend: QiskitBackend,
        job: fomac.Job,
        circuit_name: str,
    ) -> None:
        """Initialize the job.

        Args:
            backend: The backend to use for the job.
            job: The FoMaC Job object.
            circuit_name: The name of the circuit the job is associated with.
        """
        super().__init__(backend=backend, job_id=job.id)
        self._backend = backend
        self._job = job
        self._circuit_name = circuit_name
        self._counts: dict[str, int] | None = None

    def result(self) -> Result:
        """Get the result of the job.

        Returns:
            The result of the job.
        """
        if self._counts is None:
            self._job.wait()
            self._counts = self._job.get_counts()

        exp_result = ExperimentResult.from_dict({
            "success": True,
            "shots": self._job.num_shots,
            "data": {"counts": self._counts, "metadata": {}},
            "header": {"name": self._circuit_name},
        })

        return Result(
            backend_name=self._backend.name,
            backend_version=self._backend.backend_version,
            qobj_id=self.job_id(),
            job_id=self.job_id(),
            success=True,
            date=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            results=[exp_result],
        )

    def status(self) -> JobStatus:
        """Get the status of the job.

        Returns:
            The status of the job.

        Raises:
            ValueError: If the job status is unknown.
        """
        qdmi_status = self._job.check()
        # Map QDMI status to Qiskit JobStatus
        status_map = {
            fomac.JobStatus.DONE: JobStatus.DONE,
            fomac.JobStatus.RUNNING: JobStatus.RUNNING,
            fomac.JobStatus.CANCELED: JobStatus.CANCELLED,
            fomac.JobStatus.SUBMITTED: JobStatus.QUEUED,
            fomac.JobStatus.QUEUED: JobStatus.QUEUED,
            fomac.JobStatus.CREATED: JobStatus.INITIALIZING,
            fomac.JobStatus.FAILED: JobStatus.ERROR,
        }
        if qdmi_status in status_map:
            return status_map[qdmi_status]
        msg = f"Unknown job status: {qdmi_status}"
        raise ValueError(msg)

    def submit(self) -> None:
        """This method should not be called.

        QDMI jobs are submitted via :class:`~mqt.core.qdmi.qiskit.QiskitBackend`'s run() method.
        """
        msg = (
            "You should never have to submit jobs by calling this method. "
            "The job instance is only for checking the progress and retrieving the results of the submitted job."
        )
        raise NotImplementedError(msg)
