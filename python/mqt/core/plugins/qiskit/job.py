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
    from .backend import QDMIBackend

__all__ = ["QDMIJob"]


def __dir__() -> list[str]:
    return __all__


class QDMIJob(JobV1):  # type: ignore[misc]
    """Qiskit job wrapping a QDMI/FoMaC job.

    Args:
        backend: The backend this job runs on.
        job: The FoMaC Job object.
        circuit_name: The name of the circuit being executed.
    """

    def __init__(
        self,
        backend: QDMIBackend,
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
        self._backend: QDMIBackend = backend
        self._job = job
        self._circuit_name = circuit_name
        self._counts: dict[str, int] | None = None

    def result(self) -> Result:
        """Get the result of the job.

        Returns:
            The result of the job.
        """
        status = self._job.check()
        if status not in {fomac.Job.Status.DONE, fomac.Job.Status.FAILED, fomac.Job.Status.CANCELED}:
            self._job.wait()
            status = self._job.check()

        success = status == fomac.Job.Status.DONE
        if self._counts is None and success:
            self._counts = self._job.get_counts()

        exp_result = ExperimentResult.from_dict({
            "success": success,
            "shots": self._job.num_shots,
            "data": {"counts": self._counts, "metadata": {}},
            "header": {"name": self._circuit_name},
        })

        return Result(
            backend_name=self._backend.name,
            backend_version=self._backend.backend_version,
            qobj_id=self.job_id(),
            job_id=self.job_id(),
            success=success,
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
            fomac.Job.Status.DONE: JobStatus.DONE,
            fomac.Job.Status.RUNNING: JobStatus.RUNNING,
            fomac.Job.Status.CANCELED: JobStatus.CANCELLED,
            fomac.Job.Status.SUBMITTED: JobStatus.QUEUED,
            fomac.Job.Status.QUEUED: JobStatus.QUEUED,
            fomac.Job.Status.CREATED: JobStatus.INITIALIZING,
            fomac.Job.Status.FAILED: JobStatus.ERROR,
        }
        if qdmi_status in status_map:
            return status_map[qdmi_status]
        msg = f"Unknown job status: {qdmi_status}"
        raise ValueError(msg)

    def submit(self) -> None:
        """This method should not be called.

        QDMI jobs are submitted via :meth:`~mqt.core.plugins.qiskit.QDMIBackend.run`.
        """
        msg = (
            "You should never have to submit jobs by calling this method. "
            "The job instance is only for checking the progress and retrieving the results of the submitted job."
        )
        raise NotImplementedError(msg)
