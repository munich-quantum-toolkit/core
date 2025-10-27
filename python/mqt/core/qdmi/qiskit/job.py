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

from typing import TYPE_CHECKING, Any

from qiskit.providers import JobStatus, JobV1
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit

    from .backend import QiskitBackend

__all__ = ["QiskitJob"]


class QiskitJob(JobV1):  # type: ignore[misc]
    """Qiskit job for QDMI backend execution.

    Args:
        backend: The backend this job runs on.
        circuits: Circuits to execute.
        results: Execution results for each circuit.
        shots: Number of shots per circuit.
    """

    def __init__(
        self,
        backend: QiskitBackend,
        circuits: list[QuantumCircuit],
        results: list[dict[str, Any]],
        shots: int,
    ) -> None:
        """Initialize the job."""
        super().__init__(backend, str(id(self)))
        self._backend = backend
        self._circuits = circuits
        self._results = results
        self._shots = shots
        self._status = JobStatus.INITIALIZING

    def submit(self) -> None:
        """Transition job to completed state."""
        self._status = JobStatus.DONE

    def result(self, timeout: float | None = None) -> Result:  # noqa: ARG002
        """Return the result object.

        Args:
            timeout: Unused (job is already complete).

        Returns:
            Qiskit Result object with experiment results.
        """
        # Build experiment results for each circuit
        experiment_results = []
        for idx, (circuit, result_dict) in enumerate(zip(self._circuits, self._results, strict=False)):
            counts = result_dict["counts"]
            metadata = result_dict.get("metadata", {})

            # Create experiment result data
            data = ExperimentResultData(counts=counts)

            # Create experiment result
            exp_result = ExperimentResult(
                shots=result_dict["shots"],
                success=result_dict.get("success", True),
                data=data,
                header={"name": circuit.name or f"circuit_{idx}"},
                metadata=metadata,
            )
            experiment_results.append(exp_result)

        # Create and return Result
        return Result(
            backend_name=self._backend.name,
            backend_version=self._backend.backend_version,
            qobj_id=self.job_id(),
            job_id=self.job_id(),
            success=all(r.get("success", True) for r in self._results),
            results=experiment_results,
            date=None,
        )

    def status(self) -> JobStatus:
        """Return the job status.

        Returns:
            The job status.
        """
        return self._status

    def get_counts(self, circuit: QuantumCircuit | int | None = None) -> dict[str, int]:
        """Convenience method to get counts from the result.

        Args:
            circuit: Optional circuit index or circuit object (default: first circuit).

        Returns:
            Dictionary of measurement counts.

        Raises:
            ValueError: If the specified circuit is not found in the job.
        """
        result = self.result()
        if circuit is None:
            return result.get_counts(0)  # type: ignore[no-any-return]
        if isinstance(circuit, int):
            return result.get_counts(circuit)  # type: ignore[no-any-return]
        # Find circuit index
        for idx, circ in enumerate(self._circuits):
            if circ is circuit:
                return result.get_counts(idx)  # type: ignore[no-any-return]
        msg = "Circuit not found in job"
        raise ValueError(msg)
