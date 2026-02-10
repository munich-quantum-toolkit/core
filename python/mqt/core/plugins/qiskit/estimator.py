# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Estimator implementation for QDMI backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.primitives.base import BaseEstimatorV2
from qiskit.primitives.containers import DataBin, PrimitiveResult, PubResult

try:
    # For Qiskit versions >= 1.1
    from qiskit.primitives.containers import make_data_bin
except ImportError:
    # For Qiskit versions < 1.1
    from qiskit.primitives.containers.data_bin import (
        _make_data_bin as make_data_bin,  # noqa: PLC2701 # ty: ignore[unresolved-import]
    )

from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.quantum_info import SparsePauliOp

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray
    from qiskit.primitives.containers import EstimatorPubLike
    from qiskit.primitives.containers.observables_array import ObservablesArray
    from qiskit.quantum_info import Pauli

    from .backend import QDMIBackend


class QDMIEstimator(BaseEstimatorV2):
    """QDMI implementation of Qiskit's EstimatorV2."""

    def __init__(
        self,
        backend: QDMIBackend,
        *,
        default_precision: float = 0.0,
        options: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the QDMI Estimator.

        Args:
            backend: The QDMI backend to execute circuits on.
            default_precision: The default precision for expectation-value estimates.
            options: Default options for the estimator.
        """
        self._backend = backend
        self._default_precision = default_precision
        self._options = options or {}

    @property
    def backend(self) -> QDMIBackend:
        """Return the backend used by this estimator."""
        return self._backend

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None
    ) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        """Estimate expectation values for each provided PUB.

        Args:
            pubs: An iterable of pub-like objects.
            precision: The target precision. If None, the stored default precision is used.

        Returns:
            A job that will return the primitive result.
        """
        if precision is None:
            precision = self._default_precision

        # Coerce inputs to EstimatorPub objects
        coerced_pubs = [EstimatorPub.coerce(pub, precision) for pub in pubs]

        # Use PrimitiveJob to handle asynchronous execution
        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()  # noqa: SLF001
        return job

    def _run(self, pubs: list[EstimatorPub]) -> PrimitiveResult[PubResult]:
        """Execute the validated pubs.

        Args:
            pubs: The list of estimator pubs to execute.

        Returns:
            The execution results.
        """
        # Calculate results for each pub
        results = [self._run_pub(pub) for pub in pubs]
        return PrimitiveResult(results, metadata={"version": 2})

    def _run_pub(self, pub: EstimatorPub) -> PubResult:
        """Execute a single PUB.

        Args:
            pub: The estimator pub to execute.

        Returns:
            The execution result.
        """
        circuit = pub.circuit
        observables = pub.observables
        parameter_values = pub.parameter_values
        precision = pub.precision

        # Calculate shots based on precision (if provided), otherwise use default
        shots = self._options.get("default_shots", 1024)
        if precision is not None and precision > 0:
            shots = int(np.ceil(1.0 / precision**2))

        # Get observable measurement circuits (one for each Pauli term)
        observable_circuits = self._get_observable_circuits(observables, circuit.num_qubits)

        # Bind parameters to the base circuit
        bound_circuits = parameter_values.bind_all(circuit)

        # Broadcast bound circuits and observable circuits
        bc_bound_circuits, bc_observable_circuits = np.broadcast_arrays(bound_circuits, observable_circuits)

        evs = np.zeros_like(bc_bound_circuits, dtype=np.float64)
        stds = np.zeros_like(bc_bound_circuits, dtype=np.float64)

        # Flatten all combinations of bound circuits and measurement preparations
        total_circuits_to_run = []
        metadata_map = []  # type: list[tuple[tuple[int, ...], complex, NDArray[np.int_]]]

        for index in np.ndindex(*bc_bound_circuits.shape):
            bound_circuit = bc_bound_circuits[index]
            obs_coeffs, meas_circuits, term_indices_list = bc_observable_circuits[index]

            for coeff, meas_circ, indices in zip(obs_coeffs, meas_circuits, term_indices_list, strict=True):
                full_circ = bound_circuit.compose(meas_circ)
                total_circuits_to_run.append(full_circ)
                metadata_map.append((index, coeff, indices))

        if not total_circuits_to_run:
            return PubResult(DataBin(evs=evs, stds=stds, shape=evs.shape))

        # Run all circuits
        job = self._backend.run(total_circuits_to_run, shots=shots)
        result = job.result()

        # Process results
        # result.get_counts returns list of dicts corresponding to total_circuits_to_run
        all_counts = result.get_counts()
        if isinstance(all_counts, dict):
            all_counts = [all_counts]

        # Group results by (index, coeff) to reconstruct expectation values
        # index points to the broadcasted shape of parameters/observables
        temp_results = {}  # index -> {"ev": float, "var": float}

        for i, (index, coeff, indices) in enumerate(metadata_map):
            counts = all_counts[i]

            # Calculate expectation value for this Pauli term
            exp_val = 0.0
            for bitstring, count in counts.items():
                clean_bitstring = bitstring.replace(" ", "")

                # Calculate parity only for the qubits involved in this Pauli term
                parity = 0
                for bit_idx in indices:
                    # Map bit_idx to string position (right-to-left 0-indexed)
                    # bitstring is like "cn...c1c0"
                    if clean_bitstring[-1 - bit_idx] == "1":
                        parity += 1

                sign = -1 if parity % 2 else 1
                exp_val += sign * count

            exp_val /= shots

            # Variance for this term = 1 - E^2
            variance = 1.0 - exp_val**2
            term_var = variance / shots

            # Accumulate weighted expectation value and variance
            if index not in temp_results:
                temp_results[index] = {"ev": 0.0, "var": 0.0}

            # Add term contribution
            temp_results[index]["ev"] += exp_val * coeff
            # Variance adds up as sum(coeff^2 * var_term) assuming independent measurements
            temp_results[index]["var"] += (coeff.real**2) * term_var

        # Fill output arrays
        for index, res in temp_results.items():
            evs[index] = res["ev"].real
            stds[index] = np.sqrt(res["var"]).real

        cls = make_data_bin([("evs", np.ndarray), ("stds", np.ndarray)], shape=evs.shape)
        data = cls(evs=evs, stds=stds)

        return PubResult(data, metadata={"shots": shots})

    def _get_observable_circuits(
        self,
        observables: ObservablesArray,
        num_qubits: int,
    ) -> NDArray[np.object_]:
        """Get the quantum-circuit representations of the observables.

        Args:
            observables: The observables to decompose.
            num_qubits: The number of qubits in the circuit.

        Returns:
            The observable circuits.
        """
        observable_circuits = np.zeros_like(observables, dtype=object)

        for index in np.ndindex(*observables.shape):
            observable = observables[index]  # SparsePauliOp

            # Decompose into Pauli strings
            if not isinstance(observable, SparsePauliOp):
                if isinstance(observable, dict):
                    observable = SparsePauliOp.from_list(list(observable.items()))
                else:
                    observable = SparsePauliOp(observable)

            paulis = observable.paulis
            coeffs = observable.coeffs

            # Create a list of circuits (one per Pauli term)
            term_circuits = []
            term_indices = []

            for pauli in paulis:
                qc = self._create_measurement_circuit(pauli, num_qubits)
                term_circuits.append(qc)

                # Store indices of qubits that are part of the Pauli term (non-identity)
                active = np.logical_or(pauli.x, pauli.z)
                term_indices.append(np.flatnonzero(active))

            observable_circuits[index] = (coeffs, term_circuits, term_indices)

        return observable_circuits

    @staticmethod
    def _create_measurement_circuit(pauli: Pauli, num_qubits: int) -> QuantumCircuit:
        """Create a circuit to measure a specific Pauli operator.

        Args:
            pauli: The Pauli operator to measure.
            num_qubits: The number of qubits in the circuit.

        Returns:
            The measurement circuit.
        """
        qc = QuantumCircuit(num_qubits, num_qubits)

        # Iterate qubits in Pauli to determine basis rotation
        for i in range(num_qubits):
            is_x = pauli.x[i]
            is_z = pauli.z[i]

            if not is_x and not is_z:
                # Identity, no measurement needed
                pass
            else:
                # Basis rotation
                if is_x and not is_z:  # X basis
                    qc.h(i)
                elif is_x and is_z:  # Y basis
                    qc.sdg(i)
                    qc.h(i)
                # Z basis (is_z and not is_x) -> No rotation needed

                # Measure
                qc.measure(i, i)

        return qc
