# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Sampler implementation for QDMI backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit.primitives.base import BaseSamplerV2
from qiskit.primitives.containers import (
    BitArray,
    DataBin,
    PrimitiveResult,
    SamplerPubResult,
)
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.primitives.primitive_job import PrimitiveJob

if TYPE_CHECKING:
    from collections.abc import Iterable

    from qiskit.primitives.containers import SamplerPubLike

    from .backend import QDMIBackend


class QDMISampler(BaseSamplerV2):
    """QDMI implementation of Qiskit's SamplerV2."""

    def __init__(
        self,
        backend: QDMIBackend,
        *,
        default_shots: int = 1024,
        options: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the QDMI Sampler.

        Args:
            backend: The QDMI backend to execute circuits on.
            default_shots: The default number of shots to use if not specified in run.
            options: Default options for the sampler.
        """
        self._backend = backend
        self._default_shots = default_shots
        self._options = options or {}

    @property
    def backend(self) -> QDMIBackend:
        """Return the backend used by this sampler."""
        return self._backend

    def run(
        self, pubs: Iterable[SamplerPubLike], *, shots: int | None = None
    ) -> PrimitiveJob[PrimitiveResult[SamplerPubResult]]:
        """Run and collect samples from each provided PUB.

        Args:
            pubs: An iterable of pub-like objects.
            shots: The number of shots to sample. If None, the default is used.

        Returns:
            A job that will return the primitive result.
        """
        if shots is None:
            shots = self._default_shots

        # Coerce inputs to SamplerPub objects
        coerced_pubs = [SamplerPub.coerce(pub, shots) for pub in pubs]

        # Use PrimitiveJob to handle asynchronous execution
        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()  # noqa: SLF001
        return job

    def _run(self, pubs: list[SamplerPub]) -> PrimitiveResult[SamplerPubResult]:
        """Execute the validated pubs.

        Args:
            pubs: The list of sampler pubs to execute.

        Returns:
            The execution results.
        """
        results = [self._run_pub(pub) for pub in pubs]
        return PrimitiveResult(results, metadata={"version": 2})

    def _run_pub(self, pub: SamplerPub) -> SamplerPubResult:
        """Execute a single PUB.

        Args:
            pub: The sampler pub to execute.

        Returns:
            The execution result.
        """
        circuit = pub.circuit
        parameter_values = pub.parameter_values
        shots = pub.shots

        # Bind parameters
        bound_circuits = parameter_values.bind_all(circuit)

        # Flatten structure to a list of circuits for backend execution
        bound_circuits_flat = np.ravel(bound_circuits).tolist()

        if not bound_circuits_flat:
            return SamplerPubResult(DataBin(shape=pub.shape), metadata={"shots": shots})

        # Run all circuits in batch on the backend
        job = self._backend.run(bound_circuits_flat, shots=shots)
        result = job.result()

        # Extract counts
        # result.get_counts() returns a list of dictionaries if multiple circuits
        # or a single dictionary if one circuit.
        all_counts = []
        if len(bound_circuits_flat) == 1:
            all_counts = [result.get_counts()]
        else:
            all_counts = result.get_counts()
            if isinstance(all_counts, dict):
                all_counts = [all_counts]

        # Restructure counts into the shape of pub.shape for each classical register
        cregs = circuit.cregs
        bit_arrays = self._get_bit_arrays(cregs, all_counts, pub.shape)

        return SamplerPubResult(
            DataBin(**bit_arrays, shape=pub.shape),
            metadata={"shots": shots, "circuit_metadata": circuit.metadata},
        )

    @staticmethod
    def _get_bit_arrays(cregs: list[Any], counts: list[dict[str, int]], shape: tuple[int, ...]) -> dict[str, BitArray]:
        """Convert counts to BitArrays for each creg.

        Args:
            cregs: The classical registers to process.
            counts: The list of counts dictionaries.
            shape: The shape of the output.

        Returns:
            The raw bit arrays.
        """
        bit_arrays = {}

        # Qiskit bitstrings are concatenated in reverse order of register declaration: "cn ... c1 c0"

        start_index = 0
        for creg in cregs:
            # Prepare list of counts for this specific register
            creg_counts = []
            for count_dict in counts:
                new_dict = {}
                for key, val in count_dict.items():
                    clean_key = key.replace(" ", "")

                    # Slice the bitstring corresponding to the current register (from right to left)
                    end = start_index if start_index != 0 else None
                    start = start_index - creg.size

                    sliced_key = clean_key[start:end]
                    new_dict[sliced_key] = new_dict.get(sliced_key, 0) + val
                creg_counts.append(new_dict)

            # Create BitArray and reshape
            array = BitArray.from_counts(creg_counts, creg.size)
            array = array.reshape(shape)
            bit_arrays[creg.name] = array

            start_index -= creg.size

        return bit_arrays
