# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact Qiskit expectations using Core's decision-diagram engine."""

from __future__ import annotations

__lazy_modules__ = {
    "math",
    "mqt.core.mlir",
    "numpy",
    "qiskit.primitives.containers",
    "qiskit.primitives.containers.estimator_pub",
    "qiskit.primitives.primitive_job",
}

from math import isfinite
from typing import TYPE_CHECKING

import numpy as np
from qiskit.primitives import BaseEstimatorV2, PrimitiveResult, PubResult
from qiskit.primitives.containers import DataBin
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.primitive_job import PrimitiveJob

from mqt.core.mlir import QCProgram, expectation_values

if TYPE_CHECKING:
    from collections.abc import Iterable

    from qiskit.primitives.containers import EstimatorPubLike

__all__ = ["DDEstimator"]


def __dir__() -> list[str]:
    return __all__


def _validate_precision(precision: float) -> None:
    if not isfinite(precision) or precision < 0:
        msg = "precision must be finite and nonnegative"
        raise ValueError(msg)


class DDEstimator(BaseEstimatorV2):
    """Evaluate sparse Pauli observables exactly with decision diagrams.

    Each PUB is imported once. Each parameter binding prepares one
    state for all its broadcast observables.
    Eligible terminal measurements are deferred; resets and feedback are
    rejected. The result describes the uncollapsed state before readout.

    Precision is a finite nonnegative target. All requests use numerical exact
    evaluation, including positive precision; ``stds=0`` reports zero sampling
    uncertainty, not zero floating-point error or quantum variance.

    Args:
        default_precision: Target precision when neither the PUB nor ``run``
            specifies one. It does not change the exact execution method.
    """

    def __init__(self, *, default_precision: float = 0.0) -> None:
        """Set the default finite nonnegative target precision."""
        _validate_precision(default_precision)
        self._default_precision = float(default_precision)

    @property
    def default_precision(self) -> float:
        """Default requested precision; execution always remains exact."""
        return self._default_precision

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None
    ) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        """Evaluate PUBs using their broadcast observable and binding shapes.

        PUB precision overrides run precision, which overrides the default.
        Parameter columns follow ``circuit.parameters``; keyed bindings use
        Qiskit's parameter identities. Results contain ``evs`` and zero
        ``stds`` arrays, with exact-method and target-precision metadata.

        Returns:
            Asynchronous job containing one result per PUB.

        Raises:
            ValueError: If a PUB, precision or parameter value is invalid.
        """
        effective_precision = self.default_precision if precision is None else precision
        _validate_precision(effective_precision)
        coerced_pubs = [EstimatorPub.coerce(pub, effective_precision) for pub in pubs]
        for pub in coerced_pubs:
            if pub.precision is not None:
                _validate_precision(pub.precision)
            if not np.all(np.isfinite(pub.parameter_values.as_array())):
                msg = "parameter values must be finite"
                raise ValueError(msg)
        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()  # ruff: ignore[private-member-access] -- Qiskit's PrimitiveJob exposes submission only through this method.
        return job

    def _run(self, pubs: list[EstimatorPub]) -> PrimitiveResult[PubResult]:
        return PrimitiveResult([self._run_pub(pub) for pub in pubs], metadata={"version": 2})

    @staticmethod
    def _run_pub(pub: EstimatorPub) -> PubResult:
        bindings = pub.parameter_values
        binding_indices = np.arange(bindings.size).reshape(bindings.shape)
        broadcast_bindings, broadcast_observables = np.broadcast_arrays(binding_indices, pub.observables)
        groups: dict[int, list[int]] = {}
        for output_index, binding_index in enumerate(broadcast_bindings.flat):
            groups.setdefault(int(binding_index), []).append(output_index)

        evs = np.empty(pub.shape, dtype=np.float64)
        if groups:
            parameters = tuple(pub.circuit.parameters)
            names = tuple(parameter.name for parameter in parameters)
            values = bindings.as_array(parameters).reshape(bindings.size, len(parameters))
            program = QCProgram.from_qiskit(pub.circuit).to_qco()
            for binding_index, output_indices in groups.items():
                bound = program
                if parameters:
                    bound = program.bind_parameters(dict(zip(names, values[binding_index], strict=True)), copy=True)
                observables = [broadcast_observables.flat[index] for index in output_indices]
                evs.flat[output_indices] = expectation_values(bound, observables)

        return PubResult(
            DataBin(evs=evs, stds=np.zeros(pub.shape, dtype=np.float64), shape=pub.shape),
            metadata={
                "method": "exact",
                "target_precision": pub.precision,
                "circuit_metadata": pub.circuit.metadata,
            },
        )
