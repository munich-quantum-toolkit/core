# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for QDMIEstimator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import BaseEstimatorV2
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.quantum_info import SparsePauliOp

from mqt.core.plugins.qiskit import QDMIEstimator

if TYPE_CHECKING:
    from mqt.core.plugins.qiskit import QDMIBackend

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*Device operation.*cannot be mapped to a Qiskit gate.*:UserWarning"),
]


def test_estimator_instantiation(backend_with_mock_jobs: QDMIBackend) -> None:
    """Estimator can be instantiated with a backend."""
    estimator = QDMIEstimator(backend_with_mock_jobs)
    assert isinstance(estimator, BaseEstimatorV2)
    assert estimator.backend == backend_with_mock_jobs


def test_estimator_run_simple_observable(backend_with_mock_jobs: QDMIBackend) -> None:
    """Estimator runs a simple observable estimation."""
    estimator = QDMIEstimator(backend_with_mock_jobs)
    qc = QuantumCircuit(1)  # |0> state
    op = [SparsePauliOp("Z"), SparsePauliOp("X")]  # Expectation of Z should be 1, X should be 0

    job = estimator.run([(qc, op)], precision=None)
    result = job.result()

    assert len(result) == 1
    pub_result = result[0]

    # Values are simulated by mock backend (random), so we just check structure
    data = cast("Any", pub_result.data)
    assert data.evs.shape == (2,)  # 2 observables
    assert data.stds.shape == (2,)


def test_estimator_run_parameterized_observable(backend_with_mock_jobs: QDMIBackend) -> None:
    """Estimator runs parameterized circuit and observables."""
    estimator = QDMIEstimator(backend_with_mock_jobs)
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)

    op = SparsePauliOp("Z")

    # Run with 2 parameters using explicit EstimatorPub via coerce
    pub = EstimatorPub.coerce((qc, op, [0.0, np.pi]))  # type: ignore[arg-type]
    job = estimator.run(cast("Any", [pub]))
    result = job.result()

    pub_result = result[0]
    data = cast("Any", pub_result.data)
    assert data.evs.shape == (2,)


def test_estimator_precision_handling(backend_with_mock_jobs: QDMIBackend) -> None:
    """Test that precision argument controls the number of shots."""
    estimator = QDMIEstimator(backend_with_mock_jobs)
    qc = QuantumCircuit(1)
    qc.h(0)
    op = SparsePauliOp("Z")

    # Case 1: Default precision (None) -> default shots (1024)
    job = estimator.run([(qc, op)])
    result = job.result()
    assert result[0].metadata["shots"] == 1024

    # Case 2: Specific precision -> 1/precision^2 shots
    precision = 0.1
    expected_shots = int(np.ceil(1.0 / precision**2))  # 100
    job = estimator.run([(qc, op)], precision=precision)
    result = job.result()
    assert result[0].metadata["shots"] == expected_shots

    # Case 3: Pub-specific precision override
    other_precision = 0.05
    other_expected_shots = int(np.ceil(1.0 / other_precision**2))  # 400
    pub = EstimatorPub.coerce((qc, op), precision=other_precision)
    job = estimator.run([pub])
    result = job.result()
    assert result[0].metadata["shots"] == other_expected_shots


def test_estimator_options(backend_with_mock_jobs: QDMIBackend) -> None:
    """Test estimator options handling."""
    # Test default_shots option
    estimator = QDMIEstimator(backend_with_mock_jobs, options={"default_shots": 500})
    qc = QuantumCircuit(1)
    op = SparsePauliOp("Z")

    job = estimator.run([(qc, op)])
    result = job.result()
    assert result[0].metadata["shots"] == 500

    # Test default_precision option
    estimator_prec = QDMIEstimator(backend_with_mock_jobs, default_precision=0.1)
    job = estimator_prec.run([(qc, op)])
    result = job.result()
    # Should use default_precision -> 100 shots
    assert result[0].metadata["shots"] == 100


def test_estimator_observable_bases(backend_with_mock_jobs: QDMIBackend) -> None:
    """Test estimating observables in different bases."""
    estimator = QDMIEstimator(backend_with_mock_jobs)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    # Test X, Y, Z, I mixed
    # Terms: II, XX, YY, ZZ
    ops = [
        SparsePauliOp("II"),
        SparsePauliOp("XX"),
        SparsePauliOp("YY"),
        SparsePauliOp("ZZ"),
        SparsePauliOp("IX"),
        SparsePauliOp("YI"),
    ]

    job = estimator.run([(qc, ops)])
    result = job.result()

    assert len(result) == 1
    data = cast("Any", result[0].data)
    assert data.evs.shape == (6,)

    # Check that we can run without error.
    # Logic verification is hard with mock random results, but we verify it doesn't crash on basis changes.


def test_estimator_broadcasting(backend_with_mock_jobs: QDMIBackend) -> None:
    """Test broadcasting of parameters and observables."""
    estimator = QDMIEstimator(backend_with_mock_jobs)

    # 1 qubit, parameterized
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.rx(theta, 0)

    # 2 Observables
    ops = [SparsePauliOp("Z"), SparsePauliOp("X")]

    # 2 Parameter sets (shape (2,) / two values)
    # Test asserts broadcasting for 2 observables x 2 parameter sets
    # We align shapes to (2,) so they broadcast element-wise.
    vals = [[0.0], [np.pi]]

    pub = EstimatorPub.coerce((qc, ops, vals))  # type: ignore[arg-type]
    job = estimator.run([pub])
    result = job.result()

    data = cast("Any", result[0].data)
    # Shape expectations: (2,) result from broadcasting
    assert data.evs.shape == (2,)
    assert data.stds.shape == (2,)


def test_estimator_no_circuits(backend_with_mock_jobs: QDMIBackend) -> None:
    """Test run with empty pub list."""
    estimator = QDMIEstimator(backend_with_mock_jobs)
    job = estimator.run([])
    result = job.result()
    assert len(result) == 0


def test_estimator_identity_observable_only(backend_with_mock_jobs: QDMIBackend) -> None:
    """Test case where no measurement circuits are needed (Identity)."""
    estimator = QDMIEstimator(backend_with_mock_jobs)
    qc = QuantumCircuit(1)
    op = SparsePauliOp("I")

    job = estimator.run([(qc, op)])
    result = job.result()

    data = cast("Any", result[0].data)
    # Expectation of I is always 1
    # Result should be a scalar array 1.0
    assert float(data.evs) == 1.0
