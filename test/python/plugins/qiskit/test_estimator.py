# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for BackendEstimatorV2."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import BackendEstimatorV2
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.quantum_info import SparsePauliOp

from mqt.core.plugins.qiskit import QDMIBackend


@pytest.fixture
def backend() -> QDMIBackend:
    """Open a device for the estimator tests.

    Returns:
        A DDSIM backend with enough sites for the two-qubit circuits.
    """
    return QDMIBackend.from_device_id("mqt.ddsim.default", custom1="2")


@pytest.fixture
def estimator(backend: QDMIBackend) -> BackendEstimatorV2:
    """Create a fresh estimator for each test.

    Returns:
        An estimator with 16 shots by default.
    """
    return backend.estimator(default_precision=0.25)


def test_estimator_run_simple_observable(estimator: BackendEstimatorV2) -> None:
    """Estimator runs a simple observable estimation."""
    qc = QuantumCircuit(1)  # |0> state
    op = [SparsePauliOp("Z"), SparsePauliOp("X")]  # Expectation of Z should be 1, X should be 0

    job = estimator.run([(qc, op)], precision=None)
    result = job.result()

    assert len(result) == 1
    pub_result = result[0]

    evs = pub_result.data["evs"]
    stds = pub_result.data["stds"]
    assert evs.shape == (2,)  # 2 observables
    assert stds.shape == (2,)


def test_estimator_run_parameterized_observable(estimator: BackendEstimatorV2) -> None:
    """Estimator runs parameterized circuit and observables."""
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)

    op = SparsePauliOp("Z")

    # Run with 2 parameters using explicit EstimatorPub via coerce
    pub = EstimatorPub.coerce((qc, op, {theta: [0.0, np.pi]}))
    job = estimator.run([pub])
    result = job.result()

    pub_result = result[0]
    evs = pub_result.data["evs"]
    assert evs.shape == (2,)


def test_estimator_precision_handling(backend: QDMIBackend) -> None:
    """Test that precision argument controls the number of shots."""
    estimator = backend.estimator()
    qc = QuantumCircuit(1)
    qc.h(0)
    op = SparsePauliOp("Z")

    # Case 1: Default precision (None) -> default shots (4096)
    job = estimator.run([(qc, op)])
    result = job.result()
    assert result[0].metadata["shots"] == 4096

    # An individual pub can override the run's precision.
    pub = EstimatorPub.coerce((qc, op), precision=0.05)
    result = estimator.run([(qc, op), pub], precision=0.1).result()
    assert result[0].metadata["shots"] == 100
    assert result[1].metadata["shots"] == 400


def test_backend_constructs_estimator(backend: QDMIBackend) -> None:
    """A backend constructs an estimator that retains its identity and defaults."""
    estimator = backend.estimator(default_precision=0.125)
    qc = QuantumCircuit(1)
    op = SparsePauliOp("Z")

    assert isinstance(estimator, BackendEstimatorV2)
    assert estimator.backend is backend
    assert estimator.run([(qc, op)]).result()[0].metadata["shots"] == 64


def test_estimator_observable_bases(estimator: BackendEstimatorV2) -> None:
    """Test estimating observables in different bases."""
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
    evs = result[0].data["evs"]
    assert evs.shape == (6,)


def test_estimator_broadcasting(estimator: BackendEstimatorV2) -> None:
    """Test broadcasting of parameters and observables."""
    # 1 qubit, parameterized
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.rx(theta, 0)

    # 2 Observables
    ops = [SparsePauliOp("Z"), SparsePauliOp("X")]

    # Matching (2,) shapes pair each observable with one parameter set.
    vals = {theta: [[0.0], [np.pi]]}

    pub = EstimatorPub.coerce((qc, ops, vals))
    job = estimator.run([pub])
    result = job.result()

    evs = result[0].data["evs"]
    stds = result[0].data["stds"]
    # Shape expectations: (2,) result from broadcasting
    assert evs.shape == (2,)
    assert stds.shape == (2,)


def test_estimator_no_circuits(estimator: BackendEstimatorV2) -> None:
    """Test run with empty pub list."""
    job = estimator.run([])
    result = job.result()
    assert len(result) == 0


def test_estimator_mismatched_qubits(estimator: BackendEstimatorV2) -> None:
    """Test estimator run with mismatched observable vs circuit qubit count."""
    qc = QuantumCircuit(1)  # 1 qubit
    op = SparsePauliOp("XX")  # 2 qubits

    # Should raise ValueError because EstimatorPub validation checks dimensions
    with pytest.raises(ValueError, match="number of qubits"):
        estimator.run([(qc, op)])


@pytest.mark.parametrize("precision", [-0.1, 0.0])
def test_estimator_invalid_precision(estimator: BackendEstimatorV2, precision: float) -> None:
    """Test estimator run with invalid precision."""
    qc = QuantumCircuit(1)
    op = SparsePauliOp("Z")

    # Qiskit's EstimatorPub.coerce checks for precision validation if validate=True (default)
    # Negative precision should raise ValueError
    with pytest.raises(ValueError, match="precision"):
        # The run method calls EstimatorPub.coerce(pub, precision)
        estimator.run([(qc, op)], precision=precision)


def test_estimator_identity_observable_only(estimator: BackendEstimatorV2) -> None:
    """Test case where no measurement circuits are needed (Identity)."""
    qc = QuantumCircuit(1)
    op = SparsePauliOp("I")

    job = estimator.run([(qc, op)])
    result = job.result()

    evs = result[0].data["evs"]
    # Expectation of I is always 1
    # Result should be a scalar array 1.0
    assert float(evs) == pytest.approx(1.0)


def test_estimator_feedback_ensemble(estimator: BackendEstimatorV2) -> None:
    """Feedback produces a classical ensemble with zero target-Z mean."""
    circuit = QuantumCircuit(2, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    with circuit.if_test((circuit.clbits[0], True)):
        circuit.x(1)
    result = estimator.run([(circuit, "ZI")], precision=1 / 64).result()[0]
    # Hoeffding: P(|mean| >= .1) <= 2 exp(-4096 * .1**2 / 2) < 3e-9.
    assert abs(float(result.data["evs"])) < 0.1
    assert result.metadata["shots"] == 4096
