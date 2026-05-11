# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for QDMIEstimator."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.quantum_info import SparsePauliOp

from mqt.core import fomac
from mqt.core.plugins.qiskit import QDMIBackend, QDMIEstimator


@pytest.fixture
def estimator() -> QDMIEstimator:
    """Returns a QDMIEstimator based on the DDSIM backend."""
    session = fomac.Session()
    devices = session.get_devices()
    for device in devices:
        if "DDSIM" in device.name():
            backend = QDMIBackend(device=device, provider=None)
            return QDMIEstimator(backend)
    pytest.skip("DDSIM device not available")


def test_estimator_run_simple_observable(estimator: QDMIEstimator) -> None:
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


def test_estimator_run_parameterized_observable(estimator: QDMIEstimator) -> None:
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


def test_estimator_precision_handling(estimator: QDMIEstimator) -> None:
    """Test that precision argument controls the number of shots."""
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


def test_estimator_options(estimator: QDMIEstimator) -> None:
    """Test estimator options handling."""
    # Test default_shots option
    estimator2 = QDMIEstimator(estimator.backend, options={"default_shots": 500})
    qc = QuantumCircuit(1)
    op = SparsePauliOp("Z")

    job = estimator2.run([(qc, op)])
    result = job.result()
    assert result[0].metadata["shots"] == 500

    # Test default_precision option
    estimator_prec = QDMIEstimator(estimator.backend, default_precision=0.1)
    job = estimator_prec.run([(qc, op)])
    result = job.result()
    # Should use default_precision -> 100 shots
    assert result[0].metadata["shots"] == 100


def test_estimator_observable_bases(estimator: QDMIEstimator) -> None:
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


def test_estimator_broadcasting(estimator: QDMIEstimator) -> None:
    """Test broadcasting of parameters and observables."""
    # 1 qubit, parameterized
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.rx(theta, 0)

    # 2 Observables
    ops = [SparsePauliOp("Z"), SparsePauliOp("X")]

    # 2 Parameter sets (shape (2,) / two values)
    # Test asserts broadcasting for 2 observables x 2 parameter sets
    # We align shapes to (2,) so they broadcast element-wise.
    vals = {theta: [[0.0], [np.pi]]}

    pub = EstimatorPub.coerce((qc, ops, vals))
    job = estimator.run([pub])
    result = job.result()

    evs = result[0].data["evs"]
    stds = result[0].data["stds"]
    # Shape expectations: (2,) result from broadcasting
    assert evs.shape == (2,)
    assert stds.shape == (2,)


def test_estimator_no_circuits(estimator: QDMIEstimator) -> None:
    """Test run with empty pub list."""
    job = estimator.run([])
    result = job.result()
    assert len(result) == 0


def test_estimator_mismatched_qubits(estimator: QDMIEstimator) -> None:
    """Test estimator run with mismatched observable vs circuit qubit count."""
    qc = QuantumCircuit(1)  # 1 qubit
    op = SparsePauliOp("XX")  # 2 qubits

    # Should raise ValueError because EstimatorPub validation checks dimensions
    with pytest.raises(ValueError, match="number of qubits"):
        estimator.run([(qc, op)])


def test_estimator_invalid_precision(estimator: QDMIEstimator) -> None:
    """Test estimator run with invalid precision."""
    qc = QuantumCircuit(1)
    op = SparsePauliOp("Z")

    # Qiskit's EstimatorPub.coerce checks for precision validation if validate=True (default)
    # Negative precision should raise ValueError
    with pytest.raises(ValueError, match="precision"):
        # The run method calls EstimatorPub.coerce(pub, precision)
        estimator.run([(qc, op)], precision=-0.1)


def test_estimator_identity_observable_only(estimator: QDMIEstimator) -> None:
    """Test case where no measurement circuits are needed (Identity)."""
    qc = QuantumCircuit(1)
    op = SparsePauliOp("I")

    job = estimator.run([(qc, op)])
    result = job.result()

    evs = result[0].data["evs"]
    # Expectation of I is always 1
    # Result should be a scalar array 1.0
    assert float(evs) == pytest.approx(1.0)
