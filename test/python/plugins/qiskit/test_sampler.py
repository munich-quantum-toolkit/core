# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for QDMISampler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, Parameter
from qiskit.primitives import BaseSamplerV2
from qiskit.primitives.containers.sampler_pub import SamplerPub

from mqt.core.plugins.qiskit import QDMISampler

if TYPE_CHECKING:
    from mqt.core.plugins.qiskit import QDMIBackend

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*Device operation.*cannot be mapped to a Qiskit gate.*:UserWarning"),
]


def test_sampler_instantiation(backend_with_mock_jobs: QDMIBackend) -> None:
    """Sampler can be instantiated with a backend."""
    sampler = QDMISampler(backend_with_mock_jobs)
    assert isinstance(sampler, BaseSamplerV2)
    assert sampler.backend == backend_with_mock_jobs


def test_sampler_run_simple_circuit(backend_with_mock_jobs: QDMIBackend) -> None:
    """Sampler runs a simple circuit."""
    sampler = QDMISampler(backend_with_mock_jobs)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    job = sampler.run([(qc,)], shots=100)
    result = job.result()

    assert len(result) == 1
    pub_result = result[0]
    assert pub_result.metadata["shots"] == 100

    # Check data bin structure
    data = cast("Any", pub_result.data)
    assert hasattr(data, "meas")
    bit_array = data.meas
    assert bit_array.num_shots == 100
    assert bit_array.num_bits == 2

    # Check that we got some counts (mock backend returns random distribution)
    counts = bit_array.get_counts()
    assert sum(counts.values()) == 100


def test_sampler_run_parameterized_circuit(backend_with_mock_jobs: QDMIBackend) -> None:
    """Sampler runs a parameterized circuit."""
    sampler = QDMISampler(backend_with_mock_jobs)
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    qc.measure_all()

    # Run with two different parameter values using explicit SamplerPub via coerce
    pub = SamplerPub.coerce((qc, [[0.0], [np.pi]]))  # type: ignore[arg-type]
    job = sampler.run(cast("Any", [pub]), shots=100)
    result = job.result()

    pub_result = result[0]
    assert pub_result.metadata["shots"] == 100

    # Shape should be (2,) because we provided 2 parameter sets
    data = cast("Any", pub_result.data)
    bit_array = data.meas
    assert bit_array.shape == (2,)
    assert bit_array.num_shots == 100
    assert bit_array.num_bits == 1

    # Check individual results
    counts0 = bit_array[0].get_counts()
    counts1 = bit_array[1].get_counts()
    assert sum(counts0.values()) == 100
    assert sum(counts1.values()) == 100


def test_sampler_run_multiple_cregs(backend_with_mock_jobs: QDMIBackend) -> None:
    """Sampler correctly handles multiple classical registers."""
    sampler = QDMISampler(backend_with_mock_jobs)
    c0 = ClassicalRegister(1, "c0")
    c1 = ClassicalRegister(1, "c1")
    qc = QuantumCircuit(2)
    qc.add_register(c0)
    qc.add_register(c1)
    qc.h(0)
    qc.measure(0, c0[0])
    qc.measure(1, c1[0])

    job = sampler.run([(qc,)], shots=100)
    result = job.result()

    pub_result = result[0]
    data = cast("Any", pub_result.data)

    assert hasattr(data, "c0")
    assert hasattr(data, "c1")

    assert data.c0.num_bits == 1
    assert data.c1.num_bits == 1


def test_sampler_options(backend_with_mock_jobs: QDMIBackend) -> None:
    """Test sampler options mechanism."""
    # 1. Use default shots from init
    sampler = QDMISampler(backend_with_mock_jobs, default_shots=500)
    qc = QuantumCircuit(1)
    qc.measure_all()

    job = sampler.run([(qc,)])
    result = job.result()
    assert result[0].metadata["shots"] == 500

    # 2. Override via run method
    job = sampler.run([(qc,)], shots=200)
    result = job.result()
    assert result[0].metadata["shots"] == 200


def test_sampler_no_circuits(backend_with_mock_jobs: QDMIBackend) -> None:
    """Test run with empty pub list."""
    sampler = QDMISampler(backend_with_mock_jobs)
    job = sampler.run([])
    result = job.result()
    assert len(result) == 0


def test_sampler_broadcasting(backend_with_mock_jobs: QDMIBackend) -> None:
    """Test sampler with parameter broadcasting."""
    sampler = QDMISampler(backend_with_mock_jobs)
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.rx(theta, 0)
    qc.measure_all()

    # Broadcast parameters
    params = np.zeros((2, 2))

    pub = SamplerPub.coerce((qc, params))  # type: ignore[arg-type]
    job = sampler.run([pub], shots=100)  # type: ignore[arg-type]
    result = job.result()

    pub_result = result[0]
    data = cast("Any", pub_result.data)
    bit_array = data.meas
    assert bit_array.shape == (2, 2)
    assert bit_array.num_shots == 100
