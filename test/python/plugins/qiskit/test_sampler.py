# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for QDMISampler."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, Parameter
from qiskit.primitives import BaseSamplerV2
from qiskit.primitives.containers import BitArray
from qiskit.primitives.containers.sampler_pub import SamplerPub

from mqt.core.plugins.qiskit import QDMISampler

if TYPE_CHECKING:
    from mqt.core.plugins.qiskit import QDMIBackend


def _get_bit_array(data: object, register: str) -> BitArray:
    """Extract a classical-register BitArray from sampler pub result data.

    Returns:
        The extracted bit array for the requested classical register.
    """
    bit_array = getattr(data, register, None)
    assert isinstance(bit_array, BitArray)
    return bit_array


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
    bit_array = _get_bit_array(pub_result.data, "meas")
    assert bit_array.num_shots == 100
    assert bit_array.num_bits == 2
    assert bit_array.shape == ()

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
    job = sampler.run([pub], shots=100)  # type: ignore[arg-type]
    result = job.result()

    pub_result = result[0]
    assert pub_result.metadata["shots"] == 100

    # Shape should be (2,) because we provided 2 parameter sets
    bit_array = _get_bit_array(pub_result.data, "meas")
    assert bit_array.shape == (2,)
    assert bit_array.num_shots == 100
    assert bit_array.num_bits == 1

    # Check individual results
    try:
        # For Qiskit versions >=2.3
        counts0 = bit_array[0].get_counts()
        counts1 = bit_array[1].get_counts()
    except TypeError:
        # For Qiskit versions <2.3
        bitstrings = bit_array.get_bitstrings()
        # The bitstrings are interleaved for the two parameter sets. We take every second one.
        bitstrings0 = bitstrings[0::2]
        bitstrings1 = bitstrings[1::2]
        counts0 = Counter(bitstrings0)
        counts1 = Counter(bitstrings1)

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
    c0_bits = _get_bit_array(pub_result.data, "c0")
    c1_bits = _get_bit_array(pub_result.data, "c1")

    assert c0_bits.num_bits == 1
    assert c1_bits.num_bits == 1


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
    bit_array = _get_bit_array(pub_result.data, "meas")
    assert bit_array.shape == (2, 2)
    assert bit_array.num_shots == 100
