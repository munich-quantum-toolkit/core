# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for BackendSamplerV2."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, Parameter
from qiskit.primitives import BackendSamplerV2
from test_mock_backend import ShotQDMIDevice

from mqt.core.plugins.qiskit import QDMIBackend

if TYPE_CHECKING:
    from mqt.core.qdmi import Device


@pytest.fixture
def sampler() -> BackendSamplerV2:
    """Return a native sampler using genuine reference-simulator shots."""
    return BackendSamplerV2(backend=QDMIBackend(cast("Device", ShotQDMIDevice())))


def test_sampler_run_simple_circuit(sampler: BackendSamplerV2) -> None:
    """Sampler runs a simple circuit."""
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
    bit_array = pub_result.data["meas"]
    assert bit_array.num_shots == 100
    assert bit_array.num_bits == 2
    assert bit_array.shape == ()

    # Check that we got some counts
    counts = bit_array.get_counts()
    assert sum(counts.values()) == 100


def test_sampler_run_parameterized_circuit(sampler: BackendSamplerV2) -> None:
    """Sampler runs a parameterized circuit."""
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(theta, 0)
    qc.measure_all()

    # Run with two different parameter values
    params = {theta: [[0], [np.pi]]}
    job = sampler.run([(qc, params)], shots=100)
    result = job.result()

    pub_result = result[0]
    assert pub_result.metadata["shots"] == 100

    # Shape should be (2,) because we provided 2 parameter sets
    bit_array = pub_result.data["meas"]
    assert bit_array.shape == (2,)
    assert bit_array.num_shots == 100
    assert bit_array.num_bits == 1

    assert bit_array.get_counts(0) == {"0": 100}
    assert bit_array.get_counts(1) == {"1": 100}


def test_sampler_run_multiple_cregs(sampler: BackendSamplerV2) -> None:
    """Sampler correctly handles multiple classical registers."""
    c0 = ClassicalRegister(2, "c0")
    c1 = ClassicalRegister(1, "c1")
    qc = QuantumCircuit(3)
    qc.add_register(c0)
    qc.add_register(c1)
    qc.x(0)
    qc.x(2)
    qc.measure(0, c0[0])
    qc.measure(1, c0[1])
    qc.measure(2, c1[0])

    job = sampler.run([(qc,)], shots=100)
    result = job.result()

    pub_result = result[0]
    c0_bits = pub_result.data["c0"]
    c1_bits = pub_result.data["c1"]

    assert c0_bits.num_bits == 2
    assert c1_bits.num_bits == 1
    assert c0_bits.get_counts() == {"01": 100}
    assert c1_bits.get_counts() == {"1": 100}


def test_sampler_shot_defaults(sampler: BackendSamplerV2) -> None:
    """Test sampler shot defaults."""
    # 1. Use default shots from init
    sampler2 = BackendSamplerV2(backend=sampler.backend, options={"default_shots": 500})
    qc = QuantumCircuit(1)
    qc.measure_all()

    job = sampler2.run([(qc,)])
    result = job.result()
    assert result[0].metadata["shots"] == 500

    # 2. Override via run method
    job = sampler2.run([(qc,)], shots=200)
    result = job.result()
    assert result[0].metadata["shots"] == 200


def test_backend_constructs_sampler() -> None:
    """A backend constructs a sampler that retains its identity and defaults."""
    backend = QDMIBackend(cast("Device", ShotQDMIDevice()))
    sampler = backend.sampler(options={"default_shots": 37})
    qc = QuantumCircuit(1)
    qc.measure_all()

    assert isinstance(sampler, BackendSamplerV2)
    assert sampler.backend is backend
    assert sampler.run([(qc,)]).result()[0].metadata["shots"] == 37


def test_sampler_no_circuits(sampler: BackendSamplerV2) -> None:
    """Test run with empty pub list."""
    job = sampler.run([])
    result = job.result()
    assert len(result) == 0


def test_sampler_unmeasured_bits_and_mapping(sampler: BackendSamplerV2) -> None:
    """Measure into reordered classical destinations while preserving unmeasured zeros."""
    qc = QuantumCircuit(2)
    qc.add_register(ClassicalRegister(2, "a"), ClassicalRegister(3, "b"))
    qc.x(0)
    qc.measure(0, 3)
    qc.measure(1, 1)
    result = sampler.run([qc], shots=4).result()[0]
    assert result.data["a"].get_bitstrings() == ["00"] * 4
    assert result.data["b"].get_bitstrings() == ["010"] * 4


def test_sampler_broadcasting(sampler: BackendSamplerV2) -> None:
    """Test sampler with parameter broadcasting."""
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.rx(theta, 0)
    qc.measure_all()

    # Broadcast parameters
    params = {theta: np.zeros((2, 2))}
    job = sampler.run([(qc, params)], shots=100)
    result = job.result()

    pub_result = result[0]
    bit_array = pub_result.data["meas"]
    assert bit_array.shape == (2, 2)
    assert bit_array.num_shots == 100
