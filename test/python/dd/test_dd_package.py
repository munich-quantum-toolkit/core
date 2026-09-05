# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the MQT Core DD package."""

from __future__ import annotations

import sys

import numpy as np

from mqt.core.dd import DDPackage, build_functionality, build_unitary, sample, simulate
from mqt.core.ir import QuantumComputation
from mqt.core.ir.operations import OpType


def test_sample_simple_circuit() -> None:
    """Test sampling a simple circuit."""
    qc = QuantumComputation(2)
    qc.x(0)
    qc.measure_all()

    shots = 1000
    results = sample(qc, shots)
    assert results == {"01": shots}


def test_sample_dynamic_circuit() -> None:
    """Test sampling a dynamic circuit."""
    qc = QuantumComputation(1, 1)
    # put the qubit into superposition
    qc.h(0)
    # reset the qubit
    qc.measure(0, 0)
    qc.if_(OpType.x, target=0, control_bit=0)
    # flip to |1>
    qc.x(0)
    # measure the qubit
    qc.measure(0, 0)

    shots = 1000
    results = sample(qc, shots)
    assert results == {"1": shots}


def test_build_unitary_and_functionality_simple_circuit() -> None:
    """Test building the unitary and functionality for a simple circuit."""
    qc = QuantumComputation(2)
    qc.h(0)
    qc.cx(0, 1)

    p = DDPackage(2)
    package_refcount = sys.getrefcount(p)
    functionality = build_functionality(qc, p)
    try:
        assert sys.getrefcount(p) == package_refcount + 1
        mat = functionality.get_matrix(2)
        arr = np.array(mat, copy=False)
        assert arr.shape == (4, 4)
        assert np.allclose(arr, np.array([[1, 1, 0, 0], [0, 0, 1, -1], [0, 0, 1, 1], [1, -1, 0, 0]]) / np.sqrt(2))
        assert np.allclose(np.array(build_unitary(qc), copy=False), arr)
    finally:
        p.dec_ref_mat(functionality)


def test_simulate_simple_circuit() -> None:
    """Test simulating a simple circuit."""
    qc = QuantumComputation(2)
    qc.h(0)
    qc.cx(0, 1)

    p = DDPackage(2)
    in_state = p.zero_state(2)

    out_state = simulate(qc, in_state, p)
    vec = out_state.get_vector()
    arr = np.array(vec, copy=False)
    assert arr.shape == (4,)
    assert np.allclose(arr, np.array([1, 0, 0, 1]) / np.sqrt(2))
