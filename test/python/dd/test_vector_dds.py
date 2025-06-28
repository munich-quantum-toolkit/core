# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for VectorDDs in the MQT Core DD package."""

from __future__ import annotations

import numpy as np

from mqt.core.dd import BasisStates, DDPackage


def test_zero_state() -> None:
    """Test the zero state."""
    p = DDPackage(3)
    for i in range(p.max_qubits + 1):
        dd = p.zero_state(i)
        assert dd.size() == i + 1
        vec = dd.get_vector()
        arr = np.array(vec, copy=False)
        assert arr.shape == (2**i,)
        assert np.allclose(arr, np.array([1] + [0] * (2**i - 1)))
        p.untrack_vec(dd)


def test_computational_basis_state() -> None:
    """Test the computational basis state."""
    p = DDPackage(3)
    for i in range(p.max_qubits + 1):
        for j in range(2**i):
            state = [bool(int(x)) for x in f"{j:0{i}b}"][::-1]
            dd = p.computational_basis_state(i, state)
            assert dd.size() == i + 1
            vec = dd.get_vector()
            arr = np.array(vec, copy=False)
            assert arr.shape == (2**i,)
            assert np.allclose(arr, np.array([0] * j + [1] + [0] * (2**i - j - 1)))
            p.untrack_vec(dd)


def test_plus_state() -> None:
    """Test the basis state."""
    p = DDPackage(3)
    for i in range(p.max_qubits + 1):
        state = [BasisStates.plus] * i
        dd = p.basis_state(i, state)
        assert dd.size() == i + 1
        vec = dd.get_vector()
        arr = np.array(vec, copy=False)
        assert arr.shape == (2**i,)
        assert np.allclose(arr, np.array([(1 / np.sqrt(2) ** i)] * 2**i))
        p.untrack_vec(dd)


def test_ghz_state() -> None:
    """Test the GHZ state."""
    p = DDPackage(3)
    for i in range(1, p.max_qubits + 1):
        dd = p.ghz_state(i)
        assert dd.size() == 2 * i
        vec = dd.get_vector()
        arr = np.array(vec, copy=False)
        assert arr.shape == (2**i,)
        assert np.allclose(arr, np.array([1 / np.sqrt(2)] + [0] * (2**i - 2) + [1 / np.sqrt(2)]))
        p.untrack_vec(dd)


def test_w_state() -> None:
    """Test the W state."""
    p = DDPackage(3)
    for i in range(1, p.max_qubits + 1):
        dd = p.w_state(i)
        assert dd.size() == 2 * i
        vec = dd.get_vector()
        arr = np.array(vec, copy=False)
        assert arr.shape == (2**i,)
        target = np.zeros(2**i)
        for j in range(i):
            target[2**j] = 1 / np.sqrt(i)
        assert np.allclose(arr, target)
        p.untrack_vec(dd)


def test_from_vector() -> None:
    """Test the from_vector method."""
    p = DDPackage(3)
    rng = np.random.default_rng(1337)
    for i in range(p.max_qubits + 1):
        for _ in range(10):
            vec = rng.random(2**i) + 1j * rng.random(2**i)
            vec /= np.linalg.norm(vec)
            dd = p.from_vector(vec)
            vec2 = dd.get_vector()
            assert np.allclose(vec, vec2)
            p.untrack_vec(dd)
