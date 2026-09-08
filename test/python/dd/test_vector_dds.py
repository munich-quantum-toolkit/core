# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for VectorDDs in the MQT Core DD package."""

from __future__ import annotations

import gc

import numpy as np
import pytest

from mqt.core.dd import BasisStates, DDPackage, VectorDD


def test_vector_array_ownership() -> None:
    """Test vector array ownership and write access."""
    package = DDPackage(2)
    vector_dd = package.zero_state(2)
    vector = vector_dd.get_vector()

    assert vector.dtype == np.complex128
    assert vector.shape == (4,)
    assert vector.flags.c_contiguous
    assert vector.flags.writeable  # spellchecker:disable-line

    vector[0] = 2 + 3j
    assert np.allclose(vector_dd.get_vector(), [1, 0, 0, 0])

    view = vector[::2]
    package.dec_ref_vec(vector_dd)
    del vector, vector_dd, package
    gc.collect()

    assert np.allclose(view, [2 + 3j, 0])
    view[0] = -1j
    assert np.allclose(view[0], -1j)


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
        p.dec_ref_vec(dd)


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
            p.dec_ref_vec(dd)


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
        p.dec_ref_vec(dd)


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
        p.dec_ref_vec(dd)


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
        p.dec_ref_vec(dd)


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
            p.dec_ref_vec(dd)


def test_from_strided_vector() -> None:
    """Preserve offsets, negative strides, and read-only broadcast amplitudes."""
    package = DDPackage(3)
    values = np.arange(16, dtype=np.float64)
    vector = values + 1j * (values + 1)
    for view in (vector[1::2], vector[7::-1], np.broadcast_to(vector[3], (8,))):
        state = package.from_vector(view)
        package.garbage_collect(force=True)
        assert np.allclose(state.get_vector(), view)
        package.dec_ref_vec(state)


def test_from_vector_dimensions() -> None:
    """Reject oversized vectors and retain scalar states across collection."""
    package = DDPackage(1)
    for length in (3, 4, 8):
        with pytest.raises(ValueError, match=r"power of two|capacity"):
            package.from_vector(np.zeros(length, dtype=np.complex128))
    empty_package = DDPackage(0)
    with pytest.raises(ValueError, match="capacity"):
        empty_package.from_vector(np.zeros(2, dtype=np.complex128))
    for vector in (np.empty(0, dtype=np.complex128), np.array([0.25 + 0.5j])):
        state = empty_package.from_vector(vector)
        empty_package.garbage_collect(force=True)
        expected = vector if vector.size else np.array([1])
        assert np.allclose(state.get_vector(), expected)
        empty_package.dec_ref_vec(state)


@pytest.mark.parametrize("binary", [False, True])
def test_serialization(*, binary: bool) -> None:
    """Test serializing and deserializing vector DDs."""
    p = DDPackage(3)
    for dd in (p.zero_state(0), p.ghz_state(3)):
        data = dd.to_bytes(binary=binary)
        assert isinstance(data, bytes)

        restored = VectorDD.from_bytes(DDPackage(3), data, binary=binary)
        assert np.allclose(restored.get_vector(), dd.get_vector())
        p.dec_ref_vec(dd)


def test_measurement_rejects_missing_qubits() -> None:
    """Reject qubits outside the state even when they fit in the package."""
    package = DDPackage(4)
    for width in (0, 2):
        state = package.zero_state(width)
        before = state.get_vector().copy()
        with pytest.raises(ValueError, match="outside the state"):
            package.measure_collapsing(state, width)
        assert np.array_equal(state.get_vector(), before)
        package.dec_ref_vec(state)
