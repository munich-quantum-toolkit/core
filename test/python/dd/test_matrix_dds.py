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
import numpy.typing as npt
import pytest

from mqt.core.dd import DDPackage


def test_identity() -> None:
    """Test the identity matrix."""
    p = DDPackage(3)
    for i in range(p.max_qubits + 1):
        dd = p.identity()
        assert dd.size() == 1
        mat = dd.get_matrix(i)
        arr = np.array(mat, copy=False)
        assert arr.shape == (2**i, 2**i)
        assert np.allclose(arr, np.eye(2**i))


@pytest.fixture
def gate_matrices() -> dict[str, npt.NDArray[np.complex128]]:
    """Returns a dictionary of single-qubit gate matrices."""
    return {
        "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
        "H": np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2),
        "S": np.array([[1, 0], [0, 1j]], dtype=np.complex128),
        "T": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128),
    }


def test_single_qubit_gate(gate_matrices: dict[str, npt.NDArray[np.complex128]]) -> None:
    """Test constructing single-qubit gate DDs."""
    p = DDPackage(3)
    for i in range(p.max_qubits + 1):
        for gate_mat in gate_matrices.values():
            for j in range(i):
                dd = p.single_qubit_gate(gate_mat, j)
                assert dd.size() == 2
                mat = dd.get_matrix(i)
                arr = np.array(mat, copy=False)
                assert arr.shape == (2**i, 2**i)
                target = gate_mat
                if j > 0:
                    target = np.kron(target, np.eye(2**j))
                if j < i - 1:
                    target = np.kron(np.eye(2 ** (i - j - 1)), target)
                assert np.allclose(arr, target)


def test_controlled_single_qubit_gate(gate_matrices: dict[str, npt.NDArray[np.complex128]]) -> None:
    """Test constructing controlled single-qubit gate DDs."""
    p = DDPackage(5)
    for i in range(p.max_qubits + 1):
        for c in range(i):
            for t in range(c):
                for gate_mat in gate_matrices.values():
                    dd = p.controlled_single_qubit_gate(gate_mat, c, t)
                    assert dd.size() == 3
                    mat = dd.get_matrix(i)
                    arr = np.array(mat, copy=False)
                    assert arr.shape == (2**i, 2**i)
                    target = gate_mat
                    if t > 0:
                        target = np.kron(target, np.eye(2**t))
                    if c - t > 1:
                        target = np.kron(np.eye(2 ** (c - t - 1)), target)
                    target = np.kron(np.array([[0, 0], [0, 1]]), target)
                    target += np.kron(np.array([[1, 0], [0, 0]]), np.eye(2**c))
                    if c < i - 1:
                        target = np.kron(np.eye(2 ** (i - c - 1)), target)
                    assert np.allclose(arr, target)


def test_two_qubit_gate() -> None:
    """Test constructing two-qubit gate DDs."""
    p = DDPackage(5)
    gate_mat = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128)
    for i in range(p.max_qubits + 1):
        for t0 in range(i):
            for t1 in range(t0):
                dd = p.two_qubit_gate(gate_mat, t0, t1)
                assert dd.size() == 6
                mat = dd.get_matrix(i)
                arr = np.array(mat, copy=False)
                assert arr.shape == (2**i, 2**i)


def test_controlled_two_qubit_gate() -> None:
    """Test constructing controlled two-qubit gate DDs."""
    p = DDPackage(5)
    gate_mat = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128)
    for i in range(p.max_qubits + 1):
        for c in range(i):
            for t0 in range(c):
                for t1 in range(t0):
                    dd = p.controlled_two_qubit_gate(gate_mat, c, t0, t1)
                    assert dd.size() == 7
                    mat = dd.get_matrix(i)
                    arr = np.array(mat, copy=False)
                    assert arr.shape == (2**i, 2**i)


def test_from_matrix() -> None:
    """Test constructing a DD from a random unitary matrix."""
    p = DDPackage(3)
    rng = np.random.default_rng(1337)
    for i in range(p.max_qubits + 1):
        for _ in range(10):
            mat = rng.random((2**i, 2**i)) + 1j * rng.random((2**i, 2**i))
            mat = np.linalg.qr(mat)[0]
            dd = p.from_matrix(mat)
            mat2 = dd.get_matrix(i)
            assert np.allclose(mat, mat2)
