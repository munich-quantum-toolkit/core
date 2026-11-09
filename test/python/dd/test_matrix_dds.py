# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Matrix DD array conversion, ownership, and Python errors."""

from __future__ import annotations

import gc

import numpy as np
import pytest

from mqt.core.dd import Control, DDPackage


def test_matrix_array_ownership() -> None:
    """Test matrix array ownership and the zero-qubit export."""
    package = DDPackage(1)
    zero_qubit_matrix = package.identity().get_matrix(0)
    with pytest.raises(ValueError, match="20-qubit"):
        package.identity().get_matrix(21)
    assert zero_qubit_matrix.dtype == np.complex128
    assert zero_qubit_matrix.shape == (1, 1)
    assert zero_qubit_matrix.flags.c_contiguous
    assert zero_qubit_matrix.flags.writeable  # spellchecker:disable-line
    assert zero_qubit_matrix[0, 0] == 1

    expected = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    matrix_dd = package.single_qubit_gate(expected, 0)
    matrix = matrix_dd.get_matrix(1)
    two_qubit_package = DDPackage(2)
    two_qubit_dd = two_qubit_package.single_qubit_gate(expected, 1)
    for width in (0, 1):
        with pytest.raises(ValueError, match="does not cover"):
            two_qubit_dd.get_matrix(width)
    assert matrix.dtype == np.complex128
    assert matrix.shape == (2, 2)
    assert matrix.flags.c_contiguous
    assert matrix.flags.writeable  # spellchecker:disable-line

    matrix[0, 0] = 2 + 3j
    assert np.allclose(matrix_dd.get_matrix(1), expected)

    view = matrix[:, :1]
    del matrix, matrix_dd, package, zero_qubit_matrix
    gc.collect()

    assert np.allclose(view, [[2 + 3j], [1]])
    view[0, 0] = -1j
    assert np.allclose(view[0, 0], -1j)


def test_gate_construction() -> None:
    """Convert complex gate arrays and controls through the Python bindings."""
    package = DDPackage(3)
    gate = np.array([[1, 1j], [1, -1j]]) / np.sqrt(2)
    swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    np.testing.assert_allclose(package.identity().get_matrix(3), np.eye(8))
    np.testing.assert_allclose(package.single_qubit_gate(gate, 0).get_matrix(2), np.kron(np.eye(2), gate))
    np.testing.assert_allclose(
        package.controlled_single_qubit_gate(gate, 1, 0).get_matrix(2),
        np.block([[np.eye(2), np.zeros((2, 2))], [np.zeros((2, 2)), gate]]),
    )
    np.testing.assert_allclose(package.two_qubit_gate(swap, 1, 0).get_matrix(2), swap)
    np.testing.assert_allclose(
        package.controlled_two_qubit_gate(swap, 2, 1, 0).get_matrix(3),
        np.block([[np.eye(4), np.zeros((4, 4))], [np.zeros((4, 4)), swap]]),
    )


def test_dd_owned_control_type() -> None:
    """Expose controls as DD-owned binding types."""
    p = DDPackage(2)
    x_matrix = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    positive = p.controlled_single_qubit_gate(x_matrix, 1, 0)
    multi_positive = p.multi_controlled_single_qubit_gate(x_matrix, {1}, 0)
    assert np.allclose(positive.get_matrix(2), multi_positive.get_matrix(2))

    control = Control(1, Control.Type.Neg)
    assert control == Control(1, Control.Type.Neg)
    assert len({control, Control(1, Control.Type.Neg)}) == 1
    assert repr(control) == 'Control(qubit=1, type_="Neg")'

    negative = p.controlled_single_qubit_gate(x_matrix, control, 0)
    expected = np.array(
        [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.complex128,
    )
    assert np.allclose(negative.get_matrix(2), expected)


def test_rejects_conflicting_controls() -> None:
    """Reject opposite polarities on the same physical control qubit."""
    package = DDPackage(2)
    matrix = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    controls = {Control(0, Control.Type.Pos), Control(0, Control.Type.Neg)}
    with pytest.raises(RuntimeError, match="duplicate"):
        package.multi_controlled_single_qubit_gate(matrix, controls, 1)


def test_from_strided_matrix() -> None:
    """Read matrix views without losing offsets, negative strides, or broadcasts."""
    package = DDPackage(3)
    values = np.arange(256, dtype=np.float64).reshape(16, 16)
    matrix = values + 1j * (values + 1)
    for view in (
        matrix[1::2, ::2],
        matrix[:8, :8].T,
        matrix[7::-1, 7::-1],
        np.broadcast_to(matrix[0, :8], (8, 8)),
    ):
        assert np.allclose(package.from_matrix(view).get_matrix(3), view)


def test_from_matrix_dimensions() -> None:
    """Validate shape and capacity before reading matrix entries."""
    package = DDPackage(1)
    assert np.array_equal(package.from_matrix(np.empty((0, 0), dtype=np.complex128)).get_matrix(0), [[1]])
    scalar = np.array([[0.25 + 0.5j]])
    assert np.allclose(package.from_matrix(scalar).get_matrix(0), scalar)
    for shape in ((2, 3), (3, 3)):
        with pytest.raises(ValueError, match=r"square|power of two"):
            package.from_matrix(np.zeros(shape, dtype=np.complex128))
    with pytest.raises(RuntimeError, match="capacity"):
        package.from_matrix(np.zeros((4, 4), dtype=np.complex128))


def test_invalid_identity_path() -> None:
    """Validate matrix digits even when all identity levels are implicit."""
    package = DDPackage(1)
    identity = package.identity()
    with pytest.raises(ValueError, match="invalid digit"):
        identity.get_entry_by_path(1, "4")
    assert identity.get_entry_by_path(1, "3ignored") == 1


def test_untracked_release() -> None:
    """Reject an unbalanced release at the Python boundary."""
    package = DDPackage(1)
    gate = package.single_qubit_gate(np.array([[0, 1], [1, 0]], dtype=np.complex128), 0)
    package.inc_ref_mat(gate)
    package.dec_ref_mat(gate)
    with pytest.raises(ValueError, match="Edge is not part of the root set"):
        package.dec_ref_mat(gate)
