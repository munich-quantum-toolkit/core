# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for QCO DD Python bindings."""

from __future__ import annotations

import numpy as np
import pytest

dd = pytest.importorskip("mqt.core.dd")
mlir = pytest.importorskip("mqt.core.mlir")


def _x_program() -> mlir.QCOProgram:
    return mlir.QCOProgram.from_mlir_str("""
module {
  func.func @main() {
    %q = qco.static 0 : !qco.qubit
    %q1 = qco.x %q : !qco.qubit -> !qco.qubit
    qco.sink %q1 : !qco.qubit
    return
  }
}
""")


def test_build_functionality_and_simulate_unitary() -> None:
    """X on |0> builds a unitary and simulates to |1>."""
    program = _x_program()
    package = dd.DDPackage(1)
    matrix = mlir.build_functionality(program, package)
    assert matrix is not None
    package.dec_ref_mat(matrix)

    zero = package.zero_state(1)
    out = mlir.simulate(program, zero, package)
    expected = package.computational_basis_state(1, [True])

    assert np.allclose(out.get_vector(), expected.get_vector())
    package.dec_ref_vec(out)
    package.dec_ref_vec(expected)


def test_sample_and_sample_with_classics() -> None:
    """Unitary X samples as all-ones with empty mid-circuit classics."""
    program = _x_program()
    package = dd.DDPackage(1)
    hist = mlir.sample(program, package, shots=32, seed=1)
    assert hist == {"1": 32}

    result = mlir.sample_with_classics(program, package, shots=16, seed=2)
    assert result.shots == {"1": 16}
    assert result.classical == {}


def test_sample_with_classics_records_midcircuit_measure() -> None:
    """Measure then classically controlled X records classical bit '1'."""
    # |1> → measure (bit 1) → if then X → |0>
    program = mlir.QCOProgram.from_mlir_str("""
module {
  func.func @main() {
    %q = qco.static 0 : !qco.qubit
    %q1 = qco.x %q : !qco.qubit -> !qco.qubit
    %q2, %bit = qco.measure %q1 : !qco.qubit
    %q3 = qco.if %bit args(%q_in = %q2) -> (!qco.qubit) {
      %qx = qco.x %q_in : !qco.qubit -> !qco.qubit
      qco.yield %qx : !qco.qubit
    } else args(%q_in = %q2) {
      qco.yield %q_in : !qco.qubit
    }
    qco.sink %q3 : !qco.qubit
    return
  }
}
""")
    package = dd.DDPackage(1)
    result = mlir.sample_with_classics(program, package, shots=20, seed=3)
    assert result.shots == {"0": 20}
    assert result.classical == {"1": 20}
