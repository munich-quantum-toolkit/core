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


def _measure_program() -> mlir.QCOProgram:
    return mlir.QCOProgram.from_mlir_str("""
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


def _symbolic_rx_program() -> mlir.QCOProgram:
    return mlir.QCOProgram.from_mlir_str("""
module {
  func.func @main(%theta: f64) {
    %q = qco.static 0 : !qco.qubit
    %q1 = qco.rx(%theta) %q : !qco.qubit -> !qco.qubit
    qco.sink %q1 : !qco.qubit
    return
  }
}
""")


def _dynamic_qtensor_program() -> mlir.QCOProgram:
    return mlir.QCOProgram.from_mlir_str("""
module {
  func.func @main(%arg0: tensor<?x!qco.qubit>) -> tensor<?x!qco.qubit> {
    %c1 = arith.constant 1 : index
    %remaining, %q = qtensor.extract %arg0[%c1]
        : tensor<?x!qco.qubit>
    %q1 = qco.x %q : !qco.qubit -> !qco.qubit
    %result = qtensor.insert %q1 into %remaining[%c1]
        : tensor<?x!qco.qubit>
    return %result : tensor<?x!qco.qubit>
  }
}
""")


def test_unitary_x_build_simulate_and_sample() -> None:
    """X on |0>: unitary matrix, simulation to |1>, deterministic sampling."""
    program = _x_program()
    package = dd.DDPackage(1)
    matrix = mlir.build_functionality(program, package)
    package.dec_ref_mat(matrix)

    zero = package.zero_state(1)
    out = mlir.simulate(program, zero, package)
    expected = package.computational_basis_state(1, [True])
    assert np.allclose(out.get_vector(), expected.get_vector())
    package.dec_ref_vec(out)
    package.dec_ref_vec(expected)

    assert mlir.sample(program, package, shots=32, seed=1) == {"1": 32}
    result = mlir.sample_with_classics(program, package, shots=16, seed=2)
    assert result.shots == {"1": 16}
    assert result.classical == {}


def test_simulate_measure_requires_seed() -> None:
    """Simulate without seed rejects measure/reset; with seed it succeeds."""
    program = _measure_program()
    package = dd.DDPackage(1)

    zero = package.zero_state(1)
    with pytest.raises(ValueError, match=r"cannot simulate|measure"):
        mlir.simulate(program, zero, package)

    zero = package.zero_state(1)
    out = mlir.simulate(program, zero, package, seed=3)
    expected = package.computational_basis_state(1, [False])
    assert np.allclose(out.get_vector(), expected.get_vector())
    package.dec_ref_vec(out)
    package.dec_ref_vec(expected)


def test_entry_func_required() -> None:
    """Programs without a func.func raise ValueError via entryFunc."""
    # Top-level qco op satisfies dialect checks but provides no entry function.
    program = mlir.QCOProgram.from_mlir_str("""
module {
  %theta = arith.constant 0.0 : f64
  qco.gphase(%theta)
}
""")
    package = dd.DDPackage(1)
    with pytest.raises(ValueError, match=r"no func\.func"):
        mlir.build_functionality(program, package)


def test_sample_with_classics_records_midcircuit_measure() -> None:
    """Measure then classically controlled X records classical bit '1'."""
    program = _measure_program()
    package = dd.DDPackage(1)
    result = mlir.sample_with_classics(program, package, shots=20, seed=3)
    assert result.shots == {"0": 20}
    assert result.classical == {"1": 20}


def test_symbolic_and_dynamic_qtensor_bindings() -> None:
    """Python bindings supply scalar parameters and dynamic qtensor extents."""
    symbolic = _symbolic_rx_program()
    package = dd.DDPackage(1)
    matrix = mlir.build_functionality(symbolic, package, bindings={0: float(np.pi)})
    package.dec_ref_mat(matrix)
    assert mlir.sample(symbolic, package, shots=16, seed=4, bindings={0: float(np.pi)}) == {"1": 16}

    tensor = _dynamic_qtensor_program()
    tensor_package = dd.DDPackage(2)
    matrix = mlir.build_functionality(tensor, tensor_package, bindings={0: 2})
    tensor_package.dec_ref_mat(matrix)
    assert mlir.sample(tensor, tensor_package, shots=8, seed=5, bindings={0: 2}) == {"10": 8}


def test_python_dd_bindings_reject_invalid_values() -> None:
    """Binding indices and Python values must match entry argument types."""
    symbolic = _symbolic_rx_program()
    package = dd.DDPackage(1)
    with pytest.raises(ValueError, match="index is out of range"):
        mlir.build_functionality(symbolic, package, bindings={1: 0.5})
    with pytest.raises(ValueError, match="does not match"):
        mlir.build_functionality(symbolic, package, bindings={0: 1})

    tensor = _dynamic_qtensor_program()
    tensor_package = dd.DDPackage(2)
    with pytest.raises(ValueError, match="does not match"):
        mlir.build_functionality(tensor, tensor_package, bindings={0: -1})


def test_sample_from_supplied_initial_state() -> None:
    """Both Python sampling APIs accept and consume an existing input state."""
    program = _x_program()
    package = dd.DDPackage(1)

    one = package.computational_basis_state(1, [True])
    assert mlir.sample(program, package, shots=16, seed=6, initial_state=one) == {"0": 16}

    one = package.computational_basis_state(1, [True])
    result = mlir.sample_with_classics(program, package, shots=16, seed=7, initial_state=one)
    assert result.shots == {"0": 16}
    assert result.classical == {}
