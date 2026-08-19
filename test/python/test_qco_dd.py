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

from mqt.core.dd import DDPackage
from mqt.core.mlir import OutputFormat, QCOProgram, compile_program


def _x_program() -> QCOProgram:
    """Construct a QCO program that applies X to qubit zero.

    Returns:
        The constructed QCO program.
    """
    return QCOProgram.from_mlir_str("""
module {
  func.func @main() attributes {mqt.entry_point} {
    %q = qco.static 0 : !qco.qubit
    %q1 = qco.x %q : !qco.qubit -> !qco.qubit
    qco.sink %q1 : !qco.qubit
    return
  }
}
""")


def _measure_program() -> QCOProgram:
    """Construct a QCO program with measurement-controlled execution.

    Returns:
        The constructed QCO program.
    """
    return QCOProgram.from_mlir_str("""
module {
  func.func @main() attributes {mqt.entry_point} {
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


def _bound_qtensor_program() -> QCOProgram:
    """Construct a bound RX program over a dynamic QTensor argument.

    Returns:
        The constructed QCO program.
    """
    return QCOProgram.from_mlir_str("""
module {
  func.func @main(%apply: i1, %theta: f64,
                  %input: tensor<?x!qco.qubit>) -> tensor<?x!qco.qubit>
      attributes {mqt.entry_point} {
    %c1 = arith.constant 1 : index
    %remaining, %q = qtensor.extract %input[%c1]
        : tensor<?x!qco.qubit>
    %q1 = qco.if %apply args(%q_in = %q) -> (!qco.qubit) {
      %rotated = qco.rx(%theta) %q_in : !qco.qubit -> !qco.qubit
      qco.yield %rotated : !qco.qubit
    } else args(%q_in = %q) {
      qco.yield %q_in : !qco.qubit
    }
    %result = qtensor.insert %q1 into %remaining[%c1]
        : tensor<?x!qco.qubit>
    return %result : tensor<?x!qco.qubit>
  }
}
""")


def test_unitary_x_build_simulate_and_sample() -> None:
    """X on |0>: unitary matrix, simulation to |1>, deterministic sampling."""
    program = _x_program()
    package = DDPackage(1)
    matrix = program.build_functionality(package)
    package.dec_ref_mat(matrix)

    zero = package.zero_state(1)
    out = program.simulate(zero, package)
    expected = package.computational_basis_state(1, [True])
    assert np.allclose(out.get_vector(), expected.get_vector())
    package.dec_ref_vec(out)
    package.dec_ref_vec(expected)

    assert program.sample(package, shots=32, seed=1) == {"1": 32}


def test_simulate_measure_uses_default_or_explicit_seed() -> None:
    """Simulation supports measurement with default and explicit seeds."""
    program = _measure_program()
    package = DDPackage(1)

    zero = package.zero_state(1)
    out = program.simulate(zero, package)
    expected = package.computational_basis_state(1, [False])
    assert np.allclose(out.get_vector(), expected.get_vector())
    package.dec_ref_vec(out)
    package.dec_ref_vec(expected)

    zero = package.zero_state(1)
    out = program.simulate(zero, package, seed=3)
    expected = package.computational_basis_state(1, [False])
    assert np.allclose(out.get_vector(), expected.get_vector())
    package.dec_ref_vec(out)
    package.dec_ref_vec(expected)


def test_dd_apis_reject_state_from_different_package_without_consuming_it() -> None:
    """DD APIs reject a foreign state without consuming its live reference."""
    program = _x_program()
    source_package = DDPackage(1)
    target_package = DDPackage(1)
    zero = source_package.zero_state(1)
    target_zero = target_package.zero_state(1)

    with pytest.raises(ValueError, match=r"live reference in dd_package"):
        program.simulate(zero, target_package)
    with pytest.raises(ValueError, match=r"live reference in dd_package"):
        program.simulate(zero, target_package, seed=7)

    with pytest.raises(ValueError, match=r"live reference in dd_package"):
        program.sample(target_package, initial_state=zero)

    out = program.simulate(zero, source_package)
    source_package.dec_ref_vec(out)
    target_package.dec_ref_vec(target_zero)


def test_entry_func_required() -> None:
    """Programs without a func.func raise ValueError."""
    # Top-level qco op satisfies dialect checks but provides no entry function.
    program = QCOProgram.from_mlir_str("""
module {
  %theta = arith.constant 0.0 : f64
  qco.gphase(%theta)
}
""")
    package = DDPackage(1)
    with pytest.raises(ValueError, match=r"no func\.func"):
        program.build_functionality(package)


@pytest.mark.parametrize(
    ("source", "num_qubits", "expected"),
    [
        (
            """
OPENQASM 3.0;
include "stdgates.inc";
qubit q0;
qubit q1;
bit[2] c;
h q0;
cx q0, q1;
c[0] = measure q0;
c[1] = measure q1;
""",
            2,
            {"00", "11"},
        ),
        (
            """
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
bit[2] c;
h q;
c[0] = measure q;
if (c[0]) {
  x q;
}
c[1] = measure q;
""",
            1,
            {"00", "01"},
        ),
    ],
    ids=["terminal-bell", "adaptive-reset"],
)
def test_compiler_to_sampler_outputs(source: str, num_qubits: int, expected: set[str]) -> None:
    """Compile optimized QCO and sample the declared CBit output."""
    program = compile_program(source, output=OutputFormat.QCO_OPTIMIZED)
    package = DDPackage(num_qubits)
    shots = 256

    counts = program.sample(package, shots=shots, seed=17)

    assert set(counts) == expected
    assert sum(counts.values()) == shots


def test_symbolic_bindings_across_dd_apis() -> None:
    """All DD APIs accept concrete scalar and dynamic QTensor bindings."""
    program = _bound_qtensor_program()
    package = DDPackage(2)
    bindings = {0: True, 1: float(np.pi), 2: 2}

    matrix = program.build_functionality(package, bindings=bindings)
    package.dec_ref_mat(matrix)

    zero = package.zero_state(2)
    out = program.simulate(zero, package, bindings=bindings)
    expected = package.computational_basis_state(2, [False, True])
    assert np.allclose(np.abs(out.get_vector()), np.abs(expected.get_vector()))
    package.dec_ref_vec(out)
    package.dec_ref_vec(expected)

    assert program.sample(package, shots=8, seed=4, bindings=bindings) == {"10": 8}


@pytest.mark.parametrize(
    "bindings",
    [
        pytest.param({3: 0}, id="argument-index"),
        pytest.param({0: 1}, id="boolean-type"),
        pytest.param({1: 1}, id="float-type"),
        pytest.param({2: -1}, id="negative-extent"),
    ],
)
def test_python_dd_bindings_reject_invalid_values(bindings: dict[int, bool | int | float]) -> None:
    """Binding indices and values must match entry argument types."""
    program = _bound_qtensor_program()
    package = DDPackage(2)
    with pytest.raises(ValueError, match=r"out of range|does not match"):
        program.build_functionality(package, bindings=bindings)


def test_sample_from_supplied_initial_state() -> None:
    """Sampling consumes a valid caller-supplied input state."""
    program = _x_program()
    package = DDPackage(1)

    one = package.computational_basis_state(1, [True])
    assert program.sample(package, shots=8, seed=6, initial_state=one) == {"0": 8}
