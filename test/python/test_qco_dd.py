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
    """Construct a QCO program that applies X to qubit zero.

    Returns:
        The constructed QCO program.
    """
    return mlir.QCOProgram.from_mlir_str("""
module {
  func.func @main() attributes {mqt.entry_point} {
    %q = qco.static 0 : !qco.qubit
    %q1 = qco.x %q : !qco.qubit -> !qco.qubit
    qco.sink %q1 : !qco.qubit
    return
  }
}
""")


def _measure_program() -> mlir.QCOProgram:
    """Construct a QCO program with measurement-controlled execution.

    Returns:
        The constructed QCO program.
    """
    return mlir.QCOProgram.from_mlir_str("""
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


def test_simulate_measure_requires_seed() -> None:
    """Simulate without seed rejects measure/reset; with seed it succeeds."""
    program = _measure_program()
    package = dd.DDPackage(1)

    zero = package.zero_state(1)
    with pytest.raises(ValueError, match=r"measurements require simulate\(\.\.\., rng\)"):
        mlir.simulate(program, zero, package)

    zero = package.zero_state(1)
    out = mlir.simulate(program, zero, package, seed=3)
    expected = package.computational_basis_state(1, [False])
    assert np.allclose(out.get_vector(), expected.get_vector())
    package.dec_ref_vec(out)
    package.dec_ref_vec(expected)


def test_simulate_rejects_state_from_different_package() -> None:
    """Simulation rejects a state owned by a different DD package."""
    program = _x_program()
    source_package = dd.DDPackage(1)
    target_package = dd.DDPackage(1)
    zero = source_package.zero_state(1)
    target_zero = target_package.zero_state(1)

    for seed in (None, 7):
        with pytest.raises(ValueError, match=r"live reference in dd_package"):
            mlir.simulate(program, zero, target_package, seed=seed)

    source_package.dec_ref_vec(zero)
    target_package.dec_ref_vec(target_zero)


def test_entry_func_required() -> None:
    """Programs without a func.func raise ValueError."""
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
    program = mlir.compile_program(source, output=mlir.OutputFormat.QCO_OPTIMIZED)
    package = dd.DDPackage(num_qubits)
    shots = 256

    counts = mlir.sample(program, package, shots=shots, seed=17)

    assert set(counts) == expected
    assert sum(counts.values()) == shots
