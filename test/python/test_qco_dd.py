# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for QCO DD Python bindings."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import qiskit
from packaging import version
from qiskit import QuantumCircuit

from mqt.core.dd import DDPackage
from mqt.core.mlir import (
    JeffProgram,
    OpenQASMProgram,
    OutputFormat,
    QCOProgram,
    QCProgram,
    build_functionality,
    compile_program,
    sample,
    simulate,
)

requires_qiskit_translation = pytest.mark.skipif(
    not (
        version.parse("2.5") <= version.parse(qiskit.__version__) < version.parse("2.6")
        or qiskit.__version__ == os.environ.get("MQT_QISKIT_TEST_CANDIDATE_VERSION")
    ),
    reason=f"no Qiskit translation is registered for {qiskit.__version__}",
)

UNITARY_QASM = """OPENQASM 3.0;
include "stdgates.inc";
qubit q;
x q;
"""

CompilerInput = str | Path | QuantumCircuit | QCProgram | QCOProgram | JeffProgram | OpenQASMProgram


def _compiler_input(kind: str, tmp_path: Path) -> CompilerInput:
    """Construct each compiler input supported by the simulation helpers.

    Returns:
        The requested compiler input.
    """
    if kind == "source":
        return UNITARY_QASM
    if kind == "path":
        path = tmp_path / "program.qasm"
        path.write_text(UNITARY_QASM, encoding="utf-8")
        return path
    if kind == "qc":
        return QCProgram.from_qasm_str(UNITARY_QASM)
    if kind == "qco":
        return QCProgram.from_qasm_str(UNITARY_QASM).to_qco()
    if kind == "jeff":
        return compile_program(UNITARY_QASM, output=OutputFormat.JEFF)
    if kind == "openqasm":
        return QCProgram.from_qasm_str(UNITARY_QASM).to_openqasm3()
    circuit = QuantumCircuit(1)
    circuit.x(0)
    return circuit


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

    assert program.sample(shots=32) == {"1": 32}


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


def test_simulate_rejects_state_from_different_package() -> None:
    """Simulation rejects a state owned by a different DD package."""
    program = _x_program()
    source_package = DDPackage(1)
    target_package = DDPackage(1)
    zero = source_package.zero_state(1)
    target_zero = target_package.zero_state(1)

    with pytest.raises(ValueError, match=r"live reference in dd_package"):
        program.simulate(zero, target_package)
    with pytest.raises(ValueError, match=r"live reference in dd_package"):
        program.simulate(zero, target_package, seed=7)

    source_package.dec_ref_vec(zero)
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
    with pytest.raises(ValueError, match=r"no func\.func"):
        build_functionality(program, package)


@pytest.mark.parametrize(
    "kind",
    [
        "source",
        "path",
        "qc",
        "qco",
        "jeff",
        "openqasm",
        pytest.param("qiskit", marks=requires_qiskit_translation),
    ],
)
def test_sample_accepts_compiler_inputs(kind: str, tmp_path: Path) -> None:
    """Sample each input form through its direct QCO conversion path."""
    program = _compiler_input(kind, tmp_path)

    assert sample(program, shots=8, seed=7) == {"1": 8}


def test_build_and_simulate_accept_source() -> None:
    """Build and simulate a source program through the convenience API."""
    package = DDPackage(1)
    matrix = build_functionality(UNITARY_QASM, package)
    assert np.allclose(matrix.get_matrix(1), [[0, 1], [1, 0]])
    package.dec_ref_mat(matrix)

    # OpenQASM declares and allocates its own qubit, so the incoming state is empty.
    zero = package.zero_state(0)
    out = simulate(UNITARY_QASM, zero, package, seed=7)
    expected_state = package.computational_basis_state(1, [True])
    assert np.allclose(out.get_vector(), expected_state.get_vector())
    package.dec_ref_vec(out)
    package.dec_ref_vec(expected_state)


def test_dense_build_and_simulate_hide_dd_package() -> None:
    """Dense overloads own the DD package and return NumPy arrays."""
    matrix = build_functionality(_x_program())
    assert isinstance(matrix, np.ndarray)
    assert matrix.dtype == np.complex128
    assert matrix.flags.c_contiguous
    assert matrix.base is not None
    assert np.allclose(matrix, [[0, 1], [1, 0]])

    initial_state = np.array([1, 0], dtype=np.complex128)
    initial_state.flags.writeable = False  # spellchecker:disable-line
    state = simulate(_x_program(), initial_state, seed=7)
    assert isinstance(state, np.ndarray)
    assert state.dtype == np.complex128
    assert state.flags.c_contiguous
    assert state.base is not None
    assert np.allclose(state, [0, 1])

    # Compiler frontends allocate qubits from a scalar state and retain its phase.
    assert np.allclose(simulate(UNITARY_QASM, [1j], seed=7), [0, 1j])


def test_dense_build_handles_zero_qubits_and_size_limit() -> None:
    """Dense functionality handles scalars and rejects impractical matrices."""
    empty = QCOProgram.from_mlir_str("""
module {
  func.func @main() attributes {mqt.entry_point} {
    %theta = arith.constant 0.0 : f64
    qco.gphase(%theta)
    return
  }
}
""")
    assert np.array_equal(build_functionality(empty), [[1]])

    too_large = QCOProgram.from_mlir_str("""
module {
  func.func @main() attributes {mqt.entry_point} {
    %q = qco.static 20 : !qco.qubit
    qco.sink %q : !qco.qubit
    return
  }
}
""")
    with pytest.raises(ValueError, match=r"practical limit of 20"):
        build_functionality(too_large)


@pytest.mark.parametrize("initial_state", [[], [1, 0, 0], [[1, 0]]])
def test_dense_simulate_rejects_invalid_shape(initial_state: list[int] | list[list[int]]) -> None:
    """Dense simulation requires a one-dimensional power-of-two state."""
    with pytest.raises(ValueError, match=r"initial_state"):
        simulate(_x_program(), initial_state)


@pytest.mark.parametrize(
    ("source", "expected"),
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
            {"00", "01"},
        ),
        (
            """
OPENQASM 3.0;
include "stdgates.inc";
qubit q;
h q;
bit repeat = measure q;
while (repeat) { h q; repeat = measure q; }
output bit out;
out = measure q;
""",
            {"0"},
        ),
    ],
    ids=["terminal-bell", "adaptive-reset", "while-reset"],
)
def test_compiler_to_sampler_outputs(source: str, expected: set[str]) -> None:
    """Compile optimized QCO and sample the declared CBit output."""
    program = compile_program(source, output=OutputFormat.QCO_OPTIMIZED)
    shots = 256

    counts = program.sample(shots=shots, seed=17)

    assert set(counts) == expected
    assert sum(counts.values()) == shots
    assert program.sample(shots=shots, seed=17) == counts
