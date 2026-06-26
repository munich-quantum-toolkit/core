# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the MLIR compiler Python bindings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from qiskit import QuantumCircuit

from mqt.core.ir import QuantumComputation
from mqt.core.mlir import compile_program

if TYPE_CHECKING:
    from pathlib import Path

MLIR_STRING = r"""module {
  func.func @main() -> i64 attributes {passthrough = ["entry_point"]} {
    %c0_i64 = arith.constant 0 : i64
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<2x!qc.qubit>
    %0 = memref.load %alloc[%c0] : memref<2x!qc.qubit>
    %1 = memref.load %alloc[%c1] : memref<2x!qc.qubit>
    qc.h %0 : !qc.qubit
    qc.ctrl(%0) targets (%arg0 = %1) {
      qc.x %arg0 : !qc.qubit
      qc.yield
    } : {!qc.qubit}, {!qc.qubit}
    memref.dealloc %alloc : memref<2x!qc.qubit>
    return %c0_i64 : i64
  }
}
"""

QASM_STRING = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
"""


def test_compile_program_mlir_string() -> None:
    """Compile an MLIR string."""
    result = compile_program(MLIR_STRING)

    assert "module" in result
    assert "func.func" in result


def test_compile_program_mlir_file(tmp_path: Path) -> None:
    """Compile a `.mlir` file."""
    path = tmp_path / "program.mlir"
    path.write_text(MLIR_STRING, encoding="utf-8")

    result = compile_program(path)

    assert "module" in result
    assert "func.func" in result


def test_compile_program_qasm_string() -> None:
    """Compile an OpenQASM string."""
    result = compile_program(QASM_STRING)

    assert "module" in result
    assert "func.func" in result


def test_compile_program_qasm_file(tmp_path: Path) -> None:
    """Compile a `.qasm` file."""
    path = tmp_path / "program.qasm"
    path.write_text(QASM_STRING, encoding="utf-8")

    result = compile_program(path)

    assert "module" in result
    assert "func.func" in result


def test_compile_program_quantum_computation() -> None:
    """Compile a `QuantumComputation`."""
    qc = QuantumComputation(2, 2)
    qc.h(0)
    qc.cx(0, 1)

    result = compile_program(qc)

    assert "module" in result
    assert "func.func" in result


def test_compile_program_qiskit_quantum_circuit() -> None:
    """Compile a `QuantumCircuit`."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    result = compile_program(qc)

    assert "module" in result
    assert "func.func" in result


def test_compile_program_convert_to_qir() -> None:
    """Compile with `convert_to_qir_base` enabled."""
    result = compile_program(QASM_STRING, convert_to_qir_base=True)

    assert "module" in result
    assert "llvm." in result


def test_compile_program_fails_for_missing_file() -> None:
    """A missing known input file extension raises an error."""
    with pytest.raises(RuntimeError, match="does not exist"):
        compile_program("missing_program.qasm")
