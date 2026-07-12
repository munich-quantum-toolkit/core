# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the MLIR compiler Python bindings."""

from __future__ import annotations

import re
from itertools import groupby
from pathlib import Path

import pytest
from qiskit import QuantumCircuit

from mqt.core.ir import QuantumComputation
from mqt.core.mlir import (
    JeffProgram,
    OutputFormat,
    QCOProgram,
    QCProgram,
    QIRProfile,
    QIRProgram,
    compile_program,
)

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


def _sort_constants(text: str) -> str:
    constant_re = re.compile(r"%.* = arith\.constant")
    input_lines = text.splitlines()
    output_lines = []
    for is_constant, group in groupby(input_lines, key=lambda line: bool(constant_re.match(line.strip()))):
        group_lines = list(group)
        output_lines.extend(sorted(group_lines) if is_constant else group_lines)
    return "\n".join(output_lines)


def test_compile_program_jeff_file() -> None:
    """Compile a `.jeff` file."""
    path = Path(__file__).parent.parent / "circuits" / "bell.jeff"

    result = compile_program(path)
    assert isinstance(result, QCProgram)
    assert _sort_constants(result.ir) == _sort_constants(MLIR_STRING)


def test_compile_program_mlir_string() -> None:
    """Compile an MLIR string."""
    result = compile_program(MLIR_STRING)
    assert isinstance(result, QCProgram)
    assert result.ir == MLIR_STRING


def test_compile_program_mlir_file(tmp_path: Path) -> None:
    """Compile a `.mlir` file."""
    path = tmp_path / "program.mlir"
    path.write_text(MLIR_STRING, encoding="utf-8")

    result = compile_program(path)
    assert isinstance(result, QCProgram)
    assert result.ir == MLIR_STRING


def test_compile_program_qasm_string() -> None:
    """Compile an OpenQASM string."""
    result = compile_program(QASM_STRING)
    assert isinstance(result, QCProgram)
    assert result.ir == MLIR_STRING


def test_compile_program_single_line_qasm_string() -> None:
    """Compile a single-line OpenQASM source string."""
    result = compile_program(QASM_STRING.replace("\n", " "))

    assert isinstance(result, QCProgram)
    assert "qc.h" in result.ir


def test_compile_program_qasm_file(tmp_path: Path) -> None:
    """Compile a `.qasm` file."""
    path = tmp_path / "program.qasm"
    path.write_text(QASM_STRING, encoding="utf-8")

    result = compile_program(path)
    assert isinstance(result, QCProgram)
    assert result.ir == MLIR_STRING


def test_compile_program_quantum_computation() -> None:
    """Compile a `QuantumComputation`."""
    qc = QuantumComputation(2, 2)
    qc.h(0)
    qc.cx(0, 1)

    result = compile_program(qc)
    assert isinstance(result, QCProgram)
    assert result.ir == MLIR_STRING


def test_compile_program_qiskit_quantum_circuit() -> None:
    """Compile a `QuantumCircuit`."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    result = compile_program(qc)
    assert isinstance(result, QCProgram)
    assert result.ir == MLIR_STRING


def test_jeff_program_round_trip(tmp_path: Path) -> None:
    """Store and load a `JeffProgram` through bytes and a file."""
    path = tmp_path / "program.jeff"
    result = compile_program(QASM_STRING, output=OutputFormat.JEFF)
    assert isinstance(result, JeffProgram)

    path.write_bytes(result.to_bytes())
    loaded = JeffProgram.from_file(path)
    restored = compile_program(loaded, output=OutputFormat.QC)
    assert isinstance(restored, QCProgram)
    assert _sort_constants(restored.ir) == _sort_constants(MLIR_STRING)


def test_compile_program_jeff_input_runs_from_qco(tmp_path: Path) -> None:
    """Compile a serialized Jeff program through the QCO pipeline entry point."""
    path = tmp_path / "program.jeff"
    compile_program(QASM_STRING, output=OutputFormat.JEFF).write(path)

    result = compile_program(path, output=OutputFormat.QCO)

    assert isinstance(result, QCOProgram)
    assert "qco." in result.ir


def test_program_conversions_are_composable() -> None:
    """Compose frontend, cleanup, conversion, and optimization stages."""
    source = QCProgram.from_qasm_str(QASM_STRING)
    qco = source.to_qco(copy=True)
    assert source.is_valid
    assert isinstance(qco, QCOProgram)

    qco.cleanup()
    qco.optimize()
    result = qco.to_qc()
    assert not qco.is_valid
    result.cleanup()
    assert _sort_constants(result.ir) == _sort_constants(MLIR_STRING)


def test_compile_program_convert_to_qir() -> None:
    """Compile with the QIR Base Profile output format."""
    result = compile_program(QASM_STRING, output=OutputFormat.QIR_BASE)

    assert isinstance(result, QIRProgram)
    assert "; ModuleID" in result.llvm_ir
    assert "@__quantum__qis__h__body" in result.llvm_ir


def test_compile_program_output_format_convert_to_qir() -> None:
    """Lower a QC program directly to the QIR Adaptive Profile."""
    result = QCProgram.from_qasm_str(QASM_STRING).to_qir(QIRProfile.ADAPTIVE)

    assert isinstance(result, QIRProgram)
    assert result.profile == QIRProfile.ADAPTIVE
    assert "@__quantum__qis__h__body" in result.llvm_ir


def test_compile_program_qc_import_output() -> None:
    """Expose QC directly after the frontend translation."""
    result = compile_program(QASM_STRING, output=OutputFormat.QC_IMPORT)

    assert isinstance(result, QCProgram)
    assert "qc.h" in result.ir


def test_compile_program_fails_for_missing_file() -> None:
    """A missing known input file extension raises an error."""
    with pytest.raises(RuntimeError, match="does not exist"):
        compile_program("missing_program.qasm")
