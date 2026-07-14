# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the MLIR compiler Python bindings."""

from __future__ import annotations

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
bit[2] c = measure q;
"""


def _assert_bell_program(program: QCProgram, *, measured: bool = False) -> None:
    """Check the semantics of a translated Bell-state program."""
    assert program.is_valid
    ir = program.ir
    assert "memref<2x!qc.qubit>" in ir
    assert ir.count("qc.h ") == 1
    assert ir.count("qc.ctrl(") == 1
    assert ir.count("qc.x ") == 1

    if not measured:
        assert "qc.measure" not in ir
        assert "func.func @main() -> i64" in ir
        return

    assert ir.count("qc.measure") == 2
    assert "func.func @main() -> (i1, i1)" in ir
    assert ": i1, i1" in ir


def test_compile_program_jeff_file() -> None:
    """Compile a `.jeff` file."""
    path = Path(__file__).parent.parent / "circuits" / "bell.jeff"

    result = compile_program(path)
    assert isinstance(result, QCProgram)
    _assert_bell_program(result)


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
    _assert_bell_program(result, measured=True)


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
    _assert_bell_program(result, measured=True)


def test_compile_program_quantum_computation() -> None:
    """Compile a `QuantumComputation`."""
    qc = QuantumComputation(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(range(2), range(2))

    result = compile_program(qc)
    assert isinstance(result, QCProgram)
    _assert_bell_program(result, measured=True)


def test_compile_program_qiskit_quantum_circuit() -> None:
    """Compile a `QuantumCircuit`."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(range(2), range(2))

    result = compile_program(qc)
    assert isinstance(result, QCProgram)
    _assert_bell_program(result, measured=True)


def test_jeff_program_round_trip(tmp_path: Path) -> None:
    """Store and load a `JeffProgram` through bytes and a file."""
    path = tmp_path / "program.jeff"
    result = compile_program(QASM_STRING, output=OutputFormat.JEFF)
    assert isinstance(result, JeffProgram)

    path.write_bytes(result.to_bytes())
    loaded = JeffProgram.from_file(path)
    restored = compile_program(loaded, output=OutputFormat.QC)
    assert isinstance(restored, QCProgram)
    _assert_bell_program(restored, measured=True)


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
    qco.merge_single_qubit_rotation_gates()
    result = qco.to_qc()
    assert not qco.is_valid
    result.cleanup()
    _assert_bell_program(result, measured=True)


def test_compile_program_convert_to_qir() -> None:
    """Compile with the QIR Base Profile output format."""
    result = compile_program(QASM_STRING, output=OutputFormat.QIR_BASE)

    assert isinstance(result, QIRProgram)
    assert "; ModuleID" in result.llvm_ir
    assert "@__quantum__qis__h__body" in result.llvm_ir
    bitcode = result.to_bitcode()
    assert bitcode.startswith(b"BC\xc0\xde")


def test_qir_program_writes_bitcode(tmp_path: Path) -> None:
    """Write generated LLVM bitcode to a file."""
    result = compile_program(QASM_STRING, output=OutputFormat.QIR_BASE)
    path = tmp_path / "program.bc"

    result.write_bitcode(path)

    assert path.read_bytes() == result.to_bitcode()


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


def test_compile_program_exposes_raw_and_optimized_qco() -> None:
    """Expose QCO before and after the configured optimization pipeline."""
    qasm = QASM_STRING.replace("h q[0];", "rz(1.0) q[0];\nrx(1.0) q[0];")

    raw = compile_program(qasm, output=OutputFormat.QCO)
    optimized = compile_program(qasm, output=OutputFormat.QCO_OPTIMIZED)

    assert isinstance(raw, QCOProgram)
    assert isinstance(optimized, QCOProgram)
    assert raw.ir != optimized.ir


def test_qco_program_runs_textual_pipeline() -> None:
    """Run registered QCO passes through MLIR textual pipeline syntax."""
    qco = compile_program(QASM_STRING, output=OutputFormat.QCO)
    assert isinstance(qco, QCOProgram)

    qco.run_pass_pipeline("mqt-qco-default")
    qco.lift_hadamards()

    with pytest.raises(RuntimeError, match="MLIR operation failed"):
        qco.run_pass_pipeline("not-a-pass")


def test_compile_program_fails_for_missing_file() -> None:
    """A missing known input file extension raises an error."""
    with pytest.raises(RuntimeError, match="does not exist"):
        compile_program("missing_program.qasm")
