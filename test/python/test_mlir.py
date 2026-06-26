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


def test_compile_program_from_quantum_computation() -> None:
    """Compile a `QuantumComputation` object."""
    quantum_computation = QuantumComputation(2, 2)
    quantum_computation.h(0)
    quantum_computation.cx(0, 1)

    result = compile_program(quantum_computation)

    assert "module" in result
    assert "func.func" in result


def test_compile_program_from_qasm_string() -> None:
    """Compile an OpenQASM string."""
    qasm = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
"""

    result = compile_program(qasm)

    assert "module" in result
    assert "func.func" in result


def test_compile_program_from_qasm_file(tmp_path: Path) -> None:
    """Compile a `.qasm` file."""
    program_path = tmp_path / "program.qasm"
    program_path.write_text(
        """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
""",
        encoding="utf-8",
    )

    result = compile_program(program_path)

    assert "module" in result
    assert "func.func" in result


def test_compile_program_from_mlir_string() -> None:
    """Compile from textual MLIR input."""
    qasm = """OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
h q[0];
"""
    qc_mlir = compile_program(qasm)

    result = compile_program(qc_mlir)

    assert "module" in result
    assert "func.func" in result


def test_compile_program_from_qiskit_circuit() -> None:
    """Compile from a Qiskit `QuantumCircuit`."""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)

    result = compile_program(circuit)

    assert "module" in result
    assert "func.func" in result


def test_compile_program_convert_to_qir() -> None:
    """Compile with QIR lowering enabled."""
    qasm = """OPENQASM 3.0;
include "stdgates.inc";
qubit[1] q;
h q[0];
"""

    result = compile_program(qasm, convert_to_qir_base=True)

    assert "module" in result
    assert "llvm." in result


def test_compile_program_fails_for_missing_file() -> None:
    """A missing known input file extension raises an error."""
    with pytest.raises(RuntimeError, match="does not exist"):
        _ = compile_program("missing_program.qasm")
