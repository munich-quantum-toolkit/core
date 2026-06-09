# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the MQT MLIR Python bindings."""

from __future__ import annotations

import pytest

pytest.importorskip("mqt.core.mlir", reason="mqt-core built without MLIR support")

from mqt.core.mlir import compile_program, convert_qc_to_qco, load_qasm

BELL = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0], q[1];
"""

GHZ = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[0];
cx q[0], q[1];
cx q[0], q[2];
"""


def test_load_qasm_returns_qc_dialect() -> None:
    """Load a Bell circuit and check that the output is in the QC dialect."""
    result = load_qasm(BELL)
    assert "func.func" in result
    assert "qc." in result


def test_load_qasm_ghz() -> None:
    """Load a GHZ circuit and verify the QC dialect output."""
    result = load_qasm(GHZ)
    assert "func.func" in result
    assert "qc." in result


def test_convert_qc_to_qco_produces_qco_dialect() -> None:
    """Convert a QC module to QCO and verify the dialect changed."""
    qc = load_qasm(BELL)
    qco = convert_qc_to_qco(qc)
    assert "qco." in qco
    assert "qc." not in qco


def test_compile_program_full_pipeline() -> None:
    """Run the full compiler pipeline on a Bell circuit."""
    result = compile_program(BELL)
    assert "func.func" in result


def test_compile_program_convert_to_qir() -> None:
    """Run the full pipeline including QIR lowering."""
    result = compile_program(BELL, convert_to_qir=True)
    assert "func.func" in result


def test_pipeline_is_composable() -> None:
    """Verify that load_qasm and convert_qc_to_qco can be chained."""
    qc = load_qasm(GHZ)
    qco = convert_qc_to_qco(qc)
    assert isinstance(qc, str)
    assert isinstance(qco, str)
    assert len(qco) > 0


def test_load_qasm_invalid_raises() -> None:
    """Invalid QASM input should raise a RuntimeError."""
    with pytest.raises(RuntimeError):
        load_qasm("this is not valid qasm")
