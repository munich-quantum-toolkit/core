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
    """Test that loading a Bell circuit produces a QC dialect MLIR module."""
    result = load_qasm(BELL)
    print(result)

    assert "func.func" in result
    assert "qc.h" in result


def test_load_qasm_ghz() -> None:
    """Test that loading a GHZ circuit produces a QC dialect MLIR module."""
    result = load_qasm(GHZ)
    print(result)

    assert "func.func" in result
    assert "qc." in result


def test_convert_qc_to_qco_produces_qco_dialect() -> None:
    """Test that converting a QC module produces a QCO dialect module."""
    qc = load_qasm(BELL)
    qco = convert_qc_to_qco(qc)
    print(qco)

    assert "qco." in qco
    assert "qc." not in qco


def test_compile_program_full_pipeline() -> None:
    """Test that the full compiler pipeline runs without error."""
    result = compile_program(BELL)
    print(result)

    assert "func.func" in result


def test_compile_program_convert_to_qir() -> None:
    """Test the full pipeline with QIR lowering enabled."""
    result = compile_program(BELL, convert_to_qir=True)
    print(result)

    assert "func.func" in result


def test_pipeline_is_composable() -> None:
    """Test that load_qasm and convert_qc_to_qco can be chained."""
    qc = load_qasm(GHZ)
    qco = convert_qc_to_qco(qc)

    assert isinstance(qc, str)
    assert isinstance(qco, str)
    assert len(qco) > 0


def test_load_qasm_invalid_raises() -> None:
    """Test that invalid QASM input raises a RuntimeError."""
    with pytest.raises(RuntimeError, match="failed to translate"):
        load_qasm("this is not valid qasm")
