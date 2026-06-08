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

pytest.importorskip("mlir.ir")
pytest.importorskip("mqt.core.mlir")

from mlir.ir import Module
from mqt.core.mlir import (
    CompilationResult,
    MQTContext,
    compile_program,
    compile_qasm,
    compile_with_record,
    load_qasm,
    qc_to_qco,
)

BELL = """\
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0], q[1];
"""

SINGLE_QUBIT = """\
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
h q[0];
t q[0];
h q[0];
"""


def test_load_qasm_bell_produces_qc_dialect() -> None:
    """load_qasm on the Bell circuit must produce QC dialect operations."""
    result = load_qasm(BELL)
    assert "qc.h" in result
    assert "qc.cx" in result


def test_load_qasm_single_qubit_produces_qc_dialect() -> None:
    """load_qasm on the single-qubit circuit must produce QC dialect operations."""
    result = load_qasm(SINGLE_QUBIT)
    assert "qc.h" in result


def test_load_qasm_invalid_raises() -> None:
    """load_qasm must raise on invalid QASM input."""
    with pytest.raises(RuntimeError):
        load_qasm("not qasm")


def test_qc_to_qco_bell_produces_qco_dialect() -> None:
    """qc_to_qco on the Bell circuit must produce QCO dialect operations."""
    result = qc_to_qco(load_qasm(BELL))
    assert "qco." in result


def test_compile_program_bell() -> None:
    """compile_program on the Bell circuit must return a str containing 'module'."""
    result = compile_program(BELL)
    assert isinstance(result, str)
    assert "module" in result


def test_compile_program_single_qubit() -> None:
    """compile_program on the single-qubit circuit must return a str containing 'module'."""
    result = compile_program(SINGLE_QUBIT)
    assert isinstance(result, str)
    assert "module" in result


def test_compile_program_invalid_raises() -> None:
    """compile_program must raise on invalid QASM input."""
    with pytest.raises(RuntimeError):
        compile_program("not valid qasm")


def test_compile_qasm_intermediates_all_stages_present() -> None:
    """compile_qasm with capture_intermediates must return all 11 non-empty stage keys."""
    result = compile_qasm(BELL, capture_intermediates=True)
    expected = {
        "result",
        "after_qc_import",
        "after_initial_canon",
        "after_qco_conversion",
        "after_qco_canon",
        "after_optimization",
        "after_optimization_canon",
        "after_qc_conversion",
        "after_qc_canon",
        "after_qir_conversion",
        "after_qir_canon",
    }
    assert expected == set(result.keys())
    for key, value in result.items():
        assert isinstance(value, str), f"stage {key!r} is not a str"
        # QIR stages are empty when convert_to_qir is False; all others must be non-empty.
        if key not in {"after_qir_conversion", "after_qir_canon"}:
            assert value.strip(), f"stage {key!r} is unexpectedly empty"


def test_compile_with_record_bell() -> None:
    """compile_with_record on the Bell circuit must produce module strings at key stages."""
    rec = compile_with_record(BELL)
    assert isinstance(rec, CompilationResult)
    assert "module" in rec.result
    assert "module" in rec.after_qc_import
    assert "module" in rec.after_qco_conversion


def test_mqt_context_registers_dialects() -> None:
    """A module parsed inside MQTContext must be valid (dialects are registered)."""
    qc_ir = load_qasm(BELL)
    with MQTContext() as ctx:
        module = Module.parse(qc_ir, ctx)
        assert module is not None
        assert module.operation.verify()
