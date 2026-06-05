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

from mqt.core.mlir import (
    CompilationResult,
    MQTContext,
    compile,
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


# ---------------------------------------------------------------------------
# MQTContext
# ---------------------------------------------------------------------------


def test_mqt_context_is_mlir_context() -> None:
    from mlir.ir import Context

    assert isinstance(MQTContext(), Context)


def test_mqt_context_usable_as_context_manager() -> None:
    with MQTContext():
        pass


def test_mqt_context_allows_module_parse() -> None:
    from mlir.ir import Module

    qc_ir = load_qasm(BELL)
    with MQTContext() as ctx:
        module = Module.parse(qc_ir, ctx)
        assert module is not None


# ---------------------------------------------------------------------------
# load_qasm — Stage 1: QASM → QC dialect
# ---------------------------------------------------------------------------


def test_load_qasm_returns_string() -> None:
    assert isinstance(load_qasm(BELL), str)


def test_load_qasm_output_is_mlir_module() -> None:
    ir = load_qasm(BELL)
    assert ir.strip().startswith("module")


def test_load_qasm_does_not_contain_openqasm() -> None:
    assert "OPENQASM" not in load_qasm(BELL)


def test_load_qasm_single_qubit() -> None:
    ir = load_qasm(SINGLE_QUBIT)
    assert "module" in ir


def test_load_qasm_invalid_raises() -> None:
    with pytest.raises(Exception):  # noqa: B017
        load_qasm("not qasm")


# ---------------------------------------------------------------------------
# qc_to_qco — Stage 2: QC dialect → QCO dialect
# ---------------------------------------------------------------------------


def test_qc_to_qco_returns_string() -> None:
    qc_ir = load_qasm(BELL)
    assert isinstance(qc_to_qco(qc_ir), str)


def test_qc_to_qco_output_is_module() -> None:
    result = qc_to_qco(load_qasm(BELL))
    assert "module" in result


def test_qc_to_qco_chained_from_load_qasm() -> None:
    qc_ir = load_qasm(SINGLE_QUBIT)
    qco_ir = qc_to_qco(qc_ir)
    assert isinstance(qco_ir, str)
    assert "module" in qco_ir


# ---------------------------------------------------------------------------
# compile — full pipeline
# ---------------------------------------------------------------------------


def test_compile_returns_string_by_default() -> None:
    result = compile(BELL)
    assert isinstance(result, str)
    assert "module" in result


def test_compile_bell_circuit() -> None:
    result = compile(BELL)
    assert "OPENQASM" not in result


def test_compile_single_qubit() -> None:
    result = compile(SINGLE_QUBIT)
    assert isinstance(result, str)


def test_compile_hadamard_lifting() -> None:
    result = compile(BELL, enable_hadamard_lifting=True)
    assert isinstance(result, str)
    assert "module" in result


def test_compile_disable_rotation_merging() -> None:
    result = compile(
        SINGLE_QUBIT, disable_merge_single_qubit_rotation_gates=True
    )
    assert isinstance(result, str)
    assert "module" in result


def test_compile_capture_intermediates_returns_dict() -> None:
    result = compile(BELL, capture_intermediates=True)
    assert isinstance(result, dict)


def test_compile_intermediates_has_result_key() -> None:
    result = compile(BELL, capture_intermediates=True)
    assert "result" in result
    assert "module" in result["result"]


def test_compile_intermediates_all_stage_keys_present() -> None:
    result = compile(BELL, capture_intermediates=True)
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


def test_compile_invalid_qasm_raises() -> None:
    with pytest.raises(Exception):  # noqa: B017
        compile("not valid qasm")


# ---------------------------------------------------------------------------
# compile_with_record — typed CompilationResult
# ---------------------------------------------------------------------------


def test_compile_with_record_returns_compilation_result() -> None:
    result = compile_with_record(BELL)
    assert isinstance(result, CompilationResult)


def test_compile_with_record_result_is_mlir() -> None:
    rec = compile_with_record(BELL)
    assert "module" in rec.result


def test_compile_with_record_qc_import_non_empty() -> None:
    rec = compile_with_record(BELL)
    assert rec.after_qc_import.strip() != ""


def test_compile_with_record_all_fields_are_strings() -> None:
    rec = compile_with_record(BELL)
    for fname in CompilationResult.__dataclass_fields__:
        assert isinstance(getattr(rec, fname), str), f"field {fname!r} not str"


def test_compile_with_record_single_qubit() -> None:
    rec = compile_with_record(SINGLE_QUBIT)
    assert isinstance(rec, CompilationResult)
    assert "module" in rec.result


def test_compile_with_record_hadamard_lifting() -> None:
    rec = compile_with_record(BELL, enable_hadamard_lifting=True)
    assert isinstance(rec, CompilationResult)
