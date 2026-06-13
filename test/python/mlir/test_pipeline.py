# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the MQT Compiler Collection Python bindings."""

from __future__ import annotations

import pytest

# The MLIR bindings require MQT Core to be built with the bindings and MLIR
# enabled, and the MLIR Python runtime to be importable. Skip otherwise.
pytest.importorskip("mlir.ir")
mqt_mlir = pytest.importorskip("mqt.core.mlir")

BELL_QASM = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c[0] = measure q[0];
c[1] = measure q[1];
"""


def test_translate_to_qc_produces_qc_module() -> None:
    """``translate_to_qc`` translates an OpenQASM 3 program to a QC-dialect module."""
    program = mqt_mlir.read_qasm(BELL_QASM)
    module = mqt_mlir.translate_to_qc(program)
    ir = str(module)
    assert "qc." in ir
    assert "qc.h" in ir


def test_transform_to_qco_produces_qco_module() -> None:
    """``transform_to_qco`` lowers a QC-dialect module to the QCO dialect."""
    qc_module = mqt_mlir.translate_to_qc(BELL_QASM)
    qco_module = mqt_mlir.transform_to_qco(qc_module)
    assert "qco." in str(qco_module)


def test_run_pipeline_round_trips_qc_to_qco_to_qc() -> None:
    """``run_pipeline`` runs arbitrary MQT passes; QC -> QCO -> QC round-trips."""
    module = mqt_mlir.translate_to_qc(BELL_QASM)
    mqt_mlir.run_pipeline(module, "builtin.module(qc-to-qco)")
    assert "qco." in str(module)
    mqt_mlir.run_pipeline(module, "builtin.module(qco-to-qc)")
    assert "qc." in str(module)


def test_run_pipeline_rejects_unknown_pass() -> None:
    """An unknown pass name in a pipeline raises an error."""
    module = mqt_mlir.translate_to_qc(BELL_QASM)
    with pytest.raises((ValueError, RuntimeError)):
        mqt_mlir.run_pipeline(module, "builtin.module(this-pass-does-not-exist)")


def test_pipeline_shares_context() -> None:
    """The QASM -> QC -> QCO pipeline runs end to end in one context."""
    context = mqt_mlir.create_context()
    program = mqt_mlir.read_qasm(BELL_QASM)
    qc_module = mqt_mlir.translate_to_qc(program, context=context)
    qco_module = mqt_mlir.transform_to_qco(qc_module)
    assert "qco." in str(qco_module)
