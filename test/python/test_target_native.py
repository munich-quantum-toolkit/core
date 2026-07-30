# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for progressive QCOProgram.target_native / target_device."""

from __future__ import annotations

import pytest
from plugins.qiskit.test_mock_backend import MockQDMIDevice

from mqt.core.ir import QuantumComputation
from mqt.core.mlir import OutputFormat, QCOProgram, compile_program


def test_target_native_menu_only() -> None:
    """Menu-only targeting removes H in favor of native u factors."""
    qc = QuantumComputation(2)
    qc.h(0)
    qc.cx(0, 1)
    qco = compile_program(qc, output=OutputFormat.QCO)
    assert isinstance(qco, QCOProgram)
    qco.target_native(native_gates="u,cx")
    assert "qco.h" not in qco.ir


def test_target_device_with_coupling() -> None:
    """Device-derived menu+coupling lowers CX(0,2) without leftover swaps."""
    device = MockQDMIDevice(
        num_qubits=3,
        operations=["u", "cx"],
        coupling_map=[(0, 1), (1, 2)],
    )
    qc = QuantumComputation(3)
    qc.cx(0, 2)
    qco = compile_program(qc, output=OutputFormat.QCO)
    assert isinstance(qco, QCOProgram)
    qco.target_device(device)
    assert "qco.swap" not in qco.ir
    assert "qco.ctrl" in qco.ir
    # Unrouted CX(0,2) would keep static qubits 0 and 2 on the same ctrl.
    assert "qco.ctrl(%0) targets (%arg0 = %2)" not in qco.ir


def test_target_native_rejects_empty_menu() -> None:
    """Empty native_gates must fail."""
    qc = QuantumComputation(1)
    qc.h(0)
    qco = compile_program(qc, output=OutputFormat.QCO)
    assert isinstance(qco, QCOProgram)
    with pytest.raises(RuntimeError, match=r"(?i)fail|empty|native"):
        qco.target_native(native_gates="")


def test_target_native_rejects_invalid_menu() -> None:
    """Unsupported menus fail before mutating the IR."""
    qc = QuantumComputation(1)
    qc.h(0)
    qco = compile_program(qc, output=OutputFormat.QCO)
    assert isinstance(qco, QCOProgram)
    before = qco.ir
    with pytest.raises(RuntimeError, match=r"(?i)unsupported|native|fail"):
        qco.target_native(native_gates="not-a-gate")
    assert qco.ir == before


def test_target_native_one_way_coupling() -> None:
    """One-direction coupling edges are treated as undirected."""
    qc = QuantumComputation(3)
    qc.cx(0, 2)
    qco = compile_program(qc, output=OutputFormat.QCO)
    assert isinstance(qco, QCOProgram)
    qco.target_native(native_gates="u,cx", coupling=[(0, 1), (1, 2)])
    assert "qco.swap" not in qco.ir
    assert "qco.ctrl" in qco.ir
    assert "qco.ctrl(%0) targets (%arg0 = %2)" not in qco.ir
