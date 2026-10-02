# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Direct native QC program construction."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, random_unitary

from mqt.core.mlir import QCProgramBuilder


def test_native_builder_symbolic_controls_and_phase() -> None:
    """Construct, bind, and export without importing a circuit."""
    builder = QCProgramBuilder(3)
    builder.gate("h", [0]).gate("ry", [1], ["theta"])
    builder.gate("x", [2], controls=[0, 1]).gate("gphase", [], [0.3])
    program = builder.finish()
    assert program.parameters == ["theta"]
    program.bind_parameters({"theta": 0.4})
    expected = QuantumCircuit(3)
    expected.h(0)
    expected.ry(0.4, 1)
    expected.ccx(0, 1, 2)
    expected.global_phase = 0.3
    np.testing.assert_allclose(Operator(program.to_qiskit()).data, Operator(expected).data, rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="finished"):
        builder.gate("x", [0])
    with pytest.raises(ValueError, match="finished"):
        builder.finish()


def test_native_builder_matrix_order_and_measurement() -> None:
    """Dense matrices use Core's most-significant-first target order."""
    matrix = random_unitary(4, seed=19).data
    builder = QCProgramBuilder(3)
    builder.unitary(matrix, [0, 2])
    program = builder.finish()
    expected = QuantumCircuit(3)
    expected.unitary(matrix, [2, 0])
    np.testing.assert_allclose(Operator(program.to_qiskit()).data, Operator(expected).data, rtol=0, atol=1e-12)

    measured = QCProgramBuilder(1, 1).gate("x", [0]).measure(0, 0).reset(0).finish()
    assert measured.to_qco().sample(shots=8, seed=7) == {"1": 8}


def test_invalid_builder_calls_leave_builder_usable() -> None:
    """Reject invalid user input before handing it to the native builder."""
    builder = QCProgramBuilder(2)
    with pytest.raises(IndexError):
        builder.gate("x", [2])
    with pytest.raises(ValueError, match="distinct"):
        builder.gate("x", [0], controls=[0])
    with pytest.raises(ValueError, match="count"):
        builder.gate("ry", [0], [])
    with pytest.raises(ValueError, match="unknown"):
        builder.gate("unknown", [0])
    with pytest.raises(ValueError, match="finite"):
        builder.gate("ry", [0], [np.nan])
    with pytest.raises(ValueError, match="nonempty"):
        builder.gate("ry", [0], [""])
    with pytest.raises(ValueError, match="unitary"):
        builder.unitary(np.zeros((2, 2), dtype=complex), [0])
    with pytest.raises(IndexError):
        builder.measure(0, 0)
    assert builder.gate("x", [0]).finish().num_gates() == 1
