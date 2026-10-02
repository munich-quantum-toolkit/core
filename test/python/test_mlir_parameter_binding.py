# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Native QC/QCO parameter binding."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator

from mqt.core.mlir import QCProgram


@pytest.mark.parametrize("qco", [False, True])
def test_partial_binding_preserves_identity_and_expression(*, qco: bool) -> None:
    """Binding stays native while preserving remaining Qiskit parameter identity."""
    a, b = Parameter("a"), Parameter("b")
    circuit = QuantumCircuit(2)
    circuit.global_phase = a / 3
    circuit.ry(a.sin() + 2 * b, 0)
    circuit.cx(0, 1)
    program = QCProgram.from_qiskit(circuit)
    if qco:
        program = program.to_qco()
    original = program.copy()
    program.bind_parameters({"a": 0.7})
    assert program.parameters == ["b"]
    assert set(program.to_qiskit().parameters) == {b}
    program.bind_parameters({"b": -0.25})
    assert program.parameters == []
    np.testing.assert_allclose(
        Operator(program.to_qiskit()).data,
        Operator(circuit.assign_parameters({a: 0.7, b: -0.25})).data,
        rtol=0,
        atol=1e-12,
    )
    assert original.parameters == ["a", "b"]


@pytest.mark.parametrize("values", [{"a": 1.0, "unknown": 2.0}, {"a": np.inf}, {"a": np.nan}])
def test_invalid_binding_is_atomic(values: dict[str, float]) -> None:
    """Invalid assignments leave the input reusable."""
    circuit = QuantumCircuit(1)
    circuit.rx(Parameter("a"), 0)
    program = QCProgram.from_qiskit(circuit)
    before = program.ir
    with pytest.raises(ValueError, match=r"unknown f64 parameter|must be finite"):
        program.bind_parameters(values)
    assert program.ir == before
