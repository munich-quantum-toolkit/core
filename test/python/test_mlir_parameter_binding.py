# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Native QC/QCO parameter binding."""

from __future__ import annotations

from collections import UserDict

import numpy as np
import numpy.typing as npt
import pytest
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator
from qiskit_support import supports_qiskit_translation

from mqt.core.mlir import QCProgram, sample


@pytest.mark.skipif(not supports_qiskit_translation(), reason=f"No registered Qiskit adapter for {qiskit.__version__}")
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
    assert set(program.to_qiskit(copy=True).parameters) == {b}
    program.bind_parameters({"b": -0.25})
    assert program.parameters == []
    np.testing.assert_allclose(
        Operator(program.to_qiskit()).data,
        Operator(circuit.assign_parameters({a: 0.7, b: -0.25})).data,
        rtol=0,
        atol=1e-12,
    )
    assert original.parameters == ["a", "b"]


@pytest.mark.skipif(not supports_qiskit_translation(), reason=f"No registered Qiskit adapter for {qiskit.__version__}")
def test_scalar_parameters_round_trip_through_openqasm() -> None:
    """Named float inputs connect Qiskit and OpenQASM parameter workflows."""
    theta, phi = Parameter("theta"), Parameter("phi")
    circuit = QuantumCircuit(1)
    circuit.ry(theta + 2 * phi, 0)
    program = QCProgram.from_qiskit(circuit)
    source = program.to_openqasm3().source
    assert "input float[64] theta;" in source
    assert "input float[64] phi;" in source

    restored = QCProgram.from_openqasm_str(source)
    assert set(restored.parameters) == {"theta", "phi"}
    assert {parameter.name for parameter in restored.to_qiskit(copy=True).parameters} == {"theta", "phi"}
    restored.bind_parameters({"theta": 0.7})
    assert restored.parameters == ["phi"]
    restored.bind_parameters({"phi": -0.25})
    np.testing.assert_allclose(
        Operator(restored.to_qiskit()).data,
        Operator(circuit.assign_parameters({theta: 0.7, phi: -0.25})).data,
        rtol=0,
        atol=1e-12,
    )


@pytest.fixture
def array_program() -> QCProgram:
    """Create two rotations with an array input and a scalar offset.

    Returns:
        The parameterized QC program.
    """
    return QCProgram.from_mlir_str("""
    module {
      func.func @main(%angles: tensor<2xf64> {mqt.input_name = "angles"},
                      %offset: f64 {mqt.input_name = "offset"})
          -> !cbit.reg<2> attributes {mqt.entry_point} {
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %first = tensor.extract %angles[%zero] : tensor<2xf64>
        %second = tensor.extract %angles[%one] : tensor<2xf64>
        %shifted = arith.addf %first, %offset : f64
        %q0 = qc.alloc : !qc.qubit
        %q1 = qc.alloc : !qc.qubit
        qc.rx(%shifted) %q0 : !qc.qubit
        qc.ry(%second) %q1 : !qc.qubit
        %b0 = qc.measure %q0 : !qc.qubit -> i1
        %b1 = qc.measure %q1 : !qc.qubit -> i1
        %result = cbit.alloc(#cbit.init<zero>) : !cbit.reg<2>
        cbit.store %b0, %result[%zero] : !cbit.reg<2>
        cbit.store %b1, %result[%one] : !cbit.reg<2>
        qc.dealloc %q0 : !qc.qubit
        qc.dealloc %q1 : !qc.qubit
        return %result : !cbit.reg<2>
      }
    }
    """)


@pytest.mark.parametrize("qco", [False, True], ids=["qc", "qco"])
@pytest.mark.parametrize("as_array", [False, True], ids=["list", "numpy"])
def test_array_binding_preserves_scalar_input(array_program: QCProgram, *, as_array: bool, qco: bool) -> None:
    """Binding one array preserves the scalar input and the resulting circuit."""
    program = array_program.to_qco() if qco else array_program
    original = program.copy()
    angles = [np.pi / 2, np.pi]
    program.bind_parameters(UserDict({"angles": np.asarray(angles) if as_array else angles}))
    assert program.parameters == ["offset"]
    program.bind_parameters({"offset": np.pi / 2})
    assert program.parameters == []
    assert sample(program, shots=4, seed=42) == {"11": 4}
    assert original.parameters == ["angles", "offset"]


@pytest.mark.parametrize(
    "offset",
    [np.longdouble(0.25), np.float64(0.25), np.array(0.25)],
    ids=["numpy-longdouble", "numpy-float64", "numpy-scalar-array"],
)
def test_numpy_scalar_and_single_element_array_binding(
    offset: np.floating | npt.NDArray[np.float64],
) -> None:
    """Bind NumPy scalars and a read-only array with one element."""
    program = QCProgram.from_mlir_str("""
    module {
      func.func @main(%angles: tensor<1xf64> {mqt.input_name = "angles"},
                      %offset: f64 {mqt.input_name = "offset"})
          attributes {mqt.entry_point} {
        return
      }
    }
    """)
    angles = np.array([0.5])
    angles.setflags(write=False)
    program.bind_parameters({"angles": angles, "offset": offset})
    assert program.parameters == []


@pytest.mark.parametrize(
    ("angles", "message"),
    [
        (np.array(0.0), "requires an array"),
        (np.zeros((2, 1)), "must be one-dimensional"),
        (np.zeros((2, 1), dtype=object), "must be one-dimensional"),
        (np.zeros((2, 1), dtype=">f8"), "must be one-dimensional"),
    ],
)
def test_numpy_array_binding_rejects_other_ranks(array_program: QCProgram, angles: npt.NDArray, message: str) -> None:
    """A scalar or matrix cannot silently become a one-dimensional array."""
    before = array_program.ir
    with pytest.raises(ValueError, match=message):
        array_program.bind_parameters({"angles": angles, "offset": 0.25})
    assert array_program.ir == before


@pytest.mark.parametrize("qco", [False, True], ids=["qc", "qco"])
def test_invalid_binding_is_atomic(array_program: QCProgram, *, qco: bool) -> None:
    """Reject a scalar mismatch after a valid array without changing either input."""
    program = array_program.to_qco() if qco else array_program
    before = program.ir
    with pytest.raises(ValueError, match="requires a scalar"):
        program.bind_parameters({"angles": [0.0, 0.0], "offset": [0.0]})
    assert program.ir == before
