# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent numerical and V2 contract checks for the exact DD estimator."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.quantum_info import SparsePauliOp, Statevector

from mqt.core.mlir import expectation_values
from mqt.core.plugins.qiskit import DDEstimator
from mqt.core.plugins.qiskit import estimator as estimator_module

if TYPE_CHECKING:
    from qiskit.primitives.containers import EstimatorPubLike


def test_exact_estimator_dense_oracle_and_phase() -> None:
    """Signed Y terms, wire order and global phase match an independent engine."""
    circuit = QuantumCircuit(4, global_phase=0.37)
    circuit.ry(0.43, 0)
    circuit.s(0)
    circuit.rx(-0.27, 2)
    circuit.cx(0, 2)
    circuit.x(3)
    observables = [
        SparsePauliOp(["IXIY", "IZIX", "ZIII"], np.array([0.7, -0.4, 0.2])),
        SparsePauliOp(["IYIX", "IXIY"], np.array([-0.3, 0.9])),
        SparsePauliOp("IIII"),
    ]
    expected = [Statevector(circuit).expectation_value(observable).real for observable in observables]
    result = DDEstimator().run([(circuit, observables)]).result()[0]
    np.testing.assert_allclose(result.data["evs"], expected, atol=1e-12)
    np.testing.assert_array_equal(result.data["stds"], np.zeros(3))
    assert result.data.shape == (3,)
    circuit.global_phase = 1.23
    np.testing.assert_allclose(
        DDEstimator().run([(circuit, observables)]).result()[0].data["evs"], expected, atol=1e-12
    )


def test_exact_estimator_broadcast_parameter_order_and_preparation_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """One state per binding preserves broadcast order."""
    z, a = Parameter("z"), Parameter("a")
    circuit = QuantumCircuit(2)
    circuit.ry(z, 0)
    circuit.rx(a, 1)
    assert list(circuit.parameters) == [a, z]
    values = np.array([[[0.1, 0.2], [0.3, 0.7]], [[1.2, -0.4], [0.8, 1.1]]])
    observables = [[[SparsePauliOp("IZ")]], [[SparsePauliOp("ZI")]]]  # spellchecker:disable-line
    evaluate = Mock(wraps=expectation_values)
    monkeypatch.setattr(estimator_module, "expectation_values", evaluate)
    # Qiskit's PUB type alias omits supported unkeyed parameter arrays.
    pub = cast("EstimatorPubLike", (circuit, observables, values))
    result = DDEstimator().run([pub]).result()[0]
    expected = np.stack([np.cos(values[..., 1]), np.cos(values[..., 0])])
    assert result.data.shape == (2, 2, 2)
    np.testing.assert_allclose(result.data["evs"], expected, atol=1e-12)
    assert evaluate.call_count == 4


def test_exact_estimator_parameter_vectors_and_keyed_expressions() -> None:
    """Keyed vector parameters bind expressions in circuit parameter order."""
    theta = ParameterVector("theta", 2)
    circuit = QuantumCircuit(1)
    circuit.global_phase = theta[1]
    circuit.ry(theta[0].sin() + 2 * theta[1] ** 2, 0)
    values = {(theta[1], theta[0]): np.array([[0.2, 0.1], [0.4, -0.3]])}
    result = DDEstimator().run([(circuit, "Z", values)]).result()[0]
    np.testing.assert_allclose(
        result.data["evs"], np.cos([np.sin(0.1) + 2 * 0.2**2, np.sin(-0.3) + 2 * 0.4**2]), atol=1e-12
    )


def test_exact_estimator_precision_precedence_and_metadata() -> None:
    """Positive precision stays exact, with explicit method and no shots claim."""
    circuit = QuantumCircuit(1, metadata={"purpose": "precision"})
    circuit.ry(0.41, 0)
    estimator = DDEstimator(default_precision=0.3)
    assert estimator.default_precision == pytest.approx(0.3)
    pubs = [(circuit, "Z"), EstimatorPub.coerce((circuit, "Z"), precision=0.1)]
    result = estimator.run(pubs, precision=0.2).result()
    assert result.metadata == {"version": 2}
    for index, precision in enumerate([0.2, 0.1]):
        pub_result = result[index]
        assert pub_result.data.shape == ()
        np.testing.assert_allclose(pub_result.data["evs"], np.cos(0.41), atol=1e-12)
        np.testing.assert_array_equal(pub_result.data["stds"], 0)
        assert pub_result.metadata == {
            "method": "exact",
            "target_precision": precision,
            "circuit_metadata": circuit.metadata,
        }
    assert estimator.run([(circuit, "Z")]).result()[0].metadata["target_precision"] == pytest.approx(0.3)
    assert len(estimator.run([]).result()) == 0
    assert estimator.run([(circuit, [])]).result()[0].data.shape == (0,)


@pytest.mark.parametrize("precision", [-1.0, np.nan])
def test_exact_estimator_rejects_invalid_precision(precision: float) -> None:
    """Every precision boundary rejects negative or nonfinite values."""
    with pytest.raises(ValueError, match="precision"):
        DDEstimator(default_precision=precision)
    with pytest.raises(ValueError, match="precision"):
        DDEstimator().run([(QuantumCircuit(1), "Z")], precision=precision)


@pytest.mark.parametrize("value", [np.nan])
def test_exact_estimator_rejects_nonfinite_parameters(value: float) -> None:
    """Invalid parameter values fail before asynchronous execution."""
    circuit = QuantumCircuit(1)
    circuit.ry(Parameter("angle"), 0)
    with pytest.raises(ValueError, match="finite"):
        DDEstimator().run([(circuit, "Z", {"angle": [value]})])


def test_exact_estimator_terminal_readout_and_dynamic_rejection() -> None:
    """Terminal measurements preserve coherence; feedback and reset do not."""
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    result = DDEstimator().run([(circuit, "X")]).result()[0]
    np.testing.assert_allclose(result.data["evs"], 1, atol=1e-12)
    with circuit.if_test((circuit.clbits[0], True)):
        circuit.x(0)
    with pytest.raises(ValueError, match=r"statevector|measurement|terminal"):
        DDEstimator().run([(circuit, "Z")]).result()
    circuit = QuantumCircuit(1)
    circuit.reset(0)
    with pytest.raises(ValueError, match=r"statevector|reset|terminal"):
        DDEstimator().run([(circuit, "Z")]).result()


def test_exact_estimator_compact_64_qubit_batch() -> None:
    """Wide GHZ expectations avoid exponential dense state extraction."""
    circuit = QuantumCircuit(64)
    circuit.h(0)
    for qubit in range(1, 64):
        circuit.cx(0, qubit)
    result = DDEstimator().run([(circuit, ["X" * 64, "Z" + "I" * 62 + "Z"])]).result()[0]
    np.testing.assert_allclose(result.data["evs"], [1, 1], atol=1e-12)


def test_exact_estimator_binding_preserves_qubit_width() -> None:
    """Canceling bound rotations retain the observable's qubit space."""
    angle = Parameter("angle")
    circuit = QuantumCircuit(1)
    circuit.rx(angle, 0)
    circuit.rx(-angle, 0)
    result = DDEstimator().run([(circuit, "Z", {angle: [0.0, 0.3]})]).result()[0]
    np.testing.assert_allclose(result.data["evs"], [1, 1], atol=1e-12)
