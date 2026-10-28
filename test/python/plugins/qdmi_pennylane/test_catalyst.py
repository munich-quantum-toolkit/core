# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exercise the actual Catalyst factory and runtime with local QDMI jobs."""

from __future__ import annotations

from importlib import import_module
from inspect import signature
from re import fullmatch
from typing import TYPE_CHECKING

import numpy as np
import pytest

from .support import supports_catalyst

if TYPE_CHECKING:
    from pennylane.measurements import CountsMP, ExpectationMP, ProbabilityMP, SampleMP, VarianceMP
    from pennylane.typing import Result

if not supports_catalyst():
    pytest.skip("Install Catalyst to run compiled device tests.", allow_module_level=True)

catalyst = import_module("catalyst")
qp = pytest.importorskip("pennylane")

_DECOMPOSITION_OPTIONS = (
    {"collect_decomp_rules": False} if "collect_decomp_rules" in signature(catalyst.qjit).parameters else {}
)


@pytest.mark.parametrize("capture", [False, True])
def test_correlated_readout_and_wire_order(*, capture: bool) -> None:
    """Terminal measurements share a readout, including reordered subsets."""
    device = qp.device("mqt.ddsim.default", wires=3)

    @qp.qnode(device, shots=100)
    def circuit() -> tuple[SampleMP, CountsMP, ProbabilityMP, ExpectationMP, VarianceMP]:
        qp.Hadamard(0)
        qp.CNOT(wires=[0, 2])
        qp.X(1)
        return (
            qp.sample(wires=[1, 2, 0]),
            qp.counts(wires=[1, 0], all_outcomes=True),
            qp.probs(wires=[2, 0]),
            qp.expval(qp.Z(2) @ qp.Z(0)),
            qp.var(qp.Z(0)),
        )

    samples, (keys, counts), probabilities, parity, variance = qp.qjit(
        circuit, capture=capture, skip_preprocess=capture, **_DECOMPOSITION_OPTIONS
    )()
    np.testing.assert_array_equal(samples[:, 0], 1)
    np.testing.assert_array_equal(samples[:, 1], samples[:, 2])
    ones = np.count_nonzero(samples[:, 2])
    np.testing.assert_array_equal(keys, np.arange(4))
    np.testing.assert_array_equal(counts, [0, 0, 100 - ones, ones])
    np.testing.assert_allclose(probabilities, [(100 - ones) / 100, 0, 0, ones / 100])
    assert parity == 1
    assert variance == pytest.approx(1 - (1 - 2 * ones / 100) ** 2)


@pytest.mark.parametrize("capture", [None, False, True])
def test_runtime_gate_sets_match_reference(recwarn: pytest.WarningsRecorder, *, capture: bool | None) -> None:
    """Eager native gates and compiled decompositions have the same action."""
    device = qp.device("mqt.ddsim.default", wires=3)

    def circuit() -> ProbabilityMP:
        qp.SX(0)
        qp.U1(0.7, 0)
        qp.U2(0.4, 0.2, 1)
        qp.U3(0.8, 0.1, -0.4, 2)
        qp.ECR(wires=[2, 0])
        return qp.probs(wires=[2, 0, 1])

    expected = qp.QNode(circuit, qp.device("default.qubit", wires=3))()
    node = qp.QNode(circuit, device, shots=20_000)
    if capture:
        capabilities = qp.devices.DeviceCapabilities.from_toml_file(device.config_filepath, "qjit")
        node = qp.transforms.decompose(node, gate_set=set(capabilities.operations))
    actual = node() if capture is None else qp.qjit(node, capture=capture, skip_preprocess=capture)()
    np.testing.assert_allclose(actual, expected, atol=0.025)
    for warning in recwarn:
        category = import_module("catalyst.decomposition.rule_lowering_warning").RuleLoweringWarning
        assert issubclass(warning.category, category)
        assert fullmatch(
            r"Excluded the .+ decomposition rule for .+ since "
            r"--decompose-lowering cannot work with multiple registers yet",
            str(warning.message),
        )


@pytest.mark.parametrize("capture", [False, True])
def test_repeated_calls_and_observable_lowering(*, capture: bool) -> None:
    """Runtime pooling resets circuits and upstream lowers non-Z observables."""
    device = qp.device("mqt.ddsim.default", wires=2)

    @qp.qnode(device, shots=10)
    def circuit(angle: float) -> ExpectationMP:
        qp.Hadamard(0)
        qp.S(0)
        qp.RY(angle, 1)
        return qp.expval(0.5 * qp.Y(0) + qp.Z(1))

    if capture:
        circuit = qp.transform(
            pass_name="split-non-commuting"  # ruff:ignore[hardcoded-password-func-arg] Compiler pass identifier.
        )(circuit)
        circuit = catalyst.passes.diagonalize_measurements(circuit)

    @qp.qjit(capture=capture, skip_preprocess=capture, **_DECOMPOSITION_OPTIONS)
    def compiled(angle: float) -> tuple[Result, Result, Result]:
        return circuit(angle), circuit(np.pi), circuit(angle)

    np.testing.assert_allclose(compiled(0.0), [1.5, -0.5, 1.5])
    np.testing.assert_allclose(compiled(np.pi), [-0.5, -0.5, -0.5])
    with pytest.raises(RuntimeError, match=r"[Nn]on-finite"):
        compiled(np.nan)
    np.testing.assert_allclose(compiled(0.0), [1.5, -0.5, 1.5])


@pytest.mark.parametrize("probabilities", [False, True])
def test_compiled_parameter_shift(*, probabilities: bool) -> None:
    """Differentiate finite-shot expectations and probabilities."""
    device = qp.device("mqt.ddsim.default", wires=1)

    @qp.qnode(device, shots=10, diff_method="parameter-shift")
    def circuit(angle: float) -> ProbabilityMP | ExpectationMP:
        qp.RY(angle, 0)
        return qp.probs(wires=[0]) if probabilities else qp.expval(qp.Z(0))

    derivative = qp.qjit(catalyst.jacobian(circuit))
    np.testing.assert_allclose(derivative(np.pi / 2), [-0.5, 0.5] if probabilities else -1, atol=1e-12)


@pytest.mark.parametrize("capture", [False, True])
def test_runtime_basis_state_and_unitary(*, capture: bool) -> None:
    """Native preparation and matrix gates preserve runtime data and wire order."""
    device = qp.device("mqt.ddsim.default", wires=3)

    @qp.qnode(device, shots=4)
    def circuit(bits: np.ndarray, matrix: np.ndarray) -> SampleMP:
        qp.BasisState(bits, wires=[2, 0])
        qp.adjoint(qp.QubitUnitary)(matrix, wires=[2, 0])
        qp.ctrl(qp.QubitUnitary, control=0, control_values=[False])(np.array([[0, 1j], [1j, 0]]), wires=1)
        return qp.sample(wires=[0, 1, 2])

    compiled = qp.qjit(circuit, capture=capture, skip_preprocess=capture, **_DECOMPOSITION_OPTIONS)
    matrix = np.eye(4, dtype=complex)[[0, 2, 3, 1]]
    np.testing.assert_array_equal(compiled(np.array([0, 1]), matrix), [[0, 1, 1]] * 4)
    np.testing.assert_array_equal(compiled(np.array([1, 1]), np.eye(4, dtype=complex)), [[1, 0, 1]] * 4)
