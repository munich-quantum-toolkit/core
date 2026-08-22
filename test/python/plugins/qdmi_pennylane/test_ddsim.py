# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""End-to-end PennyLane tests with the DDSIM QDMI device."""

# ruff: file-ignore[missing-return-type-private-function]

from __future__ import annotations

import numpy as np
import pytest

try:
    import pennylane as qp
except ImportError:
    pytest.skip("Install the PennyLane extra to run these tests.", allow_module_level=True)

import networkx as nx

from mqt.core.plugins.pennylane import DDSIMDevice

GRAPH_EDGES = ((0, 1), (0, 2), (1, 2), (2, 3))


def test_stable_entry_point_and_wire_order() -> None:
    """Discover the stable device ID and map QDMI bit strings to PennyLane wires."""
    device = qp.device("mqt.ddsim.default", wires=["first", "second"], shots=20)

    @qp.qnode(device)
    def circuit():
        qp.PauliX("first")
        return qp.counts(wires=["first", "second"])

    assert isinstance(device, DDSIMDevice)
    assert device.device_id == "mqt.ddsim.default"
    assert circuit() == {"10": 20}


def test_bell_results_and_shot_vector() -> None:
    """Execute probabilities, samples, and shot-vector partitions end to end."""
    device = qp.device("mqt.ddsim.default", wires=2, shots=[(1000, 2), 2000])

    @qp.qnode(device)
    def circuit():
        qp.Hadamard(0)
        qp.CNOT(wires=[0, 1])
        return qp.probs(wires=[0, 1])

    results = circuit()

    assert len(results) == 3
    for probabilities in results:
        assert probabilities[0] == pytest.approx(0.5, abs=0.1)
        assert probabilities[3] == pytest.approx(0.5, abs=0.1)
        assert probabilities[1] + probabilities[2] == pytest.approx(0.0)


def test_parameter_shift_gradient() -> None:
    """Compute a finite sampled parameter-shift gradient through QDMI."""
    device = qp.device("mqt.ddsim.default", wires=1, shots=10_000)

    @qp.qnode(device, diff_method="parameter-shift")
    def circuit(angle: float):
        qp.RY(angle, 0)
        return qp.expval(qp.PauliZ(0))

    angle = qp.numpy.array(0.4, requires_grad=True)
    gradient = qp.grad(circuit)(angle)

    assert np.isfinite(gradient)
    assert gradient == pytest.approx(-np.sin(0.4), abs=0.04)
    assert device.submitted_jobs >= 2


@pytest.mark.parametrize(
    "operations",
    [
        [qp.Hadamard(0), qp.PhaseShift(0.43, 0), qp.Hadamard(0)],
        [qp.Hadamard(0), qp.adjoint(qp.S)(0), qp.adjoint(qp.T)(0), qp.Hadamard(0)],
        [qp.SX(0), qp.adjoint(qp.SX)(1), qp.CNOT(wires=[1, 0])],
        [qp.IsingXX(0.37, wires=[0, 1])],
        [qp.IsingYY(0.37, wires=[0, 1])],
        [
            qp.Hadamard(0),
            qp.Hadamard(1),
            qp.IsingZZ(0.37, wires=[0, 1]),
            qp.Hadamard(0),
            qp.Hadamard(1),
        ],
    ],
)
def test_gate_semantics_against_pennylane_reference(operations: list[qp.operation.Operator]) -> None:
    """Match phase, inverse, parameter, and wire-order semantics."""
    qdmi_device = qp.device("mqt.ddsim.default", wires=2, shots=20_000)
    reference_device = qp.device("default.qubit", wires=2)
    sampled_tape = qp.tape.QuantumScript(operations, [qp.probs(wires=[0, 1])], shots=20_000)
    analytic_tape = qp.tape.QuantumScript(operations, [qp.probs(wires=[0, 1])])

    (actual,) = qp.execute((sampled_tape,), qdmi_device, diff_method=None)
    (expected,) = qp.execute((analytic_tape,), reference_device, diff_method=None)

    np.testing.assert_allclose(actual, expected, atol=0.025)


def test_qaoa_application() -> None:
    """Optimize one finite-shot MaxCut QAOA step through the real QDMI device."""
    graph = nx.Graph(GRAPH_EDGES)
    cost_hamiltonian, mixer_hamiltonian = qp.qaoa.maxcut(graph)
    device = qp.device("mqt.ddsim.default", wires=4, shots=200)

    def ansatz(parameters: np.ndarray) -> None:
        for wire in graph.nodes:
            qp.Hadamard(wire)
        qp.qaoa.cost_layer(parameters[0], cost_hamiltonian)
        qp.qaoa.mixer_layer(parameters[1], mixer_hamiltonian)

    @qp.qnode(device, diff_method="parameter-shift")
    def cost(parameters: np.ndarray):
        ansatz(parameters)
        return qp.expval(cost_hamiltonian)

    @qp.qnode(device)
    def sample(parameters: np.ndarray):
        ansatz(parameters)
        return qp.sample(wires=range(4))

    parameters = qp.numpy.array([0.5, 0.5], requires_grad=True)
    initial_cost = float(cost(parameters))
    gradient = np.asarray(qp.grad(cost)(parameters), dtype=float)
    optimized = qp.GradientDescentOptimizer(stepsize=0.15).step(cost, parameters)
    samples = np.asarray(sample(optimized), dtype=np.int8)
    bitstrings = {"".join(str(int(bit)) for bit in row) for row in samples}
    cuts = [sum(bitstring[first] != bitstring[second] for first, second in GRAPH_EDGES) for bitstring in bitstrings]

    assert np.isfinite(initial_cost)
    assert np.all(np.isfinite(gradient))
    assert not np.allclose(optimized, parameters)
    assert samples.shape == (200, 4)
    assert bitstrings
    assert all(len(bitstring) == 4 and set(bitstring) <= {"0", "1"} for bitstring in bitstrings)
    assert all(0 <= cut <= len(GRAPH_EDGES) for cut in cuts)
    assert device.submitted_jobs > 1
