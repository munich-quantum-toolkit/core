# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Run a finite-shot MaxCut QAOA application through QDMI."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pennylane as qp

from mqt.core.plugins.pennylane import QDMIDevice

GRAPH = nx.Graph([(0, 1), (0, 2), (1, 2), (2, 3)])
COST_HAMILTONIAN, MIXER_HAMILTONIAN = qp.qaoa.maxcut(GRAPH)


@dataclass(frozen=True)
class QAOAResult:
    """Summary of the sampled QAOA run."""

    initial_cost: float
    initial_gradient: tuple[float, float]
    final_cost: float
    parameters: tuple[float, float]
    best_bitstring: str
    best_cut: int
    jobs: int
    elapsed: float


def cut_value(bitstring: str) -> int:
    """Return the number of graph edges cut by a bit string.

    Returns:
        The MaxCut objective value.
    """
    return sum(bitstring[first] != bitstring[second] for first, second in GRAPH.edges)


def _ansatz(parameters: np.ndarray) -> None:
    """Prepare one QAOA layer."""
    for wire in GRAPH.nodes:
        qp.Hadamard(wire)
    qp.qaoa.cost_layer(parameters[0], COST_HAMILTONIAN)
    qp.qaoa.mixer_layer(parameters[1], MIXER_HAMILTONIAN)


def make_circuits(device: QDMIDevice) -> tuple[qp.QNode, qp.QNode]:
    """Construct cost and sampling circuits for any QDMI PennyLane device.

    Returns:
        The finite-shot cost and sampling QNodes.
    """

    @qp.qnode(device, diff_method="parameter-shift")
    def cost(parameters: np.ndarray) -> float:
        _ansatz(parameters)
        return qp.expval(COST_HAMILTONIAN)

    @qp.qnode(device)
    def sample(parameters: np.ndarray) -> np.ndarray:
        _ansatz(parameters)
        return qp.sample(wires=range(4))

    return cost, sample


def run_qaoa(
    device: QDMIDevice,
    *,
    steps: int = 4,
    step_size: float = 0.15,
) -> QAOAResult:
    """Optimize and sample one QAOA layer on a QDMI device.

    Returns:
        Costs, gradient, parameters, best cut, and execution statistics.
    """
    cost, sample = make_circuits(device)
    parameters = qp.numpy.array([0.5, 0.5], requires_grad=True)
    jobs_before = device.submitted_jobs
    started = time.monotonic()

    initial_cost = float(cost(parameters))
    initial_gradient_array = np.asarray(qp.grad(cost)(parameters), dtype=float)
    optimizer = qp.GradientDescentOptimizer(stepsize=step_size)
    for _ in range(steps):
        parameters = optimizer.step(cost, parameters)

    final_cost = float(cost(parameters))
    samples = np.asarray(sample(parameters), dtype=np.int8)
    observed = {"".join(str(int(bit)) for bit in row) for row in samples}
    best_bitstring = max(observed, key=lambda bitstring: (cut_value(bitstring), bitstring))

    return QAOAResult(
        initial_cost=initial_cost,
        initial_gradient=tuple(float(value) for value in initial_gradient_array),
        final_cost=final_cost,
        parameters=tuple(float(value) for value in parameters),
        best_bitstring=best_bitstring,
        best_cut=cut_value(best_bitstring),
        jobs=device.submitted_jobs - jobs_before,
        elapsed=time.monotonic() - started,
    )


def main() -> None:
    """Run the checked-in application on the local DDSIM QDMI device.

    Raises:
        TypeError: If the stable entry point resolves to an unexpected device.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()

    device = qp.device("mqt.ddsim.default", wires=4, shots=args.shots)
    if not isinstance(device, QDMIDevice):
        msg = "The mqt.ddsim.default entry point did not create a QDMIDevice."
        raise TypeError(msg)
    result = run_qaoa(device, steps=args.steps)

    print(f"initial cost: {result.initial_cost:.6f}")
    print(f"initial gradient: {result.initial_gradient}")
    print(f"parameters: {result.parameters}")
    print(f"final cost: {result.final_cost:.6f}")
    print(f"best observed cut: {result.best_bitstring} ({result.best_cut} edges)")
    print(f"QDMI jobs: {result.jobs}")
    print(f"elapsed: {result.elapsed:.3f} s")


if __name__ == "__main__":
    main()
