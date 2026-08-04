# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""End-to-end PennyLane tests with the DDSIM QDMI device."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

if sys.version_info < (3, 11):
    pytest.skip("PennyLane requires Python 3.11 or newer.", allow_module_level=True)

try:
    import pennylane as qp
except ImportError:
    pytest.skip("Install the PennyLane extra to run these tests.", allow_module_level=True)

from mqt.core.plugins.pennylane import DDSIMDevice

ROOT = Path(__file__).parents[4]
GRAPH_EDGES = ((0, 1), (0, 2), (1, 2), (2, 3))


def test_stable_entry_point_and_wire_order() -> None:
    """Discover the stable device ID and map QDMI bit strings to PennyLane wires."""
    device = qp.device("mqt.ddsim.default", wires=["first", "second"], shots=20)

    @qp.qnode(device)
    def circuit():  # ruff: ignore[missing-return-type-private-function]  # PennyLane replaces measurement return types at runtime.
        qp.PauliX("first")
        return qp.counts(wires=["first", "second"])

    assert isinstance(device, DDSIMDevice)
    assert device.device_id == "mqt.ddsim.default"
    assert circuit() == {"10": 20}


def test_bell_results_and_shot_vector() -> None:
    """Execute probabilities, samples, and shot-vector partitions end to end."""
    device = qp.device("mqt.ddsim.default", wires=2, shots=[(100, 2), 200])

    @qp.qnode(device)
    def circuit():  # ruff: ignore[missing-return-type-private-function]  # PennyLane replaces measurement return types at runtime.
        qp.Hadamard(0)
        qp.CNOT(wires=[0, 1])
        return qp.probs(wires=[0, 1])

    results = circuit()

    assert len(results) == 3
    for probabilities in results:
        assert probabilities[0] == pytest.approx(0.5, abs=0.15)
        assert probabilities[3] == pytest.approx(0.5, abs=0.15)
        assert probabilities[1] + probabilities[2] == pytest.approx(0.0)


def test_parameter_shift_gradient() -> None:
    """Compute a finite sampled parameter-shift gradient through QDMI."""
    device = qp.device("mqt.ddsim.default", wires=1, shots=10_000)

    @qp.qnode(device, diff_method="parameter-shift")
    def circuit(angle: float):  # ruff: ignore[missing-return-type-private-function]  # PennyLane replaces measurement return types at runtime.
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
    """Run the checked-in finite-shot MaxCut application."""
    completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]  # Fixed interpreter and repository-owned script.
        [
            sys.executable,
            str(ROOT / "docs/_scripts/pennylane_qaoa.py"),
            "--shots",
            "200",
            "--steps",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    initial_match = re.search(r"initial cost: ([-+0-9.]+)", completed.stdout)
    gradient_match = re.search(r"initial gradient: (.+)", completed.stdout)
    parameters_match = re.search(r"parameters: (.+)", completed.stdout)
    best = re.search(r"best observed cut: ([01]{4}) \(([0-9]+) edges\)", completed.stdout)
    jobs_match = re.search(r"QDMI jobs: ([0-9]+)", completed.stdout)
    assert initial_match is not None
    assert gradient_match is not None
    assert parameters_match is not None
    assert best is not None
    assert jobs_match is not None

    initial_cost = float(initial_match.group(1))
    gradient = ast.literal_eval(gradient_match.group(1))
    parameters = ast.literal_eval(parameters_match.group(1))
    jobs = int(jobs_match.group(1))

    assert np.isfinite(initial_cost)
    assert np.all(np.isfinite(gradient))
    assert not np.allclose(parameters, (0.5, 0.5))
    assert 0 <= int(best.group(2)) <= len(GRAPH_EDGES)
    assert jobs > 1
