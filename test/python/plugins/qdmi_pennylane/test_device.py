# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for modern PennyLane execution through QDMI."""

# ruff: file-ignore[missing-return-type-private-function]

from __future__ import annotations

import math
import sys
from collections import Counter
from typing import cast

import numpy as np
import pytest

if sys.version_info < (3, 11):
    pytest.skip("PennyLane requires Python 3.11 or newer.", allow_module_level=True)

try:
    import pennylane as qp
except ImportError:
    pytest.skip("Install the PennyLane extra to run these tests.", allow_module_level=True)

from mqt.core.plugins.pennylane import (
    PennyLaneConfigurationError,
    PennyLaneUnsupportedFormatError,
    PennyLaneValidationError,
    QDMIDevice,
)
from mqt.core.qdmi import Device as QDMIDeviceHandle
from mqt.core.qdmi import ProgramFormat

from .helpers import StubDevice, rotation_results, stub_device


def _patch_device(monkeypatch: pytest.MonkeyPatch, device: StubDevice) -> None:
    """Route fresh stable-ID opens to a test double."""
    monkeypatch.setattr(
        "mqt.core.plugins.pennylane.device.open_device",
        lambda *_args, **_kwargs: cast("QDMIDeviceHandle", device),
    )


def test_samples_counts_probabilities_expectations_and_variances(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reconstruct common PennyLane result types from raw QDMI samples."""
    qdmi = stub_device()
    _patch_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=["left", "right"], shots=100)

    @qp.qnode(device)
    def circuit() -> tuple[object, ...]:
        return (
            qp.sample(wires=["left", "right"]),
            qp.counts(wires=["left", "right"]),
            qp.probs(wires=["left", "right"]),
            qp.expval(qp.PauliZ("left")),
            qp.var(qp.PauliZ("right")),
        )

    samples, counts, probabilities, expectation, variance = circuit()

    assert samples.shape == (100, 2)
    assert counts == {"00": 50, "11": 50}
    np.testing.assert_allclose(probabilities, [0.5, 0.0, 0.0, 0.5])
    assert expectation == pytest.approx(0.0)
    assert variance == pytest.approx(1.0)
    assert device.submitted_jobs == 1
    assert math.isfinite(device.execution_time)


def test_histogram_only_device_reconstructs_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reconstruct raw samples when a QDMI implementation exposes only counts."""
    qdmi = stub_device(expose_shots=False)
    _patch_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2, shots=8)

    @qp.qnode(device)
    def circuit():
        return qp.sample(wires=[0, 1])

    samples = circuit()

    assert samples.shape == (8, 2)
    assert Counter(map(tuple, samples.tolist())) == {(0, 0): 4, (1, 1): 4}


def test_execution_time_accumulates_one_interval_per_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accumulate one wall-clock interval for every submitted QDMI job."""
    qdmi = stub_device()
    _patch_device(monkeypatch, qdmi)
    readings = iter([0.0, 1.5, 10.0, 12.25, 100.0, 100.5])
    monkeypatch.setattr("mqt.core.plugins.pennylane.device.monotonic", lambda: next(readings))
    device = QDMIDevice("fake.qdmi", wires=2, shots=[(5, 2), 7])

    @qp.qnode(device)
    def circuit():
        return qp.probs(wires=[0, 1])

    circuit()

    assert device.submitted_jobs == 3
    assert device.execution_time == pytest.approx(1.5 + 2.25 + 0.5)


def test_shot_vectors_submit_sequential_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute every shot-vector copy as a sequential QDMI job."""
    qdmi = stub_device()
    _patch_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2, shots=[(5, 2), 7])

    @qp.qnode(device)
    def circuit():
        return qp.probs(wires=[0, 1])

    results = circuit()

    assert len(results) == 3
    assert [submission[2] for submission in qdmi.submissions] == [5, 5, 7]
    for probabilities in results:
        assert np.sum(probabilities) == pytest.approx(1.0)


def test_batches_execute_in_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve batch ordering with one QDMI submission per tape."""
    qdmi = stub_device()
    _patch_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2, shots=6)
    tapes = (
        qp.tape.QuantumScript([qp.PauliX(0)], [qp.probs(wires=[0, 1])], shots=6),
        qp.tape.QuantumScript([qp.PauliX(1)], [qp.probs(wires=[0, 1])], shots=6),
    )

    results = qp.execute(tapes, device, diff_method=None)

    assert len(results) == 2
    assert len(qdmi.submissions) == 2
    assert "x q[0];" in qdmi.submissions[0][0]
    assert "x q[1];" in qdmi.submissions[1][0]


def test_parameter_shift_gradient_uses_multiple_qdmi_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Differentiate sampled execution through PennyLane's parameter-shift rule."""
    qdmi = stub_device(qubits=1, result_factory=rotation_results)
    _patch_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=["theta"], shots=4000)

    @qp.qnode(device, diff_method="parameter-shift")
    def circuit(angle: float):
        qp.RY(angle, wires="theta")
        return qp.expval(qp.PauliZ("theta"))

    angle = qp.numpy.array(0.4, requires_grad=True)
    value = circuit(angle)
    gradient = qp.grad(circuit)(angle)

    assert value == pytest.approx(np.cos(0.4), abs=0.01)
    assert gradient == pytest.approx(-np.sin(0.4), abs=0.02)
    # One explicit value call, then one forward and two shifted tapes for grad.
    assert device.submitted_jobs == 4


def test_hamiltonian_and_non_commuting_measurements_split(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let PennyLane split and aggregate Hamiltonian and non-commuting terms."""
    qdmi = stub_device()
    _patch_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2, shots=100)
    hamiltonian = 0.5 * qp.PauliZ(0) + 0.5 * qp.PauliZ(1)

    @qp.qnode(device)
    def circuit():
        return qp.expval(hamiltonian), qp.expval(qp.PauliX(0))

    energy, x_expectation = circuit()

    assert energy == pytest.approx(0.0)
    assert np.isfinite(x_expectation)
    assert device.submitted_jobs >= 2


def test_qasm2_diagonalizes_observable_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not duplicate the X-basis rotation in PennyLane's QASM2 serializer."""
    qdmi = stub_device(program_format=ProgramFormat.QASM2)
    _patch_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2, shots=10)

    @qp.qnode(device)
    def circuit():
        return qp.expval(qp.PauliX(0))

    assert np.isfinite(circuit())
    assert qdmi.submissions[0][1] == ProgramFormat.QASM2
    assert qdmi.submissions[0][0].count("ry(") == 1


def test_rejects_analytic_execution_before_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject analytic tapes before a QDMI job is created."""
    qdmi = stub_device()
    _patch_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2, shots=None)

    @qp.qnode(device)
    def circuit():
        return qp.expval(qp.PauliZ(0))

    with pytest.raises(PennyLaneValidationError, match="finite number of shots"):
        circuit()
    assert not qdmi.submissions


def test_validates_configuration_and_width(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject unknown QDMI parameters and excessive wire counts."""
    qdmi = stub_device()
    _patch_device(monkeypatch, qdmi)

    with pytest.raises(PennyLaneConfigurationError, match="unknown"):
        QDMIDevice(
            "fake.qdmi",
            wires=2,
            session_parameters={"unknown": "value"},  # ty: ignore[invalid-argument-type, invalid-key]
        )
    with pytest.raises(PennyLaneConfigurationError, match="3 wires"):
        QDMIDevice("fake.qdmi", wires=3)


def test_rejects_device_without_openqasm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject unsupported program formats during construction."""
    qdmi = StubDevice([], [ProgramFormat.QIR_BASE_STRING])
    _patch_device(monkeypatch, qdmi)

    with pytest.raises(PennyLaneUnsupportedFormatError, match="neither OpenQASM 3 nor OpenQASM 2"):
        QDMIDevice("fake.qdmi", wires=2)


def test_forwards_job_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward generic QDMI custom job parameters unchanged."""
    qdmi = stub_device()
    _patch_device(monkeypatch, qdmi)
    device = QDMIDevice(
        "fake.qdmi",
        wires=2,
        shots=4,
        job_parameters={"custom1": "bucket", "custom2": "prefix"},
    )

    @qp.qnode(device)
    def circuit():
        return qp.sample(wires=[0, 1])

    circuit()
    assert qdmi.submissions[0][3] == {"custom1": "bucket", "custom2": "prefix"}
