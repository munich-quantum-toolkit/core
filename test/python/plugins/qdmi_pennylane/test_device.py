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
from collections import Counter
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

try:
    import pennylane as qp
except ImportError:
    pytest.skip("Install the PennyLane extra to run these tests.", allow_module_level=True)

from mqt.core.plugins.pennylane import (
    PennyLaneConfigurationError,
    PennyLaneExecutionError,
    PennyLaneUnsupportedFormatError,
    PennyLaneValidationError,
    QDMIDevice,
)
from mqt.core.qdmi import Device as QDMIDeviceHandle
from mqt.core.qdmi import Job as QDMIJobHandle
from mqt.core.qdmi import ProgramFormat

from .helpers import StubDevice, operation, patch_open_device, rotation_results, stub_device


def test_uses_already_open_qdmi_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reuse a device session selected outside the PennyLane adapter."""
    qdmi = stub_device()
    monkeypatch.setattr(
        "mqt.core.plugins.pennylane.device.open_device",
        lambda *_args, **_kwargs: pytest.fail("The adapter reopened the QDMI device."),
    )

    device = QDMIDevice(device=cast("QDMIDeviceHandle", qdmi), wires=2, shots=10)

    assert device.qdmi_device is qdmi
    assert device.device_id is None


def test_samples_counts_probabilities_expectations_and_variances(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reconstruct common PennyLane result types from raw QDMI samples."""
    qdmi = stub_device()
    patch_open_device(monkeypatch, qdmi)
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
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2, shots=8)

    @qp.qnode(device)
    def circuit():
        return qp.sample(wires=[0, 1])

    samples = circuit()

    assert samples.shape == (8, 2)
    assert Counter(map(tuple, samples.tolist())) == {(0, 0): 4, (1, 1): 4}


def test_execution_time_accumulates_batch_wall_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Count overlapping job execution once for each PennyLane batch."""
    qdmi = stub_device()
    patch_open_device(monkeypatch, qdmi)
    readings = iter([0.0, 1.5, 10.0, 12.25])
    monkeypatch.setattr("mqt.core.plugins.pennylane.device.monotonic", lambda: next(readings))
    device = QDMIDevice("fake.qdmi", wires=2, shots=[(5, 2), 7])

    @qp.qnode(device)
    def circuit():
        return qp.probs(wires=[0, 1])

    circuit()
    circuit()

    assert device.submitted_jobs == 6
    assert device.execution_time == pytest.approx(1.5 + 2.25)


def test_shot_vectors_submit_before_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Submit every shot-vector copy before waiting for the first job."""
    qdmi = stub_device()
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2, shots=[(5, 2), 7])

    @qp.qnode(device)
    def circuit():
        return qp.probs(wires=[0, 1])

    results = circuit()

    assert len(results) == 3
    assert [submission[2] for submission in qdmi.submissions] == [5, 5, 7]
    assert qdmi.events == ["submit:1", "submit:2", "submit:3", "wait:1", "wait:2", "wait:3"]
    for probabilities in results:
        assert probabilities.shape == (4,)
        assert np.sum(probabilities) == pytest.approx(1.0)


def test_batches_execute_in_input_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve batch ordering with one QDMI submission per tape."""

    def basis_state_results(program: str, shots: int) -> list[str]:
        return ["01" if "x q[0];" in program else "10"] * shots

    qdmi = stub_device(result_factory=basis_state_results)
    patch_open_device(monkeypatch, qdmi)
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
    assert qdmi.events == ["submit:1", "submit:2", "wait:1", "wait:2"]
    np.testing.assert_equal(results, ([0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]))


@pytest.mark.parametrize("cancel_error", [RuntimeError, ValueError, KeyboardInterrupt])
def test_execution_failure_cancels_submitted_jobs(
    monkeypatch: pytest.MonkeyPatch, cancel_error: type[BaseException]
) -> None:
    """Cancel every submitted job without masking the execution failure."""
    qdmi = stub_device()
    submit_job = qdmi.submit_job

    def fail_wait() -> bool:
        msg = "wait failed"
        raise RuntimeError(msg)

    def fail_cancel() -> None:
        qdmi.events.append("cancel:1")
        msg = "cancel failed"
        raise cancel_error(msg)

    def submit(
        program: str,
        program_format: ProgramFormat,
        num_shots: int,
        **parameters: object,
    ) -> QDMIJobHandle:
        job = submit_job(program, program_format, num_shots, **parameters)
        if job.id == "1":
            monkeypatch.setattr(job, "cancel", fail_cancel)
        elif job.id == "2":
            monkeypatch.setattr(job, "wait", fail_wait)
        return job

    monkeypatch.setattr(qdmi, "submit_job", submit)
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2, shots=[2, 3, 4])
    tape = qp.tape.QuantumScript([], [qp.sample(wires=[0, 1])], shots=[2, 3, 4])

    with pytest.raises(PennyLaneExecutionError, match="wait failed"):
        device.execute(tape)

    assert qdmi.events == [
        "submit:1",
        "submit:2",
        "submit:3",
        "wait:1",
        "cancel:1",
        "cancel:2",
        "cancel:3",
    ]


def test_parameter_shift_gradient_uses_multiple_qdmi_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Differentiate sampled execution through PennyLane's parameter-shift rule."""
    qdmi = stub_device(qubits=1, result_factory=rotation_results)
    patch_open_device(monkeypatch, qdmi)
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
    patch_open_device(monkeypatch, qdmi)
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
    qdmi = stub_device(program_format=ProgramFormat.OPENQASM2)
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2, shots=10)

    @qp.qnode(device)
    def circuit():
        return qp.expval(qp.PauliX(0))

    assert np.isfinite(circuit())
    assert qdmi.submissions[0][1] == ProgramFormat.OPENQASM2
    assert qdmi.submissions[0][0].count("ry(") == 1


def test_one_shot_executes_measurement_feedback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reconstruct terminal and raw mid-circuit samples from one output vector."""
    features = (
        "mid-circuit-measurement",
        "measured-qubit-reuse",
        "measurement-result-use",
        "boolean-computation",
        "forward-branching",
    )
    qdmi = stub_device(
        operations=[operation("measure", 1), operation("reset", 1), operation("x", 1)],
        result_factory=lambda _program, shots: ["110"] * shots,
        program_features=features,
    )
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2, shots=4)

    @qp.qnode(device, mcm_method="one-shot")
    def circuit():
        measurement = qp.measure(0, reset=True)
        qp.cond(measurement, qp.PauliX)(1)
        return qp.sample(wires=[0, 1]), qp.sample(measurement)

    terminal, mid_circuit = circuit()

    np.testing.assert_array_equal(terminal, np.tile([0, 1], (4, 1)))
    np.testing.assert_array_equal(mid_circuit, np.ones(4, dtype=np.int8))
    assert len(qdmi.submissions) == 4
    assert all("c[2] = measure q[0];" in submission[0] for submission in qdmi.submissions)


@pytest.mark.parametrize(
    "invalid_features",
    [
        [SimpleNamespace(id="forward-branching", value=1, constraint_id="", constraint_value=0)],
        [SimpleNamespace(id="forward-branching", value=0, constraint_id="max-depth", constraint_value=1)],
        [
            SimpleNamespace(id="forward-branching", value=0, constraint_id="", constraint_value=0),
            SimpleNamespace(id="forward-branching", value=0, constraint_id="", constraint_value=0),
        ],
        [
            SimpleNamespace(id="forward-branching", value=0, constraint_id="", constraint_value=0),
            SimpleNamespace(id="forward-branching", value=0, constraint_id="max-depth", constraint_value=1),
        ],
    ],
)
def test_one_shot_rejects_non_boolean_feature_records(
    monkeypatch: pytest.MonkeyPatch, invalid_features: list[object]
) -> None:
    """Ignore feature records whose representation the device does not understand."""
    features: list[object] = [
        "mid-circuit-measurement",
        "measured-qubit-reuse",
        "measurement-result-use",
        "boolean-computation",
        *invalid_features,
    ]
    qdmi = stub_device(operations=[operation("measure", 1), operation("reset", 1)], program_features=features)
    patch_open_device(monkeypatch, qdmi)

    device = QDMIDevice("fake.qdmi", wires=2, shots=4)

    assert device.capabilities.supported_mcm_methods == []


def test_rejects_analytic_execution_before_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject analytic tapes before a QDMI job is created."""
    qdmi = stub_device()
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2, shots=None)

    @qp.qnode(device)
    def circuit():
        return qp.expval(qp.PauliZ(0))

    with pytest.raises(PennyLaneValidationError, match="finite number of shots"):
        circuit()
    tape = qp.tape.QuantumScript([], [qp.sample(wires=0)], shots=None)
    with pytest.raises(PennyLaneValidationError, match="finite number of shots"):
        device.execute(tape)
    assert not qdmi.submissions


def test_validates_configuration_and_width(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject unknown QDMI parameters and excessive wire counts."""
    qdmi = stub_device()
    patch_open_device(monkeypatch, qdmi)

    with pytest.raises(PennyLaneConfigurationError, match="unknown"):
        QDMIDevice(
            "fake.qdmi",
            wires=2,
            session_parameters={"unknown": "value"},  # ty: ignore[invalid-argument-type, invalid-key]
        )
    with pytest.raises(PennyLaneConfigurationError, match="3 wires"):
        QDMIDevice("fake.qdmi", wires=3)
    with pytest.raises(PennyLaneConfigurationError, match="exactly one"):
        QDMIDevice()
    with pytest.raises(PennyLaneConfigurationError, match="exactly one"):
        QDMIDevice("fake.qdmi", device=cast("QDMIDeviceHandle", qdmi))
    with pytest.raises(PennyLaneConfigurationError, match="session_parameters"):
        QDMIDevice(
            device=cast("QDMIDeviceHandle", qdmi),
            session_parameters={"token": "unused"},
        )


def test_rejects_device_without_openqasm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject unsupported program formats during construction."""
    qdmi = StubDevice([], [ProgramFormat.QIR21_BASE_TEXT])
    patch_open_device(monkeypatch, qdmi)

    with pytest.raises(PennyLaneUnsupportedFormatError, match="neither OpenQASM 3 nor OpenQASM 2"):
        QDMIDevice("fake.qdmi", wires=2)


def test_forwards_job_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward generic QDMI custom job parameters unchanged."""
    qdmi = stub_device()
    patch_open_device(monkeypatch, qdmi)
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
