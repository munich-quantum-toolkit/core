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
from typing import TYPE_CHECKING, cast

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

from .helpers import StubDevice, patch_open_device, rotation_results, stub_device

if TYPE_CHECKING:
    from unittest.mock import Mock


def test_uses_already_open_qdmi_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reuse a device session selected outside the PennyLane adapter."""
    qdmi = stub_device()
    monkeypatch.setattr(
        "mqt.core.plugins.pennylane.device.open_device",
        lambda *_args, **_kwargs: pytest.fail("The adapter reopened the QDMI device."),
    )

    device = QDMIDevice(device=cast("QDMIDeviceHandle", qdmi), wires=2)

    assert device.qdmi_device is qdmi
    assert device.device_id is None


def test_samples_counts_probabilities_expectations_and_variances(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reconstruct common PennyLane result types from raw QDMI samples."""
    qdmi = stub_device()
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=["left", "right"])

    @qp.qnode(device, shots=100)
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
    device = QDMIDevice("fake.qdmi", wires=2)

    @qp.qnode(device, shots=8)
    def circuit():
        return qp.sample(wires=[0, 1])

    samples = circuit()

    assert samples.shape == (8, 2)
    assert Counter(map(tuple, samples.tolist())) == {(0, 0): 4, (1, 1): 4}


def test_execution_time_accumulates_batch_wall_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Count overlapping job execution once for each PennyLane batch."""
    qdmi = stub_device()
    patch_open_device(monkeypatch, qdmi)
    readings = iter([0.0, 0.5, 0.5, 1.5, 10.0, 11.0, 11.0, 12.25])
    monkeypatch.setattr("mqt.core.plugins.pennylane.job.monotonic", lambda: next(readings))
    device = QDMIDevice("fake.qdmi", wires=2)

    @qp.qnode(device, shots=[(5, 2), 7])
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
    device = QDMIDevice("fake.qdmi", wires=2)

    @qp.qnode(device, shots=[(5, 2), 7])
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
    device = QDMIDevice("fake.qdmi", wires=2)
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


def test_execution_failure_preserves_submitted_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collect later jobs and retain successful samples without any cancellation."""
    qdmi = stub_device()
    submit_job = qdmi.submit_job

    def fail_wait() -> bool:
        msg = "wait failed"
        raise RuntimeError(msg)

    def submit(
        program: str,
        program_format: ProgramFormat,
        num_shots: int,
        **parameters: object,
    ) -> QDMIJobHandle:
        job = submit_job(program, program_format, num_shots, **parameters)
        if job.id == "2":
            monkeypatch.setattr(job, "wait", fail_wait)
        return job

    monkeypatch.setattr(qdmi, "submit_job", submit)
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2)
    tape = qp.tape.QuantumScript([], [qp.sample(wires=[0, 1])], shots=[2, 3, 4])

    with qp.Tracker(device) as tracker, pytest.raises(PennyLaneExecutionError, match="wait failed"):
        device.execute(tape)

    assert tracker.totals == {"batches": 1, "batch_len": 1, "executions": 3, "shots": 9}
    assert device.last_job is not None
    assert [entry.result is not None for entry in device.last_job.entries] == [True, False, True]

    assert qdmi.events == [
        "submit:1",
        "submit:2",
        "submit:3",
        "wait:1",
        "wait:3",
    ]


def test_parameter_shift_gradient_uses_multiple_qdmi_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Differentiate sampled execution through PennyLane's parameter-shift rule."""
    qdmi = stub_device(qubits=1, result_factory=rotation_results)
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=["theta"])

    @qp.qnode(device, shots=4000, diff_method="parameter-shift")
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
    device = QDMIDevice("fake.qdmi", wires=2)
    hamiltonian = 0.5 * qp.PauliZ(0) + 0.5 * qp.PauliZ(1)

    @qp.qnode(device, shots=100)
    def circuit():
        return qp.expval(hamiltonian), qp.expval(qp.PauliX(0))

    energy, x_expectation = circuit()

    assert energy == pytest.approx(0.0)
    assert np.isfinite(x_expectation)
    assert device.submitted_jobs >= 2


def test_qasm2_diagonalizes_observable_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not duplicate the X-basis rotation in PennyLane's QASM2 serializer."""
    qdmi = stub_device(program_format=ProgramFormat.QASM2)
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2)

    @qp.qnode(device, shots=10)
    def circuit():
        return qp.expval(qp.PauliX(0))

    assert np.isfinite(circuit())
    assert qdmi.submissions[0][1] == ProgramFormat.QASM2
    assert qdmi.submissions[0][0].count("ry(") == 1


def test_rejects_analytic_execution_before_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject analytic tapes before a QDMI job is created."""
    qdmi = stub_device()
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2)

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
    qdmi = StubDevice([], [ProgramFormat.QIR_BASE_STRING])
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
        job_parameters={"custom1": "bucket", "custom2": "prefix"},
    )

    @qp.qnode(device, shots=4)
    def circuit():
        return qp.sample(wires=[0, 1])

    circuit()
    assert qdmi.submissions[0][3] == {"custom1": "bucket", "custom2": "prefix"}


@pytest.mark.parametrize("method", ["one-shot", "tree-traversal"])
def test_rejects_unsupported_mid_circuit_methods(method: str) -> None:
    """Do not silently replace a requested native MCM method with deferral."""
    qdmi = stub_device()
    device = QDMIDevice(device=cast("QDMIDeviceHandle", qdmi), wires=2)

    @qp.qnode(device, shots=5, mcm_method=method)
    def circuit():
        return qp.sample(wires=0)

    with pytest.raises(qp.exceptions.QuantumFunctionError, match="unsupported by the device"):
        circuit()
    assert not qdmi.submissions


def test_rejects_deferred_measurement_without_spare_wire() -> None:
    """Fail before submission when resetting a measured wire needs an ancilla."""
    qdmi = stub_device()
    device = QDMIDevice(device=cast("QDMIDeviceHandle", qdmi), wires=["a", "b"])

    @qp.qnode(device, shots=5)
    def circuit():
        measured = qp.measure("a", reset=True)
        qp.cond(measured, qp.X)("b")
        return qp.sample(wires=["b", "a"])

    with pytest.raises(PennyLaneValidationError, match="require more wires"):
        circuit()
    assert not qdmi.submissions


def test_qnode_shots_and_tracker() -> None:
    """Track actual jobs and shot copies while QNode shots specify execution budgets."""
    qdmi = stub_device()
    device = QDMIDevice(device=cast("QDMIDeviceHandle", qdmi), wires=2)

    @qp.qnode(device, shots=7)
    def circuit():
        return qp.sample(wires=[0, 1])

    with qp.Tracker(device) as tracker:
        assert circuit().shape == (7, 2)
        results = qp.set_shots(circuit, shots=[(3, 2), 5])()
    assert [result.shape for result in results] == [(3, 2), (3, 2), (5, 2)]
    assert tracker.totals == {"batches": 2, "batch_len": 2, "executions": 4, "shots": 18}
    assert device.submitted_jobs == 4
    assert device.shots.total_shots is None
    circuit()
    assert tracker.totals["executions"] == 4


@pytest.mark.parametrize("order", [[0, 1], [1, 0], [1]])
def test_sample_decoding_preserves_wire_order_and_dtype(monkeypatch: pytest.MonkeyPatch, order: list[int]) -> None:
    """Decode spaced bit strings in the requested PennyLane wire order."""
    qdmi = stub_device(result_factory=lambda _program, _shots: ["0 1", "10"])
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2)
    tape = qp.tape.QuantumScript([], [qp.sample(wires=order)], shots=2)
    samples = device.execute(tape)
    np.testing.assert_array_equal(samples, np.array([[1, 0], [0, 1]], dtype=np.int8)[:, order])
    assert isinstance(samples, np.ndarray)
    assert samples.dtype == np.int8


@pytest.mark.parametrize("bitstrings", [["001", "0"], ["0x", "10"], ["0é", "10"], ["01"]])
def test_sample_decoding_rejects_malformed_shots(monkeypatch: pytest.MonkeyPatch, bitstrings: list[str]) -> None:
    """Validate each shot before packing; total character count is insufficient."""
    qdmi = stub_device(result_factory=lambda _program, _shots: bitstrings)
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2)
    tape = qp.tape.QuantumScript([], [qp.sample(wires=[0, 1])], shots=2)
    with pytest.raises(PennyLaneExecutionError, match=r"invalid 2-wire shot|samples for a 2-shot job"):
        device.execute(tape)


@pytest.mark.parametrize("max_retries", [0, 3])
def test_retry_shot_copies_preserves_order_and_tracking(monkeypatch: pytest.MonkeyPatch, max_retries: int) -> None:
    """Replace one failed shot copy without repeating successful copies or losing their mapping."""
    qdmi = stub_device()
    original = qdmi.submit_job
    handles = []

    def submit(program: str, program_format: ProgramFormat, num_shots: int, **parameters: object) -> QDMIJobHandle:
        handle = original(program, program_format, num_shots, **parameters)
        handle = cast("Mock", handle)
        handle.check.side_effect = None
        handle.check.return_value = QDMIJobHandle.Status.FAILED if not handles else QDMIJobHandle.Status.DONE
        handles.append(handle)
        return handle

    monkeypatch.setattr(qdmi, "submit_job", submit)
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2, max_retries=max_retries, job_parameters={"custom1": 9})
    tape = qp.tape.QuantumScript([], [qp.sample(wires=[0, 1])], shots=[2, 3, 4])
    with qp.Tracker(device) as tracker:
        if max_retries:
            result = device.execute(tape)
        else:
            with pytest.raises(PennyLaneExecutionError) as caught:
                device.execute(tape)
            assert caught.value.job is device.last_job
            assert device.last_job is not None
            device.last_job.resubmit([0])
            result = device.last_job.result()
    assert [samples.shape for samples in result] == [(2, 2), (3, 2), (4, 2)]
    assert [submission[2] for submission in qdmi.submissions] == [2, 3, 4, 2]
    assert qdmi.submissions[0] == qdmi.submissions[3]
    assert device.submitted_jobs == 4
    assert tracker.totals == {"batches": 1, "batch_len": 1, "executions": 4, "shots": 11}
    assert device.last_job is not None
    assert [(entry.input_index, entry.shot_index) for entry in device.last_job.entries] == [(0, 0), (0, 1), (0, 2)]
    assert device.last_job.entries[0].automatic_retries == int(bool(max_retries))
    for handle in handles[1:]:
        handle.get_shots.assert_called_once()
        handle.cancel.assert_not_called()


@pytest.mark.parametrize("value", [-1, True, 1.5, "3", None])
def test_invalid_retry_configuration(monkeypatch: pytest.MonkeyPatch, value: object) -> None:
    """Reject invalid retry counts through both PennyLane constructors before execution."""
    qdmi = stub_device()
    patch_open_device(monkeypatch, qdmi)
    with pytest.raises(PennyLaneConfigurationError, match="max_retries"):
        QDMIDevice("fake.qdmi", max_retries=cast("int", value))
    with pytest.raises(PennyLaneConfigurationError, match="max_retries"):
        qp.device("mqt.ddsim.default", max_retries=value)
    assert not qdmi.submissions


@pytest.mark.parametrize("interrupted", [False, True])
def test_partial_submission_retains_pennylane_batch(monkeypatch: pytest.MonkeyPatch, *, interrupted: bool) -> None:
    """Submission exceptions and interruptions retain handles for accepted shot copies."""
    qdmi = stub_device()
    original = qdmi.submit_job
    cause = KeyboardInterrupt() if interrupted else RuntimeError("submission failed")

    def submit(program: str, program_format: ProgramFormat, num_shots: int, **parameters: object) -> QDMIJobHandle:
        if len(qdmi.submissions) == 1:
            raise cause
        return original(program, program_format, num_shots, **parameters)

    monkeypatch.setattr(qdmi, "submit_job", submit)
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2)
    tape = qp.tape.QuantumScript([], [qp.sample(wires=[0, 1])], shots=[2, 3, 4])
    with pytest.raises(KeyboardInterrupt if interrupted else PennyLaneExecutionError) as caught:
        device.execute(tape)
    batch = device.last_job
    assert batch is not None
    if isinstance(caught.value, PennyLaneExecutionError):
        assert caught.value.job is batch
    assert batch.entries[1].attempts[0].failures[0].cause is cause
    assert not batch.entries[2].attempts
    batch.collect()
    assert batch.entries[0].result is not None
    monkeypatch.setattr(qdmi, "submit_job", original)
    batch.resubmit([1], allow_unknown=True)
    batch.submit()
    assert [samples.shape for samples in batch.result()] == [(2, 2), (3, 2), (4, 2)]
    assert device.submitted_jobs == 3
    assert not any(event.startswith("cancel") for event in qdmi.events)


def test_result_read_failures_retain_both_causes(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unavailable shots result and failing counts fallback keep both diagnostics."""
    qdmi = stub_device()
    original = qdmi.submit_job
    errors = (RuntimeError("shots unavailable"), RuntimeError("counts unavailable"))

    def submit(program: str, program_format: ProgramFormat, num_shots: int, **parameters: object) -> QDMIJobHandle:
        handle = original(program, program_format, num_shots, **parameters)
        handle = cast("Mock", handle)
        handle.check.side_effect = None
        handle.check.return_value = QDMIJobHandle.Status.DONE
        handle.get_shots.side_effect = errors[0]
        handle.get_counts.side_effect = errors[1]
        return handle

    monkeypatch.setattr(qdmi, "submit_job", submit)
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2)
    tape = qp.tape.QuantumScript([], [qp.sample(wires=[0, 1])], shots=2)
    with pytest.raises(PennyLaneExecutionError) as caught:
        device.execute(tape)
    assert caught.value.job is not None
    failure = caught.value.job.entries[0].attempts[0].failures[0]
    assert failure.stage == "result"
    assert isinstance(failure.cause, PennyLaneExecutionError)
    assert isinstance(failure.cause.__cause__, ExceptionGroup)
    assert failure.cause.__cause__.exceptions == errors
    assert len(qdmi.submissions) == 1


def test_tracking_interruption_keeps_accepted_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tracker callback cannot lose a job that the device already accepted."""
    qdmi = stub_device()
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2)
    tape = qp.tape.QuantumScript([], [qp.sample(wires=[0, 1])], shots=[2, 3])

    def callback(**_kwargs: object) -> None:
        if device.submitted_jobs:
            raise KeyboardInterrupt

    with qp.Tracker(device, callback=callback), pytest.raises(KeyboardInterrupt):
        device.execute(tape)
    batch = device.last_job
    assert batch is not None
    assert batch.entries[0].attempts[0].handle is not None
    assert not batch.entries[1].attempts
    batch.submit([1])
    assert [samples.shape for samples in batch.result()] == [(2, 2), (3, 2)]
    assert device.submitted_jobs == 2


@pytest.mark.parametrize("opt_in", [False, True])
def test_automatic_replacements_require_opt_in(monkeypatch: pytest.MonkeyPatch, *, opt_in: bool) -> None:
    """The default preserves failed jobs; explicit configuration permits bounded replacements."""
    qdmi = stub_device()
    original = qdmi.submit_job

    def submit(program: str, program_format: ProgramFormat, num_shots: int, **parameters: object) -> QDMIJobHandle:
        handle = cast("Mock", original(program, program_format, num_shots, **parameters))
        handle.check.side_effect = None
        handle.check.return_value = QDMIJobHandle.Status.FAILED
        return cast("QDMIJobHandle", handle)

    monkeypatch.setattr(qdmi, "submit_job", submit)
    patch_open_device(monkeypatch, qdmi)
    device = QDMIDevice("fake.qdmi", wires=2, **({"max_retries": 3} if opt_in else {}))
    tape = qp.tape.QuantumScript([], [qp.sample(wires=[0, 1])], shots=2)
    with pytest.raises(PennyLaneExecutionError):
        device.execute(tape)
    assert device.last_job is not None
    with pytest.raises(PennyLaneExecutionError):
        device.last_job.result()
    assert device.submitted_jobs == (4 if opt_in else 1)
