# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""PennyLane configuration, native batching, and retained-job integration."""

from __future__ import annotations

import warnings
from importlib.util import find_spec
from typing import TYPE_CHECKING, cast

import numpy as np
import pennylane as qp
import pytest
from qiskit import QuantumCircuit

from mqt.core import UntestedVersionWarning
from mqt.core.mlir import TargetEnvironment, serialize_program
from mqt.core.plugins.pennylane import (
    PennyLaneConfigurationError,
    PennyLaneExecutionError,
    QDMIDevice,
)
from mqt.core.qdmi import Job as QDMIJobHandle
from mqt.core.qdmi import ProgramFormat
from mqt.core.qdmi.builtin_driver import open_device

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pennylane.typing import Result, ResultBatch


@pytest.mark.parametrize("installed", ["0.14.1", "0.15.0rc1", "1.0.0rc1", "1.0.0"])
def test_rejects_incompatible_catalyst_abi(monkeypatch: pytest.MonkeyPatch, installed: str) -> None:
    """Reject unsupported runtimes before they can call an incompatible C++ vtable."""
    monkeypatch.setattr("mqt.core.plugins.pennylane.device.version", lambda _name: installed)
    with pytest.raises(PennyLaneConfigurationError, match="unsupported device ABI"):
        QDMIDevice.get_c_interface()


def test_validates_configuration_and_width() -> None:
    """Reject invalid configuration before executing a circuit."""
    provider = open_device("mqt.ddsim.default")
    with pytest.raises(PennyLaneConfigurationError, match="unknown"):
        QDMIDevice(
            "mqt.ddsim.default",
            session_parameters={"unknown": "value"},  # ty: ignore[invalid-argument-type, invalid-key]
        )
    with pytest.raises(PennyLaneConfigurationError, match="wires were requested"):
        QDMIDevice(device=provider, wires=provider.qubits_num() + 1)
    with pytest.raises(PennyLaneConfigurationError, match="exactly one"):
        QDMIDevice()
    with pytest.raises(PennyLaneConfigurationError, match="exactly one"):
        QDMIDevice("mqt.ddsim.default", device=provider)
    with pytest.raises(PennyLaneConfigurationError, match="session_parameters"):
        QDMIDevice(device=provider, session_parameters={"token": "unused"})


@pytest.mark.parametrize("value", [-1, True])
def test_invalid_retry_configuration(value: object) -> None:
    """Retry limits are nonnegative integers."""
    with pytest.raises(PennyLaneConfigurationError, match="max_retries"):
        QDMIDevice("mqt.ddsim.default", max_retries=cast("int", value))


@pytest.mark.parametrize("all_wires", [False, True])
def test_compiles_only_requested_wires(*, all_wires: bool) -> None:
    """A small tape does not turn idle device wires into readout obligations."""
    device = QDMIDevice("mqt.ddsim.default", wires=8)
    tape = qp.tape.QuantumScript([qp.X(7)], [qp.sample() if all_wires else qp.sample(wires=[7])], shots=2)
    result = device.execute(tape)
    expected = [0] * 7 + [1] if all_wires else [1]
    np.testing.assert_array_equal(result, np.tile(expected, (2, 1)))
    assert device.last_job is not None
    entry = device.last_job.entries[0]
    assert isinstance(entry.result, np.ndarray)
    assert entry.result.shape == (2, 8 if all_wires else 1)


def test_forwards_job_parameters() -> None:
    """The DDSIM seed reaches the provider through native submission."""
    tape = qp.tape.QuantumScript([qp.Hadamard(0)], [qp.sample(wires=[0, 1])], shots=64)
    results = [QDMIDevice("mqt.ddsim.default", wires=2, job_parameters={"custom1": 42}).execute(tape) for _ in range(2)]
    np.testing.assert_array_equal(results[0], results[1])


def test_native_groups_preserve_heterogeneous_shot_partitions() -> None:
    """Real multi-program jobs preserve tape order, wire order, and shot copies."""
    provider = open_device("mqt.ddsim.default")
    device = QDMIDevice(device=provider, wires=2)
    assert device.qdmi_device is provider
    tapes = (
        qp.tape.QuantumScript([qp.X(0)], [qp.sample(wires=[0, 1])], shots=[2, 3, 2]),
        qp.tape.QuantumScript([qp.X(1)], [qp.sample(wires=[0, 1])], shots=3),
    )
    with qp.Tracker(device) as tracker:
        results = qp.execute(tapes, device, diff_method=None)
    assert device.submitted_jobs == 2
    assert tracker.totals == {"batches": 1, "batch_len": 2, "executions": 4, "shots": 10}
    for result, count, expected in zip(
        (*results[0], results[1]), [2, 3, 2, 3], [[1, 0], [1, 0], [1, 0], [0, 1]], strict=True
    ):
        np.testing.assert_array_equal(result, np.tile(expected, (count, 1)))
    assert device.shots.total_shots is None


def test_submit_exposes_native_jobs_before_collection() -> None:
    """Retain real provider handles without waiting or decoding the batch."""
    device = QDMIDevice("mqt.ddsim.default", wires=2)
    tapes = (
        qp.tape.QuantumScript([qp.X(0)], [qp.sample(wires=[1, 0])], shots=[2, 3]),
        qp.tape.QuantumScript([qp.X(1)], [qp.sample(wires=[1, 0])], shots=3),
    )
    with qp.Tracker(device) as tracker:
        job = device.submit(tapes)
        assert device.last_job is job
        assert len(job.entries) == 3
        assert all(entry.result is None for entry in job.entries)
        for entry in job.entries:
            handle = entry.attempts[0].handle
            assert handle is not None
            assert handle.id
        assert device.submitted_jobs == 2
        results = job.result()
    assert tracker.totals == {"batches": 1, "batch_len": 2, "executions": 3, "shots": 8}
    for result, shots, expected in zip((*results[0], results[1]), (2, 3, 3), ([0, 1], [0, 1], [1, 0]), strict=True):
        np.testing.assert_array_equal(result, np.tile(expected, (shots, 1)))


def test_empty_batch_keeps_execution_state() -> None:
    """Empty execution does not compile, submit, or record a batch."""
    device = QDMIDevice("mqt.ddsim.default", wires=2)
    with qp.Tracker(device) as tracker:
        assert device.execute(()) == ()
    assert device.last_job is None
    assert device.submitted_jobs == 0
    assert device.execution_time == 0
    assert tracker.totals == {}
    assert device.submit(()).result() == ()


def test_execution_time_accumulates_batch_wall_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Count overlapping execution once per submit/collect phase."""
    readings = iter([0.0, 0.5, 0.5, 1.5, 10.0, 11.0, 11.0, 12.25])
    monkeypatch.setattr("mqt.core.plugins.pennylane.job.monotonic", lambda: next(readings))
    device = QDMIDevice("mqt.ddsim.default", wires=2)
    tape = qp.tape.QuantumScript([qp.X(0)], [qp.sample(wires=[0, 1])], shots=[(5, 2), 7])
    device.execute(tape)
    device.execute(tape)
    assert device.submitted_jobs == 4
    assert device.execution_time == pytest.approx(1.5 + 2.25)


@pytest.mark.parametrize("max_retries", [0, 3])
@pytest.mark.parametrize("defer_collection", [False, True])
def test_retry_shot_copies_preserves_order_and_tracking(
    monkeypatch: pytest.MonkeyPatch, max_retries: int, *, defer_collection: bool
) -> None:
    """Inject one failed job while retaining real compiled programs and jobs."""
    original_check = QDMIJobHandle.check
    original_program_status = QDMIJobHandle.get_program_status
    failed: QDMIJobHandle | None = None

    def check(job: QDMIJobHandle) -> QDMIJobHandle.Status:
        nonlocal failed
        if failed is None:
            failed = job
        if job is failed:
            return QDMIJobHandle.Status.FAILED
        return original_check(job)

    def program_status(job: QDMIJobHandle, program_index: int = 0) -> QDMIJobHandle.Status | None:
        if job is failed:
            return QDMIJobHandle.Status.FAILED
        return original_program_status(job, program_index)

    monkeypatch.setattr(QDMIJobHandle, "check", check)
    monkeypatch.setattr(QDMIJobHandle, "get_program_status", program_status)
    wires = ["left", "idle", "right"]
    device = QDMIDevice(
        "mqt.ddsim.default",
        wires=wires,
        max_retries=max_retries,
        program_format=ProgramFormat.QASM3,
    )
    native = QuantumCircuit(3, 3)
    native.x(2)
    native.measure([2, 0], [0, 2])
    program = serialize_program(
        native,
        target=TargetEnvironment.from_device(device.qdmi_device, program_format=ProgramFormat.QASM3),
    )
    assert program.result_layout is not None
    # Inject an equivalent prepared program with permuted readout and a zero slot.
    monkeypatch.setattr(device, "_prepare", lambda tapes: [program] * len(tapes))
    tape = qp.tape.QuantumScript([qp.X("right")], [qp.sample(wires=["right", "left", "idle"])], shots=[2, 3, 4])

    def execute() -> Result | ResultBatch:
        return device.submit(tape).result() if defer_collection else device.execute(tape)

    with qp.Tracker(device) as tracker:
        if max_retries:
            result = execute()
        else:
            with pytest.raises(PennyLaneExecutionError) as caught:
                execute()
            assert caught.value.job is device.last_job
            assert device.last_job is not None
            device.last_job.resubmit([0])
            result = device.last_job.result()
    reference = qp.device("default.qubit", wires=wires)
    (expected,) = qp.execute((tape,), reference, diff_method=None)
    for actual, oracle in zip(result, expected, strict=True):
        np.testing.assert_array_equal(actual, oracle)
    assert device.submitted_jobs == 4
    assert tracker.totals == {"batches": 1, "batch_len": 1, "executions": 4, "shots": 11}
    assert device.last_job is not None
    assert [(entry.input_index, entry.shot_index) for entry in device.last_job.entries] == [(0, 0), (0, 1), (0, 2)]
    assert device.last_job.entries[0].automatic_retries == int(bool(max_retries))


@pytest.mark.parametrize("compiled", [False, True])
def test_openqasm2_execution(*, compiled: bool) -> None:
    """The shared compiler supports OpenQASM 2 in both PennyLane paths."""
    if compiled and find_spec("catalyst") is None:
        pytest.skip("Install Catalyst to run compiled device tests.")
    device = QDMIDevice("mqt.ddsim.default", wires=2, program_format=ProgramFormat.QASM2)

    @qp.qnode(device, shots=16)
    def circuit() -> qp.measurements.SampleMP:
        qp.PauliX(1)
        return qp.sample(wires=[1, 0])

    execute = qp.qjit(circuit) if compiled else circuit
    np.testing.assert_array_equal(execute(), np.tile([1, 0], (16, 1)))


def test_catalyst_patch_does_not_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch releases share the tested minor interface and have a header canary."""
    monkeypatch.setattr("mqt.core.plugins.pennylane.device.version", lambda _name: "0.15.1")
    monkeypatch.setattr(QDMIDevice, "_warned_catalyst_version", False)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        QDMIDevice.get_c_interface()
    assert not recorded


@pytest.mark.parametrize("installed", ["0.16.0", "0.16.0.dev1"])
def test_warns_once_for_new_catalyst_abi(monkeypatch: pytest.MonkeyPatch, installed: str) -> None:
    """Newer compatible-major runtimes warn before attempting their interface."""
    monkeypatch.setattr("mqt.core.plugins.pennylane.device.version", lambda _name: installed)
    monkeypatch.setattr(QDMIDevice, "_warned_catalyst_version", False)
    with pytest.warns(UntestedVersionWarning, match="interface change can crash"):
        assert QDMIDevice.get_c_interface()[0] == "MQTQDMI"
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        QDMIDevice.get_c_interface()
    assert not recorded


@pytest.mark.parametrize("measured", [False, True])
def test_local_results_need_no_provider_job(*, measured: bool) -> None:
    """Empty readout and untouched measured wires are completed locally."""
    device = QDMIDevice("mqt.ddsim.default", wires=2)
    tape = qp.tape.QuantumScript([], [qp.sample(wires=[1, 0])] if measured else [], shots=[2, 3])
    result = device.execute(tape)
    if measured:
        for shots, samples in zip((2, 3), result, strict=True):
            np.testing.assert_array_equal(samples, np.zeros((shots, 2)))
    else:
        assert result == ((), ())
    assert device.submitted_jobs == 0
    assert device.last_job is not None
    assert all(entry.result is not None for entry in device.last_job.entries)


def test_aggregate_readout_preserves_wires_and_shot_partitions() -> None:
    """Histograms retain all-wire measurements, subset ordering, and cached results."""
    wires = ["left", "idle", "right"]
    device = QDMIDevice("mqt.ddsim.default", wires=wires)
    tape = qp.tape.QuantumScript(
        [qp.X("right")],
        [
            qp.probs(),
            qp.counts(),
            qp.counts(wires=["right", "left"], all_outcomes=True),
            qp.expval(2 * qp.Z("right")),
            qp.var(2 * qp.Z("left")),
            qp.measurements.ExpectationMP(wires=qp.wires.Wires(["right"]), eigvals=np.array([1j, -1j])),
        ],
        shots=[2, 3],
    )
    batch = device.submit(tape)
    actual = batch.result()
    (expected,) = qp.execute((tape,), qp.device("default.qubit", wires=wires), diff_method=None)
    np.testing.assert_equal(actual, expected)
    np.testing.assert_equal([entry.result for entry in batch.entries], actual)
    retained = batch.entries[0].result
    np.testing.assert_equal(batch.result(), actual)
    assert batch.collect()[0].result is retained


def test_custom_measurement_keeps_sample_processing() -> None:
    """A custom sample processor is not replaced by inherited counts processing."""

    class TwiceExpectation(qp.measurements.ExpectationMP):
        def process_samples(
            self,
            samples: Sequence[complex],
            wire_order: qp.wires.Wires,
            shot_range: tuple[int, ...] | None = None,
            bin_size: int | None = None,
        ) -> np.ndarray:
            return 2 * super().process_samples(samples, wire_order, shot_range, bin_size)

    tape = qp.tape.QuantumScript([qp.X(0)], [TwiceExpectation(qp.Z(0))], shots=2)
    assert QDMIDevice("mqt.ddsim.default", wires=1).execute(tape) == -2
