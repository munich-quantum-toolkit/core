# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""PennyLane configuration, native batching, and retained-job integration."""

from __future__ import annotations

from typing import cast

import numpy as np
import pennylane as qp
import pytest

from mqt.core.mlir import QCProgram, SerializationProfile, serialize_program
from mqt.core.plugins.pennylane import (
    PennyLaneConfigurationError,
    PennyLaneExecutionError,
    QDMIDevice,
)
from mqt.core.qdmi import Job as QDMIJobHandle
from mqt.core.qdmi import ProgramFormat
from mqt.core.qdmi.builtin_driver import open_device


@pytest.mark.parametrize("installed", ["0.14.1", "0.16.0"])
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


def test_execution_time_accumulates_batch_wall_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Count overlapping execution once per submit/collect phase."""
    readings = iter([0.0, 0.5, 0.5, 1.5, 10.0, 11.0, 11.0, 12.25])
    monkeypatch.setattr("mqt.core.plugins.pennylane.job.monotonic", lambda: next(readings))
    device = QDMIDevice("mqt.ddsim.default", wires=2)
    tape = qp.tape.QuantumScript([], [qp.sample(wires=[0, 1])], shots=[(5, 2), 7])
    device.execute(tape)
    device.execute(tape)
    assert device.submitted_jobs == 4
    assert device.execution_time == pytest.approx(1.5 + 2.25)


@pytest.mark.parametrize("max_retries", [0, 3])
def test_retry_shot_copies_preserves_order_and_tracking(monkeypatch: pytest.MonkeyPatch, max_retries: int) -> None:
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
        "mqt.ddsim.default", wires=wires, max_retries=max_retries, profile=SerializationProfile.BRAKET_SIMULATOR
    )
    program = serialize_program(
        QCProgram.from_openqasm_str(
            'OPENQASM 3; include "stdgates.inc"; bit[3] c = "000"; x $2; c[0] = measure $2; c[2] = measure $2;'
        ),
        target=device.qdmi_device,
        program_format=ProgramFormat.QASM3,
        profile=SerializationProfile.BRAKET_SIMULATOR,
    )
    assert program.result_layout is not None
    # Inject an equivalent prepared program with shared readout and a zero slot.
    monkeypatch.setattr(device, "_prepare", lambda tapes: [program] * len(tapes))
    tape = qp.tape.QuantumScript(
        [qp.X("left"), qp.X("right")], [qp.sample(wires=["right", "left", "idle"])], shots=[2, 3, 4]
    )
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
    reference = qp.device("default.qubit", wires=wires)
    (expected,) = qp.execute((tape,), reference, diff_method=None)
    for actual, oracle in zip(result, expected, strict=True):
        np.testing.assert_array_equal(actual, oracle)
    assert device.submitted_jobs == 4
    assert tracker.totals == {"batches": 1, "batch_len": 1, "executions": 4, "shots": 11}
    assert device.last_job is not None
    assert [(entry.input_index, entry.shot_index) for entry in device.last_job.entries] == [(0, 0), (0, 1), (0, 2)]
    assert device.last_job.entries[0].automatic_retries == int(bool(max_retries))
