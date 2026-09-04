# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Native Qiskit integration with deterministic recording QDMI jobs."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, Clbit, Parameter
from qiskit.providers import JobError
from qiskit.quantum_info import SparsePauliOp
from test_mock_backend import MockQDMIDevice

from mqt.core.plugins.qiskit import CircuitValidationError, JobSubmissionError, QDMIBackend
from mqt.core.qdmi import Job

if TYPE_CHECKING:
    from mqt.core.qdmi import Device, ProgramFormat


@pytest.fixture
def recording_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[QDMIBackend, list[MagicMock], list[str]]:
    """Record dispatch, collection, cancellation, and delayed ID queries."""
    device = MockQDMIDevice(operations=["h", "s", "sdg", "x", "cx", "rx", "ry", "rz", "measure"])
    jobs = []
    events = []
    submit = device.submit_job

    def submit_job(program: str, program_format: ProgramFormat, num_shots: int) -> MagicMock:
        original = submit(program, program_format, num_shots)
        width = original._num_clbits
        events.append("submit")
        job = MagicMock()
        job.num_shots = num_shots
        job.check.side_effect = lambda: (events.append("check"), Job.Status.DONE)[1]
        job.get_shots.side_effect = lambda: (events.append("shots"), ["0" * width] * num_shots)[1]
        job.get_counts.side_effect = lambda: (events.append("counts"), {"0" * width: num_shots})[1]
        type(job).id = PropertyMock(side_effect=lambda: (events.append("id"), "remote-id")[1])
        jobs.append(job)
        return job

    formats = device.supported_program_formats
    monkeypatch.setattr(device, "supported_program_formats", lambda: (events.append("formats"), formats())[1])
    monkeypatch.setattr(device, "submit_job", submit_job)
    return QDMIBackend(cast("Device", device)), jobs, events


@pytest.mark.parametrize("memory", [False, True])
def test_batch_order_and_repeated_reads(recording_backend, memory: bool) -> None:
    """Submission precedes collection and repeated reads do not contact the provider."""
    backend, jobs, events = recording_backend
    circuits = [QuantumCircuit(2, 2, name=f"circuit-{index}") for index in range(3)]
    job = backend.run(circuits, shots=4, memory=memory)
    assert events == ["formats", "submit", "submit", "submit"]
    for index, handle in enumerate(jobs):
        handle.get_shots.side_effect = None
        handle.get_shots.return_value = [f"{index:02b}"] * 4
        handle.get_counts.side_effect = None
        handle.get_counts.return_value = {f"{index:02b}": 4}
        handle.check.side_effect = [Job.Status.RUNNING, Job.Status.DONE]
        handle.wait.side_effect = lambda: events.append("wait")
    result = job.result()
    assert events.count("wait") == 3
    assert events.count("id") == 1
    assert [result.get_counts(circuit.name) for circuit in circuits] == [{"00": 4}, {"01": 4}, {"10": 4}]
    before = events.copy()
    assert job.result() is result
    assert job.job_id() == "remote-id"
    assert events == before
    for handle in jobs:
        assert handle.get_shots.call_count == int(memory)
        assert handle.get_counts.call_count == int(not memory)


def test_memory_order_registers_and_correlations(recording_backend) -> None:
    """Preserve provider order and correlations through Qiskit's register containers."""
    backend, jobs, _ = recording_backend
    qc = QuantumCircuit(3)
    qc.add_register(ClassicalRegister(1, "a"), ClassicalRegister(2, "b"))
    qc.metadata = {"experiment": "joint"}
    samples = ["101", "000", "011", "100"]
    job = backend.run(qc, shots=4, memory=True)
    jobs[0].get_shots.side_effect = lambda: samples
    result = job.result()
    assert result.get_memory() == ["10 1", "00 0", "01 1", "10 0"]
    assert result.get_counts() == Counter(result.get_memory())
    assert result.results[0].to_dict()["header"]["metadata"] == qc.metadata

    original_run = backend.run

    def run(*args, **kwargs):
        handle = original_run(*args, **kwargs)
        jobs[-1].get_shots.side_effect = lambda: samples
        return handle

    backend.run = run
    pub = backend.sampler().run([qc], shots=4).result()[0]
    assert pub.metadata == {"shots": 4, "circuit_metadata": qc.metadata}
    assert pub.data.a.get_bitstrings() == ["1", "0", "1", "0"]
    assert pub.data.b.get_bitstrings() == ["10", "00", "01", "10"]
    joint = pub.join_data()
    assert joint.get_bitstrings() == samples
    assert joint.postselect([0], [1]).get_bitstrings() == ["101", "011"]


def test_result_header_snapshot(recording_backend) -> None:
    """Changes to a circuit after submission cannot change an existing job's result layout."""
    backend, _, _ = recording_backend
    qc = QuantumCircuit(1, 1, name="original", metadata={"experiment": 1})
    job = backend.run(qc, shots=4, memory=True)
    qc.name = "changed"
    qc.metadata["experiment"] = 2
    qc.add_register(ClassicalRegister(1, "later"))
    result = job.result()
    assert result.get_memory("original") == ["0"] * 4
    assert result.results[0].header == {
        "name": "original", "memory_slots": 1, "creg_sizes": [["c", 1]], "metadata": {"experiment": 1},
    }


@pytest.mark.parametrize("stage", ["wait", "check", "get_shots", "get_counts", "id"])
@pytest.mark.parametrize("error", [RuntimeError("original failure"), KeyboardInterrupt()])
def test_collection_failure_cancels_every_job(recording_backend, stage: str, error: BaseException) -> None:
    """Cleanup includes later jobs and never masks the original exception."""
    backend, jobs, _ = recording_backend
    job = backend.run([QuantumCircuit(1, 1)] * 3, shots=4, memory=stage == "get_shots")
    if stage == "wait":
        jobs[0].check.side_effect = lambda: Job.Status.RUNNING
    if stage == "id":
        type(jobs[0]).id = PropertyMock(side_effect=error)
    else:
        getattr(jobs[0], stage).side_effect = error
    jobs[0].cancel.side_effect = RuntimeError("cancellation failed")
    with pytest.raises(type(error)) as caught:
        job.result()
    assert caught.value is error
    for handle in jobs:
        handle.cancel.assert_called_once()


@pytest.mark.parametrize("status", [Job.Status.FAILED, Job.Status.CANCELED, Job.Status.RUNNING])
def test_unsuccessful_jobs_raise(recording_backend, status: Job.Status) -> None:
    """An unsuccessful job must not become an empty successful primitive result."""
    backend, jobs, _ = recording_backend
    job = backend.run([QuantumCircuit(1, 1)] * 2, shots=4, memory=True)
    jobs[0].check.side_effect = lambda: status
    with pytest.raises(JobError, match="did not complete successfully"):
        job.result()
    for handle in jobs:
        handle.cancel.assert_called_once()
        handle.get_shots.assert_not_called()


@pytest.mark.parametrize("samples", [[], ["0"], ["2"] * 4, ["00"] * 4, [""] * 4])
def test_malformed_memory_raises(recording_backend, samples: list[str]) -> None:
    """Reject missing shots, invalid characters, and incorrect classical widths."""
    backend, jobs, _ = recording_backend
    job = backend.run(QuantumCircuit(1, 1), shots=4, memory=True)
    jobs[0].get_shots.side_effect = lambda: samples
    with pytest.raises(JobError, match="Invalid QDMI"):
        job.result()
    jobs[0].cancel.assert_called_once()


@pytest.mark.parametrize("counts", [{}, {"0": 3}, {"0": -1, "1": 5}, {"0": 4.0}, {"2": 4}, {"00": 4}])
def test_malformed_histogram_raises(recording_backend, counts: dict[str, int]) -> None:
    """Reject histogram data that cannot describe the requested classical samples."""
    backend, jobs, _ = recording_backend
    job = backend.run(QuantumCircuit(1, 1), shots=4)
    jobs[0].get_counts.side_effect = lambda: counts
    with pytest.raises(JobError, match="Invalid QDMI"):
        job.result()


@pytest.mark.parametrize("error", [RuntimeError("submission failed"), KeyboardInterrupt()])
def test_submission_failure_cleanup(recording_backend, monkeypatch: pytest.MonkeyPatch, error: BaseException) -> None:
    """Clean up accepted jobs when a later submission fails or is interrupted."""
    backend, jobs, _ = recording_backend
    submit = backend.device.submit_job

    def failing_submit(**kwargs):
        if len(jobs) == 2:
            raise error
        return submit(**kwargs)

    monkeypatch.setattr(backend.device, "submit_job", failing_submit)
    with pytest.raises((JobSubmissionError, KeyboardInterrupt)) as caught:
        backend.run([QuantumCircuit(1, 1)] * 3)
    assert caught.value is error or caught.value.__cause__ is error
    assert len(jobs) == 2
    for handle in jobs:
        handle.cancel.assert_called_once()


def test_validation_before_submission(recording_backend) -> None:
    """A bad circuit cannot leave earlier circuits running."""
    backend, jobs, _ = recording_backend
    invalid = QuantumCircuit(1)
    invalid.rx(Parameter("theta"), 0)
    with pytest.raises(CircuitValidationError, match="unbound parameters"):
        backend.run([QuantumCircuit(1, 1), invalid])
    assert not jobs


@pytest.mark.parametrize("options", [{"shots": 1.5}, {"shots": True}, {"memory": 1}, {"seed_simulator": 1}, {"unknown": None}])
def test_invalid_options(recording_backend, options) -> None:
    """Reject ineffective or lossy options before submission."""
    backend, jobs, _ = recording_backend
    with pytest.raises(CircuitValidationError):
        backend.run(QuantumCircuit(1, 1), **options)
    assert not jobs


def test_backend_option_defaults_and_cancel(recording_backend) -> None:
    """Honor native backend options and attempt every explicit cancellation."""
    backend, jobs, _ = recording_backend
    backend.set_options(shots=7, memory=True)
    result = backend.run(QuantumCircuit(1, 1), seed_simulator=None).result()
    assert len(result.get_memory()) == 7
    job = backend.run([QuantumCircuit(1, 1)] * 2, shots=np.int64(4), memory=False)
    jobs[-2].cancel.side_effect = RuntimeError("failed")
    assert job.cancel() is False
    jobs[-1].cancel.assert_called_once()


def test_counts_only_device() -> None:
    """DDSIM supports native Estimator but must not silently fake native Sampler shots."""
    backend = QDMIBackend.from_device_id("mqt.ddsim.default")
    qc = QuantumCircuit(1, 1)
    qc.measure(0, 0)
    assert backend.run(qc, shots=4).result().get_counts() == {"0": 4}
    assert backend.estimator().run([(QuantumCircuit(1), "Z")], precision=0.5).result()[0].data["evs"] == 1
    with pytest.raises(RuntimeError, match="Not supported") as caught:
        backend.sampler().run([qc], shots=4).result()
    assert "SHOTS" in caught.value.__notes__[0]


def test_sampler_batching_and_mixed_shots(recording_backend) -> None:
    """Native Sampler batches all broadcasts with equal shots and preserves PUB order."""
    backend, jobs, events = recording_backend
    qc = QuantumCircuit(1, 1)
    qc.ry(Parameter("theta"), 0)
    result = backend.sampler().run([(qc, [[0], [1]], 4), (qc, [2], 8), (qc, [3], 4)]).result()
    assert [pub.data.c.num_shots for pub in result] == [4, 8, 4]
    assert [pub.data.c.shape for pub in result] == [(2,), (), ()]
    assert [job.num_shots for job in jobs] == [4, 4, 4, 8]
    assert events[:4] == ["formats", "submit", "submit", "submit"]
    assert result.metadata == {"version": 2}


@pytest.mark.parametrize("grouping, expected_jobs", [(True, 2), (False, 4)])
def test_estimator_grouping_and_uncertainty(recording_backend, grouping: bool, expected_jobs: int) -> None:
    """Qiskit owns commuting groups, duplicate terms, identity, precision, and standard errors."""
    backend, jobs, events = recording_backend
    qc = QuantumCircuit(2)
    qc.metadata = {"experiment": "observable"}
    observables = [SparsePauliOp(["ZI", "IZ", "XI", "II"], np.array([1, 2, 3, 4])), SparsePauliOp("ZI")]
    result = backend.estimator(options={"abelian_grouping": grouping}).run([(qc, observables)], precision=0.5).result()
    assert len(jobs) == expected_jobs
    assert events[:expected_jobs + 1] == ["formats"] + ["submit"] * expected_jobs
    np.testing.assert_equal(result[0].data.evs, [10, 1])
    np.testing.assert_equal(result[0].data.stds, [0, 0])
    assert result[0].metadata == {"target_precision": 0.5, "shots": 4, "circuit_metadata": qc.metadata}


def test_reject_nonpartitioned_classical_bits(recording_backend) -> None:
    """Reject register layouts that native result headers cannot represent faithfully."""
    backend, jobs, _ = recording_backend
    qc = QuantumCircuit(1)
    qc.add_bits([Clbit(), Clbit()])
    qc.add_register(ClassicalRegister(bits=qc.clbits[::-1], name="reversed"))
    with pytest.raises(CircuitValidationError, match="partition"):
        backend.run(qc, memory=True)
    assert not jobs


def test_estimator_mixed_precision(recording_backend) -> None:
    """Native precision groups retain input order and PUB-specific metadata."""
    backend, jobs, _ = recording_backend
    qc = QuantumCircuit(1)
    result = backend.estimator().run([(qc, "Z", None, 0.5), (qc, "Z", None, 0.25), (qc, "Z", None, 0.5)]).result()
    assert [pub.metadata["shots"] for pub in result] == [4, 16, 4]
    assert [job.num_shots for job in jobs] == [4, 4, 16]


def test_estimator_nonzero_uncertainty(recording_backend, monkeypatch: pytest.MonkeyPatch) -> None:
    """Native coefficient-weighted standard errors are not replaced by custom statistics."""
    backend, jobs, _ = recording_backend
    original_run = backend.run

    def run(*args, **kwargs):
        job = original_run(*args, **kwargs)
        jobs[-1].get_counts.side_effect = lambda: {"00": 1, "01": 1, "10": 1, "11": 1}
        return job

    monkeypatch.setattr(backend, "run", run)
    op = SparsePauliOp(["ZI", "IZ", "II"], np.array([2, 3, 4]))
    result = backend.estimator().run([(QuantumCircuit(2), [op, "ZI", "ZI"])], precision=0.5).result()[0]
    assert len(jobs) == 1
    np.testing.assert_equal(result.data["evs"], [4, 0, 0])
    np.testing.assert_equal(result.data["stds"], [2.5, 0.5, 0.5])
