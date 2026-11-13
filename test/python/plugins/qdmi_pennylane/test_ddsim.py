# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""End-to-end PennyLane tests with the DDSIM QDMI device."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pennylane as qp
import pytest

from mqt.core.mlir import QCProgram, compile_program, submit_program
from mqt.core.plugins.pennylane import (
    DDSIMDevice,
    PennyLaneConfigurationError,
    PennyLaneJob,
    PennyLaneTranslationError,
    PennyLaneValidationError,
    QDMIDevice,
)
from mqt.core.qdmi import Job, ProgramFormat

if TYPE_CHECKING:
    from pennylane.measurements import CountsMP, ExpectationMP, ProbabilityMP, SampleMP, VarianceMP

    from mqt.core.typing import QDMIJobParameters, QDMISessionParameters

GRAPH_EDGES = ((0, 1), (0, 2), (1, 2), (2, 3))


@pytest.mark.parametrize(
    ("wires", "parameters", "width"),
    [(2, {}, 2), (["a", "b", "c"], {}, 3), (None, {}, 65535), (2, {"custom1": "4"}, 4)],
)
def test_ddsim_session_width(wires: int | list[str] | None, parameters: QDMISessionParameters, width: int) -> None:
    """Scope implicit sessions to declared wires and preserve explicit overrides."""
    original = parameters.copy()
    device = DDSIMDevice(wires=wires, session_parameters=parameters)
    assert device.qdmi_device.qubits_num() == width
    assert parameters == original


@pytest.mark.parametrize("shots", [[64] * 5, [64, 32, 64, 32, 64]])
def test_seed_streams_do_not_depend_on_batching(shots: list[int]) -> None:
    """Give each program of a seeded device its own stream across submissions."""
    tapes = [
        qp.tape.QuantumScript([qp.Hadamard(wire) for wire in range(3)], [qp.sample(wires=range(3))], shots=count)
        for count in shots
    ]
    batched = DDSIMDevice(wires=3, seed=17)
    split = DDSIMDevice(wires=3, seed=17)

    expected = [np.asarray(result) for result in batched.execute(tapes)]
    actual = [np.asarray(result) for batch in (tapes[:2], tapes[2:4], tapes[4:]) for result in split.execute(batch)]

    # Equal shots share one multi-program job; interleaved shot groups fall back to single-program jobs.
    assert batched.submitted_jobs == (1 if len(set(shots)) == 1 else len(shots))
    for expected_samples, actual_samples in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(actual_samples, expected_samples)
    first, second = (np.asarray(split.execute(tapes[0])) for _ in range(2))
    assert not np.array_equal(first, second)


@pytest.mark.parametrize(
    ("seed", "job_parameters"),
    [(0, {}), (True, {}), (2**31, {}), (1.0, {}), (5, {"custom1": 5})],
)
def test_seed_validation(seed: object, job_parameters: QDMIJobParameters) -> None:
    """Reject invalid seeds and conflicting raw DDSIM seeds before opening a session."""
    with pytest.raises(PennyLaneConfigurationError, match="seed"):
        DDSIMDevice(wires=1, seed=seed, job_parameters=job_parameters)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("max_retries", [0, 1])
def test_seeded_shot_copies_retain_streams_on_retry(monkeypatch: pytest.MonkeyPatch, max_retries: int) -> None:
    """Retry nonconsecutive copies without changing their samples or later streams."""
    tape = qp.tape.QuantumScript([qp.Hadamard(0)], [qp.sample(wires=0)], shots=[(64, 3)])
    reference = DDSIMDevice(wires=1, seed=17)
    expected = reference.execute(tape)
    device = DDSIMDevice(wires=1, seed=17, max_retries=max_retries)
    batch = device.submit(tape)
    failed = batch.entries[0].attempts[0].handle
    assert failed is not None
    assert failed.wait()
    original_check = Job.check
    original_status = Job.get_program_status

    def check(job: Job) -> Job.Status:
        return Job.Status.FAILED if job is failed else original_check(job)

    def status(job: Job, program_index: int = 0) -> Job.Status | None:
        if job is failed and program_index in {0, 2}:
            return Job.Status.FAILED
        return original_status(job, program_index)

    monkeypatch.setattr(Job, "check", check)
    monkeypatch.setattr(Job, "get_program_status", status)
    if not max_retries:
        batch.collect()
        batch.resubmit([2, 0])
    np.testing.assert_array_equal(batch.result(), expected)
    assert [len(entry.attempts) for entry in batch.entries] == [2, 1, 2]
    np.testing.assert_array_equal(device.execute(tape), reference.execute(tape))


def test_seed_range_exhaustion_prevents_submission() -> None:
    """Reject a batch whose program seeds exceed the DDSIM seed range."""
    device = DDSIMDevice(wires=1, seed=2**31 - 1)
    tape = qp.tape.QuantumScript([], [qp.sample(wires=0)], shots=1)
    with pytest.raises(PennyLaneValidationError, match="limit"):
        device.execute([tape, tape])
    assert device.submitted_jobs == 0
    np.testing.assert_array_equal(device.execute(tape), [[0]])


def test_multi_controlled_gate_uses_no_spare_device_wires() -> None:
    """Execute a gate occupying the entire session without extra target sites."""
    device = DDSIMDevice(wires=5, job_parameters={"custom1": 42})
    tape = qp.tape.QuantumScript(
        [*(qp.Hadamard(wire) for wire in range(4)), qp.MultiControlledX(wires=range(5))],
        [qp.sample(wires=range(5))],
        shots=512,
    )
    samples = np.asarray(device.execute(tape))
    assert device.qdmi_device.qubits_num() == 5
    np.testing.assert_array_equal(samples[:, 4], np.prod(samples[:, :4], axis=1))
    assert np.any(samples[:, 4])


@pytest.mark.parametrize(("wires", "expected"), [(None, "01"), (["idle", "spare", "active"], "100")])
def test_public_tape_import_returns_all_declared_wires(wires: list[str] | None, expected: str) -> None:
    """Public import preserves default or explicit wire order and reads every wire."""
    tape = qp.tape.QuantumScript(
        [qp.BasisState([1], "active"), qp.QubitUnitary(np.eye(2), "idle")], [qp.sample(wires=["active"])]
    )
    program = compile_program(QCProgram.from_pennylane(tape, wires=wires), target="mqt.ddsim.default")
    job = submit_program(program, target="mqt.ddsim.default", num_shots=3)
    assert job.wait()
    shots = job.get_shots()
    if program.result_layout is not None:
        shots = [program.result_layout.decode_bits(bits) for bits in shots]
    assert shots == [expected] * 3


def test_plugin_registration_and_wire_order() -> None:
    """Load the device plugin and preserve named-wire order."""
    device = qp.device("mqt.ddsim.default", wires=["first", "second"])

    @qp.qnode(device, shots=20)
    def circuit() -> CountsMP:
        qp.PauliX("first")
        return qp.counts(wires=["first", "second"])

    assert isinstance(device, DDSIMDevice)
    assert device.device_id == "mqt.ddsim.default"
    assert circuit() == {"10": 20}


def test_terminal_readout_preserves_named_wire_order() -> None:
    """Explicit receiver serialization preserves an asymmetric QNode result."""
    wires = ["idle", "active", "spare"]
    device = QDMIDevice(
        "mqt.ddsim.default",
        wires=wires,
        program_format=ProgramFormat.QASM3,
    )

    @qp.qnode(device, shots=4)
    def circuit() -> SampleMP:
        qp.X("active")
        return qp.sample(wires=["spare", "idle", "active"])

    np.testing.assert_array_equal(circuit(), np.tile([0, 0, 1], (4, 1)))


def test_bell_results_and_shot_vector() -> None:
    """Execute probabilities, samples, and shot-vector partitions end to end."""
    device = qp.device("mqt.ddsim.default", wires=2)

    @qp.qnode(device, shots=[(1000, 2), 2000])
    def circuit() -> ProbabilityMP:
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
    device = qp.device("mqt.ddsim.default", wires=1)

    @qp.qnode(device, shots=10_000, diff_method="parameter-shift")
    def circuit(angle: float) -> ExpectationMP:
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
        [qp.Hadamard(0), qp.ctrl(qp.GlobalPhase(0.73), control=0), qp.Hadamard(0)],
        [
            qp.Hadamard(0),
            qp.ctrl(qp.adjoint(qp.RY(0.73, 1)), control=0, control_values=[False]),
        ],
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
    """Match gate action using interference-sensitive sampled probabilities."""
    qdmi_device = qp.device("mqt.ddsim.default", wires=2)
    reference_device = qp.device("default.qubit", wires=2)
    operations = [qp.RY(-0.3, 0), qp.RX(0.3, 1), *operations, qp.RX(1.2, 0), qp.RY(-1.2, 1)]
    sampled_tape = qp.tape.QuantumScript(operations, [qp.probs(wires=[0, 1])], shots=20_000)
    analytic_tape = qp.tape.QuantumScript(operations, [qp.probs(wires=[0, 1])])

    (actual,) = qp.execute((sampled_tape,), qdmi_device, diff_method=None)
    (expected,) = qp.execute((analytic_tape,), reference_device, diff_method=None)

    np.testing.assert_allclose(actual, expected, atol=0.025)


def test_invalid_native_program_prevents_batch_submission() -> None:
    """Validate every compiled tape before accepting any job in the batch."""
    device = DDSIMDevice(wires=1)
    tape = qp.tape.QuantumScript([qp.X(0)], [qp.sample(wires=[0])], shots=2)
    samples = device.execute(tape)
    previous = device.last_job
    assert previous is not None
    invalid = qp.tape.QuantumScript([qp.RY(np.nan, 0)], [qp.sample(wires=[0])], shots=2)
    with pytest.raises(PennyLaneTranslationError, match="finite"):
        device.execute((tape, invalid))
    assert device.submitted_jobs == 1
    assert device.last_job is None
    np.testing.assert_array_equal(previous.result(), samples)


@pytest.mark.parametrize("order", [None, [0, 1], [1, 0], [1]])
def test_sample_wire_order(order: list[int] | None) -> None:
    """Select requested wires from the compiler's logical classical readout."""
    device = DDSIMDevice(wires=2)
    tape = qp.tape.QuantumScript([qp.X(0)], [qp.sample(wires=order)], shots=3)
    np.testing.assert_array_equal(
        device.execute(tape), np.array([[1, 0]] * 3, dtype=np.int8)[:, range(2) if order is None else order]
    )


def test_qaoa_application() -> None:
    """Optimize one finite-shot MaxCut QAOA step through the real QDMI device."""
    graph = nx.Graph(GRAPH_EDGES)
    cost_hamiltonian = qp.sum(*(0.5 * (qp.Z(u) @ qp.Z(v) - qp.I(u)) for u, v in graph.edges))
    mixer_hamiltonian = qp.sum(*(qp.X(wire) for wire in graph.nodes))
    device = qp.device("mqt.ddsim.default", wires=4)

    def ansatz(parameters: np.ndarray) -> None:
        for wire in graph.nodes:
            qp.Hadamard(wire)
        qp.ApproxTimeEvolution(cost_hamiltonian, parameters[0], 1)
        qp.ApproxTimeEvolution(mixer_hamiltonian, parameters[1], 1)

    @qp.qnode(device, shots=200, diff_method="parameter-shift")
    def cost(parameters: np.ndarray) -> ExpectationMP:
        ansatz(parameters)
        return qp.expval(cost_hamiltonian)

    @qp.qnode(device, shots=200)
    def sample(parameters: np.ndarray) -> SampleMP:
        ansatz(parameters)
        return qp.sample(wires=range(4))

    parameters = qp.numpy.array([0.5, 0.5], requires_grad=True)
    initial_cost = float(cost(parameters))
    gradient = np.asarray(qp.grad(cost)(parameters), dtype=float)
    optimized = qp.GradientDescentOptimizer(stepsize=0.15).step(cost, parameters)
    samples = np.asarray(sample(optimized), dtype=np.int8)
    bitstrings = {"".join(str(int(bit)) for bit in row) for row in samples}

    assert np.isfinite(initial_cost)
    assert np.all(np.isfinite(gradient))
    assert not np.allclose(optimized, parameters)
    assert samples.shape == (200, 4)
    assert all(len(bitstring) == 4 and set(bitstring) <= {"0", "1"} for bitstring in bitstrings)
    assert device.submitted_jobs > 1


@pytest.mark.parametrize("wires", [["control", "spare", "target"], [7, 3, 9], [0, 1, 2]])
def test_deferred_measurements_reuse_spare_device_wires(wires: list[str] | list[int]) -> None:
    """Keep reset and feedback semantics when an unused wire is not last."""
    device = qp.device("mqt.ddsim.default", wires=wires)
    control, _, target = wires

    @qp.qnode(device, shots=5)
    def circuit() -> CountsMP:
        qp.X(control)
        measured = qp.measure(control, reset=True)
        qp.cond(measured, qp.X)(target)
        return qp.counts(wires=[target, control])

    assert circuit() == {"10": 5}


def test_device_requires_qnode_shots() -> None:
    """Do not submit a DDSIM job when a QNode omits its shot budget."""
    device = qp.device("mqt.ddsim.default", wires=1)

    @qp.qnode(device)
    def circuit() -> ExpectationMP:
        return qp.expval(qp.Z(0))

    with pytest.raises(PennyLaneValidationError, match="finite number of shots"):
        circuit()
    assert device.submitted_jobs == 0
    assert qp.set_shots(circuit, shots=5)() == pytest.approx(1.0)


def test_retained_native_job_outlives_adapter() -> None:
    """An exposed native attempt keeps its device session after the adapter is released."""
    device = DDSIMDevice(wires=1)
    tape = qp.tape.QuantumScript([qp.PauliX(0)], [qp.sample(wires=[0])], shots=4)
    device.execute(tape)
    batch = device.last_job
    assert batch is not None
    handle = batch.entries[0].attempts[0].handle
    assert handle is not None
    del batch, device
    gc.collect()
    assert handle.get_shots() == ["1"] * 4


@pytest.mark.parametrize("aggregate", [False, True])
def test_postprocessor_recovers_forward_measurements(monkeypatch: pytest.MonkeyPatch, *, aggregate: bool) -> None:
    """Retained PennyLane postprocessing reconstructs broadcasts, shot vectors, and measurements."""
    device = DDSIMDevice(wires=1, max_retries=3)

    @qp.qnode(device, shots=[(8, 2), 16], diff_method=None)
    def circuit(
        angles: np.ndarray,
    ) -> tuple[ExpectationMP, ProbabilityMP, VarianceMP] | tuple[ExpectationMP, ProbabilityMP, VarianceMP, SampleMP]:
        qp.RY(angles, 0)
        measurements = (qp.expval(qp.Z(0)), qp.probs(wires=[0]), qp.var(2 * qp.Z(0)))
        return measurements if aggregate else (*measurements, qp.sample(2 * qp.Z(0)))

    angles = np.array([0.0, np.pi])
    expected = circuit(angles)
    for partition in expected:
        np.testing.assert_array_equal(partition[2], [0, 0])
        if not aggregate:
            np.testing.assert_array_equal(partition[3][0], 2)
            np.testing.assert_array_equal(partition[3][1], -2)
    tapes, postprocess = qp.workflow.construct_batch(circuit, level="device")(angles)
    original = PennyLaneJob._readout  # ruff:ignore[private-member-access] Inject a failure after native submission.
    reads = []

    def read(batch: PennyLaneJob, index: int, job: Job, program_index: int) -> object:
        reads.append((job, program_index))
        if len(reads) == 2:
            raise KeyboardInterrupt
        return original(batch, index, job, program_index)

    monkeypatch.setattr(PennyLaneJob, "_readout", read)
    with pytest.raises(KeyboardInterrupt):
        device.execute(tapes)
    batch = device.last_job
    assert batch is not None
    submitted = device.submitted_jobs
    first_result = batch.entries[0].result
    assert first_result is not None
    recovered = postprocess(batch.result())[0]
    np.testing.assert_equal(recovered, expected)
    assert device.submitted_jobs == submitted
    assert batch.entries[0].result is first_result
    assert reads.count(reads[0]) == 1
