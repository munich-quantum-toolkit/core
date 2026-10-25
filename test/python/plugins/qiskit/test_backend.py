# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for QDMIBackend."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, get_type_hints

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ClassicalRegister, Gate, Parameter, QuantumRegister
from qiskit.circuit.classical import expr, types
from qiskit.circuit.library import UnitaryGate
from qiskit.providers import JobStatus

from mqt.core.plugins.qiskit import (
    CircuitValidationError,
    QDMIBackend,
    TranslationError,
    UnsupportedOperationError,
)
from mqt.core.qdmi import ProgramFormat, open_device
from mqt.core.qdmi.builtin_driver import open_device as open_default_device
from mqt.core.typing import QDMISessionParameters

if TYPE_CHECKING:
    from collections.abc import Iterable

    from mqt.core.qdmi import Device as QDMIDevice


@pytest.fixture
def ddsim_backend() -> QDMIBackend:
    """Get a QDMIBackend for the DDSIM device.

    Returns:
        A QDMIBackend instance wrapping the DDSIM device.
    """
    return QDMIBackend.from_device_id("mqt.ddsim.default")


def test_backend_from_device_id_forwards_session_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open one stable device ID with explicit per-session overrides."""
    observed: tuple[str, dict[str, object]] | None = None

    def fake_open_device(device_id: str, **session_parameters: object) -> QDMIDevice:
        nonlocal observed
        observed = device_id, session_parameters
        return open_device("mqt.ddsim.default")

    monkeypatch.setattr("mqt.core.plugins.qiskit.backend.open_device", fake_open_device)

    auth_file = Path("auth.json")
    session_parameters: QDMISessionParameters = {
        "driver_path": Path("driver.so"),
        "token": "token",
        "auth_file": auth_file,
        "auth_url": "https://auth.example",
        "username": "user",
        "password": "password",
        "base_url": "https://device.example",
        "device_config": "{}",
        "custom1": "one",
        "custom2": "two",
        "custom3": "three",
        "custom4": "four",
        "custom5": "five",
    }
    backend = QDMIBackend.from_device_id("test.device", **session_parameters)

    assert observed == ("test.device", session_parameters)
    assert backend.target.num_qubits > 0


def test_backend_from_device_id_rejects_unknown_session_parameter() -> None:
    """Reject unknown stable-ID factory keywords at runtime."""
    with pytest.raises(TypeError, match="unknown"):
        QDMIBackend.from_device_id("mqt.ddsim.default", unknown="value")


def test_qdmi_session_parameter_annotations_are_runtime_resolvable() -> None:
    """Expose the public TypedDict to runtime annotation consumers."""
    annotations = get_type_hints(QDMISessionParameters)
    assert annotations["auth_file"] == str | os.PathLike[str] | None


def test_backend_instantiation(ddsim_backend: QDMIBackend) -> None:
    """Backend exposes target qubit count."""
    assert ddsim_backend.target.num_qubits > 0
    assert ddsim_backend.device_id == "mqt.ddsim.default"
    assert ddsim_backend.max_circuits is None
    assert ddsim_backend.options.shots == 1024


@pytest.mark.parametrize("device_id", [None, "allocated.device"])
def test_direct_backend_device_id(device_id: str | None) -> None:
    """Direct construction retains an optional stable ID without opening it again."""
    backend = QDMIBackend(open_device("mqt.ddsim.default"), device_id=device_id)
    assert backend.device_id == device_id


@pytest.mark.parametrize(
    ("width", "barrier", "measure", "shots"),
    [(1, False, True, 100), (2, False, True, 256), (2, True, True, 500), (2, False, False, 100)],
)
def test_backend_runs_variety_of_circuits(
    ddsim_backend: QDMIBackend, width: int, *, barrier: bool, measure: bool, shots: int
) -> None:
    """Execute rotations, CZ, barriers, and circuits without classical outputs."""
    circuit = QuantumCircuit(width)
    if width == 1:
        circuit.ry(1.5708, 0)
    else:
        circuit.cz(0, 1)
    if measure:
        circuit.measure_all()
    if barrier:
        circuit.barrier()
    job = ddsim_backend.run(circuit, shots=shots)
    result = job.result()
    assert result.success
    assert job.status() == JobStatus.DONE
    assert result.results is not None
    assert result.results[0].success
    assert result.results[0].shots == shots
    if measure:
        counts = result.get_counts()
        assert sum(counts.values()) == shots
        assert all(len(bits) == width and not set(bits) - {"0", "1"} for bits in counts)


@pytest.mark.parametrize("widths", [(2,), (2, 2, 2), (1, 2)])
def test_backend_runs_circuit_lists(ddsim_backend: QDMIBackend, widths: tuple[int, ...]) -> None:
    """Preserve each experiment's name, width, order, and shots in native batches."""
    circuits = []
    for index, width in enumerate(widths):
        circuit = QuantumCircuit(width, name=f"circuit-{index}")
        for qubit in range(min(index, width)):
            circuit.x(qubit)
        circuit.measure_all()
        circuits.append(circuit)
    result = ddsim_backend.run(circuits, shots=100).result()
    assert result.success
    assert result.results is not None
    assert len(result.results) == len(circuits)
    for index, (circuit, experiment) in enumerate(zip(circuits, result.results, strict=True)):
        assert experiment.success
        assert experiment.header["name"] == circuit.name
        assert experiment.shots == 100
        expected = "0" * (circuit.num_qubits - min(index, circuit.num_qubits)) + "1" * min(index, circuit.num_qubits)
        assert result.get_counts(index) == {expected: 100}


def test_unsupported_operation(ddsim_backend: QDMIBackend) -> None:
    """Unsupported operation raises UnsupportedOperationError."""
    qc = QuantumCircuit(1, 1)
    # Use a custom gate that won't be in the device's operation set
    unitary = UnitaryGate(np.array([[1, 0], [0, 1]]), label="custom_unitary")
    qc.append(unitary, [0])
    qc.measure_all()
    with pytest.raises(UnsupportedOperationError):
        ddsim_backend.run(qc)


@pytest.mark.parametrize(("shots", "message"), [("invalid", "Invalid 'shots' value"), (-100, "'shots' must be >= 0")])
def test_backend_rejects_invalid_shots(ddsim_backend: QDMIBackend, shots: object, message: str) -> None:
    """Reject invalid shot types and negative counts before submission."""
    with pytest.raises(CircuitValidationError, match=message):
        ddsim_backend.run(QuantumCircuit(1), shots=shots)


@pytest.mark.parametrize("binding", ["missing", "expression", "partial"])
def test_backend_rejects_unbound_parameters(ddsim_backend: QDMIBackend, binding: str) -> None:
    """Both free parameters and partially bound expressions need explicit values."""
    theta, phi = Parameter("theta"), Parameter("phi")
    circuit = QuantumCircuit(1)
    circuit.ry(theta if binding == "missing" else theta + phi, 0)
    values = [{theta: 0.5}] if binding == "partial" else None
    with pytest.raises(
        CircuitValidationError, match=r"Circuit contains unbound parameters.*Provide `parameter_values`"
    ):
        ddsim_backend.run(circuit, parameter_values=values)


@pytest.mark.parametrize("binding", ["manual", "dict", "list"])
def test_backend_binds_parameters(ddsim_backend: QDMIBackend, binding: str) -> None:
    """Manual and per-run parameter binding preserve the bound rotation."""
    theta = Parameter("theta")
    circuit = QuantumCircuit(1)
    circuit.ry(theta, 0)
    circuit.measure_all()
    values = None
    if binding == "manual":
        circuit = circuit.assign_parameters({theta: np.pi})
    else:
        values = [{theta: np.pi}] if binding == "dict" else [[np.pi]]
    result = ddsim_backend.run(circuit, parameter_values=values, shots=100).result()
    assert result.success
    assert result.get_counts() == {"1": 100}
    assert result.get_counts(circuit) == {"1": 100}
    if binding != "manual":
        assert list(circuit.parameters) == [theta]


def test_backend_run_multiple_circuits_with_parameter_values(ddsim_backend: QDMIBackend) -> None:
    """Backend should bind different parameters to multiple circuits."""
    theta = Parameter("theta")

    # Create two parameterized circuits
    qc1 = QuantumCircuit(2, name="circuit1")
    qc1.ry(theta, 0)
    qc1.measure_all()

    qc2 = QuantumCircuit(2, name="circuit2")
    qc2.ry(theta, 0)
    qc2.measure_all()

    # Pass different parameter values for each circuit
    circuits = [qc1, qc2]
    param_values = [{theta: 0.5}, {theta: 1.5}]

    job = ddsim_backend.run(circuits, parameter_values=param_values, shots=200)
    result = job.result()

    assert result.success
    assert result.results is not None
    assert len(result.results) == 2

    for idx in range(2):
        counts = result.get_counts(idx)
        assert sum(counts.values()) == 200


def test_backend_run_empty_circuit_list(ddsim_backend: QDMIBackend) -> None:
    """Backend should raise error when empty circuit list is provided."""
    with pytest.raises(CircuitValidationError, match="No circuits provided to run"):
        ddsim_backend.run([])


def test_backend_run_parameter_values_length_mismatch(ddsim_backend: QDMIBackend) -> None:
    """Backend should raise error when parameter_values length doesn't match circuits."""
    theta = Parameter("theta")
    qc = QuantumCircuit(2)
    qc.ry(theta, 0)
    qc.measure_all()

    # Pass 2 parameter values for only 1 circuit
    with pytest.raises(CircuitValidationError, match=r"Length of parameter_values.*must match"):
        ddsim_backend.run(qc, parameter_values=[{theta: 0.5}, {theta: 1.5}])


def test_backend_run_mixed_parameterized_circuits(ddsim_backend: QDMIBackend) -> None:
    """Backend should handle mix of parameterized and non-parameterized circuits correctly."""
    theta = Parameter("theta")

    # Circuit with parameter
    qc1 = QuantumCircuit(2, name="parameterized")
    qc1.ry(theta, 0)
    qc1.measure_all()

    # Circuit without parameters
    qc2 = QuantumCircuit(2, name="non_parameterized")
    qc2.h(0)
    qc2.cx(0, 1)
    qc2.measure_all()

    # Provide parameter values for both (empty dict for non-parameterized)
    circuits = [qc1, qc2]
    param_values = [{theta: 1.0}, {}]

    job = ddsim_backend.run(circuits, parameter_values=param_values, shots=100)
    result = job.result()

    assert result.success
    assert result.results is not None
    assert len(result.results) == 2


def test_backend_run_raises_unused_parameter_values(ddsim_backend: QDMIBackend) -> None:
    """Backend should raise when parameter values provided for circuit without parameters."""
    theta = Parameter("theta")

    # Circuit without parameters
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.measure_all()

    # Provide parameter values for non-parameterized circuit
    with pytest.raises(CircuitValidationError, match="Failed to bind parameters for circuit"):
        ddsim_backend.run(qc, parameter_values=[{theta: 1.0}], shots=100)


@pytest.mark.parametrize("name", [None, "my_circuit"])
def test_backend_results_queryable_by_name(ddsim_backend: QDMIBackend, name: str | None) -> None:
    """Preserve explicit or generated circuit names for result lookup."""
    circuit = QuantumCircuit(2, name=name)
    circuit.cz(0, 1)
    circuit.measure_all()
    result = ddsim_backend.run(circuit, shots=100).result()
    assert result.results is not None
    assert result.results[0].header["name"] == circuit.name
    assert result.get_counts(circuit.name) == result.get_counts() == {"00": 100}


def test_backend_advertised_classical_feedback(ddsim_backend: QDMIBackend) -> None:
    """Advertised if/else executes with its measured classical condition."""
    assert "if_else" in ddsim_backend.target.operation_names
    circuit = QuantumCircuit(3, 3)
    circuit.h(0)
    circuit.measure(0, 0)
    with circuit.if_test((circuit.clbits[0], True)) as otherwise:
        circuit.x(1)
    with otherwise:
        circuit.x(2)
    circuit.measure([1, 2], [1, 2])
    counts = ddsim_backend.run(circuit, shots=128).result().get_counts(circuit)
    assert sum(counts.values()) == 128
    assert set(counts) == {"011", "100"}


def test_job_get_counts_default(ddsim_backend: QDMIBackend) -> None:
    """result().get_counts() without arguments should return counts for first circuit."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure_all()

    job = ddsim_backend.run(qc, shots=100)
    counts = job.result().get_counts()

    assert sum(counts.values()) == 100


def test_job_submit_preserves_existing_job(ddsim_backend: QDMIBackend) -> None:
    """Calling submit() with no untouched entries must preserve the native execution."""
    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    qc.measure_all()

    job = ddsim_backend.run(qc, shots=100)
    entries = job.entries
    job.submit()
    assert job.entries == entries
    assert job.result().success


def test_backend_supports_rccx_gate(ddsim_backend: QDMIBackend) -> None:
    """DDSIM backend exposes and executes native RCCX gate circuits."""
    assert "rccx" in ddsim_backend.target.operation_names

    qc = QuantumCircuit(3)
    qc.x(0)
    qc.x(1)
    qc.rccx(0, 1, 2)
    qc.measure_all()

    job = ddsim_backend.run(qc, shots=100)
    counts = job.result().get_counts()
    assert counts == {"111": 100}


def test_backend_supports_multicontrolled_gates(ddsim_backend: QDMIBackend) -> None:
    """Backend executes multi-controlled gates (e.g., MCX) and returns counts."""
    qc = QuantumCircuit(3)
    qc.mcx([0, 1], 2)  # Toffoli gate
    qc.mcp(1.5708, [0, 1], 2)  # Multi-controlled Phase
    qc.measure_all()

    job = ddsim_backend.run(qc, shots=100)
    counts = job.result().get_counts()
    assert sum(counts.values()) == 100


def test_backend_qasm3_zero_initializes_classical_bits(ddsim_backend: QDMIBackend) -> None:
    """Unwritten bits remain zero, including across measurement feedback."""
    qc = QuantumCircuit(QuantumRegister(1, "q"), ClassicalRegister(1, "a"), ClassicalRegister(3, "b"))
    with qc.if_test((qc.cregs[0], 0)):
        qc.x(0)
    qc.measure(0, 2)
    original = qc.copy()
    assert ddsim_backend.run(qc, shots=8).result().get_counts() == {"010 0": 8}
    assert qc == original


def test_backend_openqasm3_translation_works_for_native_gates(ddsim_backend: QDMIBackend) -> None:
    """Backend executes a six-qubit MCX circuit through its selected serializer."""
    qc = QuantumCircuit(6)
    qc.mcx([0, 1, 2, 3, 4], 5)
    qc.measure_all()

    job = ddsim_backend.run(qc, shots=100)
    counts = job.result().get_counts()
    assert sum(counts.values()) == 100


@pytest.mark.parametrize(("unit", "seconds"), [("s", 1.0), ("ms", 1e-3), ("us", 1e-6), ("ns", 1e-9)])
def test_sc_target_preserves_placements_and_physical_calibration(unit: str, seconds: float) -> None:
    """SC metadata retains ordered placements and converts raw durations to seconds."""
    config = {
        "schema-version": 1,
        "name": "Target calibration test",
        "numQubits": 4,
        "durationUnit": {"unit": unit, "scaleFactor": 0.5},
        "qubitProperties": {"defaults": {}, "overrides": []},
        "couplings": [[0, 1]],
        "operations": [
            {"name": "x", "numParameters": 0, "numQubits": 1, "duration": 20, "fidelity": 0.99},
            {"name": "cx", "numParameters": 0, "numQubits": 2, "sites": [[0, 1]], "duration": 40},
            {"name": "ccx", "numParameters": 0, "numQubits": 3, "sites": [[0, 1, 2]], "duration": 60, "fidelity": 0.95},
            {"name": "measure", "numParameters": 0, "numQubits": 1},
        ],
    }
    backend = QDMIBackend(open_default_device("mqt.sc.default", device_config=json.dumps(config)))
    target = backend.target
    assert target["x"][0,].duration == pytest.approx(10 * seconds)
    assert target["x"][0,].error == pytest.approx(0.01)
    assert target["cx"][0, 1].duration == pytest.approx(20 * seconds)
    assert set(target["ccx"]) == {(0, 1, 2)}
    assert target["ccx"][0, 1, 2].duration == pytest.approx(30 * seconds)
    assert target["ccx"][0, 1, 2].error == pytest.approx(0.05)
    assert not target.instruction_supported(operation_name="ccx", qargs=(1, 2, 3))
    assert target["measure"][0,] is None


@pytest.mark.parametrize(
    ("kind", "expected"), [("for", "101 0"), ("while", "101 0"), ("break", "100 0"), ("continue", "110 1")]
)
@pytest.mark.parametrize("compile_first", [False, True])
def test_backend_loops_and_nested_jumps(
    ddsim_backend: QDMIBackend, kind: str, expected: str, *, compile_first: bool
) -> None:
    """Loop jumps preserve nearest-loop semantics and asymmetric registers."""
    q = QuantumRegister(4, "q")
    flag, readout = ClassicalRegister(1, "flag"), ClassicalRegister(3, "readout")
    circuit = QuantumCircuit(q, flag, readout)
    circuit.x(3)
    if kind == "for":
        with circuit.for_loop(range(3)):
            circuit.x(2)
    elif kind == "while":
        circuit.x(0)
        circuit.measure(0, flag[0])
        with circuit.while_loop((flag[0], True)):
            circuit.x(2)
            circuit.reset(0)
            circuit.measure(0, flag[0])
    elif kind == "break":
        with circuit.for_loop(range(2)), circuit.for_loop(range(3)):
            circuit.x(2)
            circuit.break_loop()
            circuit.x(1)
    else:
        circuit.x(0)
        circuit.measure(0, flag[0])
        with circuit.for_loop(range(3)):
            with circuit.for_loop(range(2)):
                circuit.x(2)
                with circuit.if_test((flag[0], True)):
                    circuit.continue_loop()
                circuit.x(1)
            circuit.x(1)
    circuit.measure([2, 1, 3], readout)
    if compile_first:
        circuit = transpile(circuit, ddsim_backend, optimization_level=1)
    original = circuit.copy()
    result = ddsim_backend.run(circuit, shots=16, memory=True).result()
    assert circuit == original
    assert result.get_counts() == {expected: 16}
    assert result.get_memory() == [expected] * 16


def test_backend_for_loop_binds_external_parameters(ddsim_backend: QDMIBackend) -> None:
    """Bind circuit parameters while retaining the loop's local induction value."""
    theta, index = Parameter("theta"), Parameter("index")
    circuit = QuantumCircuit(2, 2)
    with circuit.for_loop(range(1, 4, 2), index):
        circuit.rx(theta * index, 1)
    circuit.measure([0, 1], [0, 1])
    result = ddsim_backend.run(circuit, parameter_values=[{theta: np.pi / 4}], shots=16).result()
    assert result.get_counts() == {"10": 16}


@pytest.mark.parametrize(
    ("value", "readout_name", "expected"),
    [(0, "readout", "01 01"), (1, "if", "10 00"), (2, "測定", "01 11"), (3, "readout", "11 11")],
)
@pytest.mark.parametrize("compile_first", [False, True])
def test_backend_switch_preserves_only_register_outputs(
    ddsim_backend: QDMIBackend, value: int, readout_name: str, expected: str, *, compile_first: bool
) -> None:
    """Nested switches snapshot their target and exclude temporary integers from results."""
    selector = ClassicalRegister(2, "switch_dummy")
    readout = ClassicalRegister(2, readout_name)
    circuit = QuantumCircuit(QuantumRegister(4, "q"), selector, readout)
    for bit in range(2):
        if value & (1 << bit):
            circuit.x(bit)
    circuit.measure([0, 1], selector)
    with circuit.for_loop(range(1)), circuit.switch(selector) as case:
        with case(0, 2):
            circuit.x(2)
            circuit.measure(2, selector[0])
        with case(1):
            circuit.x(3)
            circuit.reset(0)
            circuit.measure(0, selector[0])
        with case(case.DEFAULT):
            circuit.x(2)
            circuit.x(3)
    circuit.measure([2, 3], readout)
    if compile_first:
        circuit = transpile(circuit, ddsim_backend, optimization_level=1)
    original = circuit.copy()
    result = ddsim_backend.run(circuit, shots=16, memory=True).result()
    assert circuit == original
    assert result.get_counts() == {expected: 16}
    assert result.get_memory() == [expected] * 16
    sampled = ddsim_backend.sampler(default_shots=16).run([circuit]).result()[0]
    assert sampled.data[readout_name].get_counts() == {expected.split(maxsplit=1)[0]: 16}
    assert sampled.data[selector.name].get_counts() == {expected.split()[1]: 16}


@pytest.mark.parametrize("compile_first", [False, True])
def test_backend_classical_assignments(ddsim_backend: QDMIBackend, *, compile_first: bool) -> None:
    """Stores update captured locals and output bits through the public backend."""
    flag, output = ClassicalRegister(1, "flag"), ClassicalRegister(3, "output")
    circuit = QuantumCircuit(QuantumRegister(1), flag, output)
    counter = circuit.add_var("counter", expr.Value(0, types.Uint(3)))
    circuit.store(
        flag[0],
        True,  # ruff: ignore[boolean-positional-value-in-call] Qiskit Store arguments are positional-only.
    )
    with circuit.while_loop(expr.less(counter, 2)):
        circuit.store(counter, expr.add(counter, 1))
    with circuit.if_test((flag[0], True)):
        circuit.store(output, counter)
        circuit.store(
            expr.index(output, 0),
            True,  # ruff: ignore[boolean-positional-value-in-call] Qiskit Store arguments are positional-only.
        )
        circuit.x(0)
    circuit.measure(0, flag[0])
    if compile_first:
        circuit = transpile(circuit, ddsim_backend, optimization_level=1)
    result = ddsim_backend.run(circuit, shots=8, memory=True).result()
    assert result.get_counts() == {"011 1": 8}
    assert result.get_memory() == ["011 1"] * 8


@pytest.mark.parametrize(
    "fmt",
    [
        ProgramFormat.QIR_BASE_STRING,
        ProgramFormat.QIR_BASE_MODULE,
        ProgramFormat.QIR_ADAPTIVE_STRING,
        ProgramFormat.QIR_ADAPTIVE_MODULE,
    ],
)
def test_qir_serializers_preserve_physical_sites_and_outputs(
    ddsim_backend: QDMIBackend, monkeypatch: pytest.MonkeyPatch, fmt: ProgramFormat
) -> None:
    """QIR keeps unused physical holes, phase, and asymmetric register outputs."""
    circuit = QuantumCircuit(QuantumRegister(3), ClassicalRegister(1, "a"), ClassicalRegister(1, "b"))
    circuit.global_phase = 0.125
    gate = Gate("x", 1, [])
    gate.definition = QuantumCircuit(1)
    gate.definition.z(0)
    circuit.append(gate, [2])
    circuit.x(2)
    circuit.x(2)
    circuit.cx(0, 2)
    circuit.measure([2, 0], [0, 1])
    serialize = ddsim_backend._serialize_circuit  # ruff: ignore[private-member-access] Select one advertised format.
    payloads: list[str | bytes] = []

    def select_format(circuit: QuantumCircuit, _formats: Iterable[ProgramFormat]) -> tuple[str | bytes, ProgramFormat]:
        payload, selected = serialize(circuit, [fmt])
        payloads.append(payload)
        return payload, selected

    monkeypatch.setattr(ddsim_backend, "_serialize_circuit", select_format)
    result = ddsim_backend.run(circuit, shots=8, memory=True).result()
    assert result.get_counts() == {"0 1": 8}
    assert result.get_memory() == ["0 1"] * 8
    payload = payloads[0]
    if isinstance(payload, str):
        assert re.search(r"call void @__quantum__qis__x__body\(ptr inttoptr \(i64 2 to ptr\)\)", payload)
        assert "call void @__quantum__qis__gphase__body(double 1.250000e-01)" in payload
        assert payload.count("call void @__quantum__qis__x__body(") == 3
        assert "__quantum__qis__z__body" not in payload
    else:
        assert payload.startswith(b"BC\xc0\xde")


def test_iqm_serializer_preserves_native_sites_and_readout() -> None:
    """IQM payloads use device site names and classical output order."""
    backend = QDMIBackend.from_device_id("mqt.sc.iqm.garnet")
    circuit = QuantumCircuit(QuantumRegister(4), ClassicalRegister(1, "a"), ClassicalRegister(1, "b"))
    circuit.global_phase = 0.125
    circuit.r(0.731, -0.419, 3)
    circuit.cz(0, 3)
    circuit.measure([3, 0], [1, 0])
    payload, fmt = backend._serialize_circuit(circuit, [ProgramFormat.IQM_JSON])  # ruff: ignore[private-member-access]
    assert fmt == ProgramFormat.IQM_JSON
    assert isinstance(payload, str)
    instructions = json.loads(payload)["instructions"]
    names = [site.name() for site in backend.device.sites()]
    assert instructions[0] == {"name": "prx", "locus": [names[3]], "args": {"angle": 0.731, "phase": -0.419}}
    assert instructions[1] == {"name": "cz", "locus": [names[0], names[3]], "args": {}}
    assert [instruction["locus"] for instruction in instructions[2:]] == [[names[0]], [names[3]]]
    assert [instruction["name"] for instruction in instructions] == ["prx", "cz", "measure", "measure"]


@pytest.mark.parametrize("kind", ["partial", "repeated", "feedback"])
def test_iqm_serializer_rejects_unsupported_readout(kind: str) -> None:
    """IQM readout must fit the compiler exporter's terminal measurement contract."""
    backend = QDMIBackend.from_device_id("mqt.sc.iqm.garnet")
    circuit = QuantumCircuit(2, 2)
    circuit.measure(0, 0)
    if kind == "repeated":
        circuit.measure(0, 1)
    elif kind == "feedback":
        circuit.r(0.5, 0.25, 0)
        circuit.measure(1, 1)
    with pytest.raises(TranslationError):
        backend._serialize_circuit(circuit, [ProgramFormat.IQM_JSON])  # ruff: ignore[private-member-access]
