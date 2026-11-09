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
import struct
from pathlib import Path
from typing import TYPE_CHECKING, cast, get_type_hints
from unittest.mock import MagicMock

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ClassicalRegister, Clbit, Parameter, QuantumRegister
from qiskit.circuit.classical import expr, types
from qiskit.circuit.library import UnitaryGate
from qiskit.providers import JobStatus
from test_mock_backend import MockQDMIDevice

from mqt.core.mlir import (
    CompiledProgram,
    CompilerTarget,
    PayloadSpecification,
    TargetEnvironment,
    sample,
    serialize_program,
)
from mqt.core.plugins.qiskit import (
    CircuitValidationError,
    QDMIBackend,
    TranslationError,
    UnsupportedOperationError,
)
from mqt.core.qdmi import Job, ProgramFormat, open_device
from mqt.core.qdmi.builtin_driver import open_device as open_default_device
from mqt.core.typing import QDMISessionParameters

if TYPE_CHECKING:
    from mqt.core.qdmi import Device as QDMIDevice


@pytest.fixture
def ddsim_backend() -> QDMIBackend:
    """Get a QDMIBackend for the DDSIM device.

    Returns:
        A QDMIBackend instance wrapping the DDSIM device.
    """
    return QDMIBackend.from_device_id("mqt.ddsim.default")


@pytest.mark.parametrize("memory", [False, True])
def test_native_readout_preserves_wide_classical_outputs(*, memory: bool) -> None:
    """Native hex conversion retains projected bits above the machine word size."""
    backend = QDMIBackend.from_device_id("mqt.ddsim.default", custom1="1")
    circuit = QuantumCircuit(1, 81)
    circuit.x(0)
    circuit.measure(0, 80)
    result = backend.run(circuit, shots=8, memory=memory).result()
    assert result.data()["counts"] == {hex(1 << 80): 8}
    if memory:
        assert result.data()["memory"] == [hex(1 << 80)] * 8


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


def test_backend_qasm3_reconstructs_classical_bits(ddsim_backend: QDMIBackend) -> None:
    """Unwritten register and loose bits remain zero alongside a partial measurement."""
    qc = QuantumCircuit(1)
    qc.add_bits([Clbit()])
    qc.add_register(ClassicalRegister(1, "a"), ClassicalRegister(3, "b"))
    qc.x(0)
    qc.measure(0, 3)
    original = qc.copy()

    program = serialize_program(
        qc, target=TargetEnvironment.from_device(ddsim_backend.device, program_format=ProgramFormat.QASM3)
    )

    assert isinstance(program, CompiledProgram)
    assert program.result_layout is not None
    assert sample(program.payload, shots=8) == {"1": 8}
    assert program.result_layout.decode_bits("1") == "01000"
    assert qc == original


def test_backend_openqasm3_translation_works_for_native_gates(ddsim_backend: QDMIBackend) -> None:
    """Backend executes a six-qubit MCX circuit through its selected serializer."""
    qc = QuantumCircuit(6)
    qc.mcx([0, 1, 2, 3, 4], 5)
    qc.measure_all()

    job = ddsim_backend.run(qc, shots=100)
    counts = job.result().get_counts()
    assert sum(counts.values()) == 100


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_sc_target_preserves_placements_and_physical_calibration(unit: str) -> None:
    """Hardware-only model snapshots retain placements and raw calibration units."""
    config = {
        "schema-version": 1,
        "name": "Target calibration test",
        "numQubits": 4,
        "durationUnit": {"unit": unit, "scaleFactor": 0.5},
        "qubitProperties": {"defaults": {}, "overrides": []},
        "couplings": [[0, 1], [1, 2], [2, 3]],
        "operations": [
            {"name": "x", "numParameters": 0, "numQubits": 1, "duration": 20, "fidelity": 0.99},
            {"name": "cx", "numParameters": 0, "numQubits": 2, "sites": [[0, 1]], "duration": 40},
            {"name": "ccx", "numParameters": 0, "numQubits": 3, "sites": [[0, 1, 2]], "duration": 60, "fidelity": 0.95},
            {"name": "measure", "numParameters": 0, "numQubits": 1},
        ],
    }
    target = CompilerTarget.from_device(open_default_device("mqt.sc.default", device_config=json.dumps(config)))
    operations = {operation.name: operation for operation in target.operations}
    assert target.duration_unit is not None
    assert target.duration_unit.unit == unit
    assert target.duration_unit.scale_factor == pytest.approx(0.5)
    assert operations["x"].duration == 20
    assert operations["x"].fidelity == pytest.approx(0.99)
    assert operations["cx"].duration == 40
    assert [placement.sites for placement in operations["ccx"].site_tuples] == [[0, 1, 2]]
    assert operations["ccx"].duration == 60
    assert operations["ccx"].fidelity == pytest.approx(0.95)
    assert operations["measure"].duration is None


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
def test_qir_payloads_execute_register_order(
    ddsim_backend: QDMIBackend, monkeypatch: pytest.MonkeyPatch, fmt: ProgramFormat
) -> None:
    """Both QIR encodings retain asymmetric output registers and bound angles."""
    monkeypatch.setattr(
        ddsim_backend, "_environment", TargetEnvironment.from_device(ddsim_backend.device, program_format=fmt)
    )
    circuit = QuantumCircuit(QuantumRegister(3, "q"), ClassicalRegister(1, "a"), ClassicalRegister(2, "b"))
    angle = Parameter("theta")
    circuit.rx(angle, 2)
    circuit.cx(2, 1)
    circuit.measure([2, 1, 0], [0, 1, 2])
    result = ddsim_backend.run(circuit, parameter_values=[[np.pi]], shots=16).result()
    assert result.get_counts() == {"01 1": 16}


@pytest.mark.parametrize("fmt", [ProgramFormat.QIR_BASE_STRING, ProgramFormat.QIR_ADAPTIVE_STRING])
def test_qir_serializer_profile_controls_feedback(
    ddsim_backend: QDMIBackend, monkeypatch: pytest.MonkeyPatch, fmt: ProgramFormat
) -> None:
    """Adaptive executes feedback; Base reports its restricted profile."""
    monkeypatch.setattr(
        ddsim_backend, "_environment", TargetEnvironment.from_device(ddsim_backend.device, program_format=fmt)
    )
    circuit = QuantumCircuit(1, 2)
    circuit.x(0)
    circuit.measure(0, 0)
    with circuit.if_test((circuit.clbits[0], True)):
        circuit.x(0)
        circuit.barrier(0)
    circuit.measure(0, 1)
    if fmt == ProgramFormat.QIR_BASE_STRING:
        with pytest.raises(TranslationError, match=r"failed to legalize operation 'scf\.if'"):
            ddsim_backend.run(circuit, shots=8)
    else:
        assert ddsim_backend.run(circuit, shots=8).result().get_counts() == {"01": 8}


def test_iqm_serializer_uses_device_site_names() -> None:
    """The fallback resolves device names and preserves the native PRX angles."""
    device = open_device("mqt.sc.iqm.garnet")
    circuit = QuantumCircuit(1, 1, global_phase=0.7)
    circuit.r(0.731, -0.419, 0)
    circuit.barrier(0)
    circuit.measure(0, 0)
    payload = serialize_program(
        circuit,
        target=TargetEnvironment(
            CompilerTarget.from_device(device), PayloadSpecification.from_program_format(ProgramFormat.IQM_JSON)
        ),
    )
    assert isinstance(payload, CompiledProgram)
    instructions = json.loads(payload.payload)["instructions"]
    site = device.sites()[0].name()
    assert instructions == [
        {"name": "prx", "locus": [site], "args": {"angle": 0.731, "phase": -0.419}},
        {"name": "measure", "locus": [site], "args": {"key": "c_1_0_0"}},
    ]


@pytest.mark.parametrize("fmt", [ProgramFormat.QASM3, ProgramFormat.QIR_BASE_STRING, ProgramFormat.QIR_ADAPTIVE_STRING])
def test_native_submission_preserves_folded_unitary_sequence(
    ddsim_backend: QDMIBackend, monkeypatch: pytest.MonkeyPatch, fmt: ProgramFormat
) -> None:
    """Inspect the payload actually handed to the local device, not just its ideal result."""
    payloads: list[str] = []

    def capture(circuit: QuantumCircuit, *, target: TargetEnvironment) -> CompiledProgram:
        program = serialize_program(circuit, target=target)
        assert isinstance(program, CompiledProgram)
        assert isinstance(program.payload, str)
        payloads.append(program.payload)
        return program

    monkeypatch.setattr(
        ddsim_backend, "_environment", TargetEnvironment.from_device(ddsim_backend.device, program_format=fmt)
    )
    monkeypatch.setattr("mqt.core.plugins.qiskit.backend.serialize_program", capture)
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.h(0)
    circuit.rx(0.0, 0)
    circuit.rx(1e-16, 0)
    circuit.barrier(0, 1)
    circuit.x(1)
    circuit.measure([1, 0], [0, 1])
    assert ddsim_backend.run(circuit, shots=16).result().get_counts() == {"01": 16}
    assert len(payloads) == 1
    pattern = r"^\s*(h|rx|x)\b" if fmt == ProgramFormat.QASM3 else r"call void @__quantum__qis__(h|rx|x)__body"
    assert re.findall(pattern, payloads[0], re.MULTILINE) == ["h", "h", "rx", "rx", "x"]
    if fmt == ProgramFormat.QASM3:
        rotations = re.findall(r"rx\(([^)]+)\) \$0;", payloads[0])
        assert [float(angle) for angle in rotations] == [0.0, 1e-16]
        assert "x $1;" in payloads[0]
    else:
        rotations = re.findall(r"@__quantum__qis__rx__body\(double ([^,]+), ptr null\)", payloads[0])
        assert len(rotations) == 2
        angles = [
            struct.unpack(">d", bytes.fromhex(token.removeprefix("f").removeprefix("0x")))[0]
            if "0x" in token
            else float(token)
            for token in rotations
        ]
        assert angles == [0.0, 1e-16]
        assert "@__quantum__qis__x__body(ptr inttoptr (i64 1 to ptr))" in payloads[0]


@pytest.mark.usefixtures("mock_device_snapshot")
def test_resonator_compilation_uses_native_device_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """A native star snapshot compiles and submits with permuted result decoding."""
    config = {
        "schema-version": 1,
        "name": "Offline star",
        "numQubits": 3,
        "durationUnit": {"unit": "ns", "scaleFactor": 1},
        "qubitProperties": {
            "defaults": {},
            "overrides": [{"qubit": 0, "name": "CR"}, {"qubit": 1, "name": "QB1"}, {"qubit": 2, "name": "QB2"}],
        },
        "couplings": [[1, 0], [2, 0], [0, 1], [0, 2]],
        "operations": [
            {"name": "r", "numParameters": 2, "numQubits": 1, "sites": [[1], [2]]},
            {"name": "measure", "numParameters": 0, "numQubits": 1, "sites": [[1], [2]]},
            {"name": "cz", "numParameters": 0, "numQubits": 2, "sites": [[0, 1], [0, 2]]},
            {"name": "move", "numParameters": 0, "numQubits": 2, "sites": [[1, 0], [2, 0]]},
        ],
    }
    device = open_default_device("mqt.sc.default", device_config=json.dumps(config))
    environment = TargetEnvironment(
        CompilerTarget.from_device(device), PayloadSpecification.from_program_format(ProgramFormat.IQM_JSON)
    )
    transport = MockQDMIDevice(num_qubits=3, operations=["r", "cz", "measure"])
    monkeypatch.setattr(TargetEnvironment, "from_device", lambda *_args, **_kwargs: environment)
    submitted = []

    def submit(payload: str, program_format: ProgramFormat, num_shots: int) -> MagicMock:
        assert program_format == ProgramFormat.IQM_JSON
        submitted.append(payload)
        job = MagicMock()
        job.id = "offline-star"
        job.num_shots = num_shots
        job.check.return_value = Job.Status.DONE
        job.get_program_status.return_value = None
        job.get_counts.return_value = {"01": num_shots}
        return job

    monkeypatch.setattr(transport, "submit_job", submit)
    backend = QDMIBackend(cast("QDMIDevice", transport))
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.h(1)
    circuit.cz(0, 1)
    circuit.measure([0, 1], [1, 0])
    compiled = backend.compile(circuit)
    assert isinstance(compiled, CompiledProgram)
    result = backend.run(compiled, shots=3).result()
    assert submitted == [compiled.payload]
    assert result.get_counts() == {"10": 3}
    instructions = json.loads(submitted[0])["instructions"]
    assert sum(instruction["name"] == "move" for instruction in instructions) == 2
