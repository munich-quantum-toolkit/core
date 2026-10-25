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
from pathlib import Path
from typing import TYPE_CHECKING, get_type_hints

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, Clbit, Parameter
from qiskit.circuit.library import UnitaryGate
from qiskit.providers import JobStatus

from mqt.core.plugins.qiskit import (
    CircuitValidationError,
    QDMIBackend,
    UnsupportedOperationError,
    program_serializer,
)
from mqt.core.qdmi import ProgramFormat, open_device
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
    """Initialize nonempty registers once and loose bits individually before measurement."""
    qc = QuantumCircuit(1)
    qc.add_bits([Clbit()])
    qc.add_register(ClassicalRegister(0, "empty"), ClassicalRegister(1, "a"), ClassicalRegister(3, "b"))
    qc.measure(0, 3)
    original = qc.copy()

    serializer = program_serializer(ProgramFormat.QASM3)
    assert serializer is not None
    program = serializer(qc, ddsim_backend)

    assert isinstance(program, str)
    for assignment in ("a = 0;", "b = 0;", "_bit0 = false;"):
        assert program.count(assignment) == 1
        assert program.index(assignment) < program.index("b[1] = measure q[0];")
    assert "empty =" not in program
    assert program.count("= false;") == 1
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
