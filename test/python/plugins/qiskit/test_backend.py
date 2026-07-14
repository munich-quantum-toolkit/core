# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for QDMIBackend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import UnitaryGate
from qiskit.providers import JobStatus

from mqt.core import qdmi
from mqt.core.plugins.qiskit import (
    CircuitValidationError,
    QDMIBackend,
    UnsupportedOperationError,
)
from mqt.core.plugins.qiskit.exceptions import UnsupportedDeviceError

if TYPE_CHECKING:

    class _QDMIDeviceLike(Protocol):  # pragma: no cover - typing helper to fix mypy errors
        def name(self) -> str: ...

        def version(self) -> str: ...

        def qubits_num(self) -> int: ...

        def operations(self) -> list[object]: ...

        def coupling_map(self) -> object: ...

    SiteSpecificDevice = _QDMIDeviceLike
    MisconfiguredDevice = _QDMIDeviceLike
    ZonedDevice = _QDMIDeviceLike


@pytest.fixture
def ddsim_backend() -> QDMIBackend:
    """Get a QDMIBackend for the DDSIM device.

    Returns:
        A QDMIBackend instance wrapping the DDSIM device.
    """
    devices, _errors = qdmi.DeviceManager().open_all()
    for device in devices.values():
        if "DDSIM" in device.name():
            return QDMIBackend(device=device, provider=None)
    pytest.skip("DDSIM device not available")


def test_backend_instantiation(ddsim_backend: QDMIBackend) -> None:
    """Backend exposes target qubit count."""
    assert ddsim_backend.target.num_qubits > 0


def _single_qubit_circuit() -> QuantumCircuit:
    """Return a simple 1-qubit circuit with a parameterized rotation and measurement."""
    qc = QuantumCircuit(1)
    qc.ry(1.5708, 0)
    qc.measure_all()
    return qc


def _two_qubit_circuit() -> QuantumCircuit:
    """Return a simple 2-qubit circuit with a CZ gate and measurement."""
    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    qc.measure_all()
    return qc


def _two_qubit_barrier_circuit() -> QuantumCircuit:
    """Return a 2-qubit circuit with a CZ gate, a barrier, and measurement to test barrier handling."""
    qc = _two_qubit_circuit()
    qc.barrier()
    return qc


def _two_qubit_circuit_without_measurements() -> QuantumCircuit:
    """Return a 2-qubit circuit with a CZ gate to test measurement handling."""
    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    return qc


@pytest.fixture(
    params=[
        (_single_qubit_circuit, 100, True),
        (_two_qubit_circuit, 256, True),
        (_two_qubit_circuit, 500, True),
        (_two_qubit_barrier_circuit, 100, True),
        (_two_qubit_circuit_without_measurements, 100, False),
    ]
)
def backend_circuit_case(request: pytest.FixtureRequest) -> tuple[QuantumCircuit, int, bool]:
    """Provide reusable backend circuit cases.

    Returns:
        Tuple containing a prepared circuit, the shots to execute, and whether counts are expected.
    """
    builder, shots, expect_meas = request.param
    return builder(), shots, expect_meas


def test_backend_runs_variety_of_circuits(
    ddsim_backend: QDMIBackend, backend_circuit_case: tuple[QuantumCircuit, int, bool]
) -> None:
    """Backend executes multiple circuit shapes and shot counts consistently."""
    circuit, shots, expect_measurements = backend_circuit_case
    job = ddsim_backend.run(circuit, shots=shots)
    result = job.result()
    assert result.success is True

    if expect_measurements:
        counts = result.get_counts()
        assert sum(counts.values()) == shots
        for key in counts:
            assert all(bit in {"0", "1"} for bit in key)


def test_backend_runs_multiple_circuits(ddsim_backend: QDMIBackend) -> None:
    """Backend executes multiple circuits in a single run call."""
    # Create three different circuits
    qc1 = QuantumCircuit(2, name="bell_state")
    qc1.h(0)
    qc1.cx(0, 1)
    qc1.measure_all()

    qc2 = QuantumCircuit(2, name="x_then_measure")
    qc2.x(0)
    qc2.measure_all()

    qc3 = QuantumCircuit(2, name="hadamard_all")
    qc3.h([0, 1])
    qc3.measure_all()

    # Submit all circuits at once
    circuits = [qc1, qc2, qc3]
    job = ddsim_backend.run(circuits, shots=500)
    result = job.result()

    # Check overall success
    assert result.success is True

    # Check we have results for all circuits
    assert result.results is not None
    assert len(result.results) == 3

    # Check each circuit result
    for idx, expected_name in enumerate(["bell_state", "x_then_measure", "hadamard_all"]):
        exp_result = result.results[idx]
        assert exp_result.success is True
        # Support both dict-style (Qiskit 2.x) and object-style (Qiskit 1.x) header access
        header = exp_result.header
        circuit_name = header["name"] if isinstance(header, dict) else header.name
        assert circuit_name == expected_name
        assert exp_result.shots == 500

        # Check counts for this circuit
        counts = result.get_counts(idx)
        assert sum(counts.values()) == 500
        for key in counts:
            assert all(bit in {"0", "1"} for bit in key)
            assert len(key) == 2  # 2 qubits


def test_backend_runs_single_circuit_in_list(ddsim_backend: QDMIBackend) -> None:
    """Backend correctly handles a single circuit passed as a list."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    # Submit as a list with one element
    job = ddsim_backend.run([qc], shots=100)
    result = job.result()

    assert result.success is True
    assert result.results is not None
    assert len(result.results) == 1
    counts = result.get_counts()
    assert sum(counts.values()) == 100


def test_backend_runs_circuits_with_different_qubit_counts(ddsim_backend: QDMIBackend) -> None:
    """Backend correctly handles multiple circuits with different qubit counts."""
    # Create a 1-qubit circuit
    qc1 = QuantumCircuit(1, name="single_qubit")
    qc1.h(0)
    qc1.measure_all()

    # Create a 2-qubit circuit
    qc2 = QuantumCircuit(2, name="two_qubit")
    qc2.h(0)
    qc2.cx(0, 1)
    qc2.measure_all()

    # Run both circuits together
    circuits = [qc1, qc2]
    shots = 200
    job = ddsim_backend.run(circuits, shots=shots)
    result = job.result()

    # Check overall success
    assert result.success is True

    # Check we have results for both circuits
    assert result.results is not None
    assert len(result.results) == 2

    # Check first circuit (1-qubit)
    counts_1q = result.get_counts(0)
    assert sum(counts_1q.values()) == shots
    for bitstring in counts_1q:
        assert len(bitstring) == 1, f"Expected 1-qubit bitstring, got '{bitstring}' with length {len(bitstring)}"
        assert all(bit in {"0", "1"} for bit in bitstring)

    # Check second circuit (2-qubit)
    counts_2q = result.get_counts(1)
    assert sum(counts_2q.values()) == shots
    for bitstring in counts_2q:
        assert len(bitstring) == 2, f"Expected 2-qubit bitstring, got '{bitstring}' with length {len(bitstring)}"
        assert all(bit in {"0", "1"} for bit in bitstring)


def test_unsupported_operation(ddsim_backend: QDMIBackend) -> None:
    """Unsupported operation raises UnsupportedOperationError."""
    qc = QuantumCircuit(1, 1)
    # Use a custom gate that won't be in the device's operation set
    unitary = UnitaryGate(np.array([[1, 0], [0, 1]]), label="custom_unitary")
    qc.append(unitary, [0])
    qc.measure_all()
    with pytest.raises(UnsupportedOperationError):
        ddsim_backend.run(qc)


def test_backend_max_circuits(ddsim_backend: QDMIBackend) -> None:
    """Backend max_circuits should return None (no limit)."""
    assert ddsim_backend.max_circuits is None


def test_backend_options(ddsim_backend: QDMIBackend) -> None:
    """Backend options should be accessible."""
    assert ddsim_backend.options.shots == 1024


def test_backend_run_with_invalid_shots_type(ddsim_backend: QDMIBackend) -> None:
    """Backend run should raise CircuitValidationError for invalid shots type."""
    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    qc.measure_all()

    with pytest.raises(CircuitValidationError, match="Invalid 'shots' value"):
        ddsim_backend.run(qc, shots="invalid")


def test_backend_run_with_negative_shots(ddsim_backend: QDMIBackend) -> None:
    """Backend run should raise CircuitValidationError for negative shots."""
    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    qc.measure_all()

    with pytest.raises(CircuitValidationError, match="'shots' must be >= 0"):
        ddsim_backend.run(qc, shots=-100)


def test_backend_circuit_with_parameters(ddsim_backend: QDMIBackend) -> None:
    """Backend should reject circuits with unbound parameters."""
    qc = QuantumCircuit(2)
    theta = Parameter("theta")
    qc.ry(theta, 0)
    qc.measure_all()

    # Unbound parameters should raise an error
    with pytest.raises(CircuitValidationError, match=r"Circuit contains unbound parameters"):
        ddsim_backend.run(qc)


def test_backend_circuit_with_bound_parameters(ddsim_backend: QDMIBackend) -> None:
    """Backend should execute circuits with bound parameters."""
    qc = QuantumCircuit(2)
    theta = Parameter("theta")
    qc.ry(theta, 0)
    qc.measure_all()

    # Bound parameters should work
    qc_bound = qc.assign_parameters({theta: 1.5708})

    job = ddsim_backend.run(qc_bound, shots=100)
    result = job.result()
    assert result.success


def test_backend_circuit_with_parameter_expression(ddsim_backend: QDMIBackend) -> None:
    """Backend should raise CircuitValidationError for unbound parameter expressions."""
    qc = QuantumCircuit(2)
    theta = Parameter("theta")
    phi = Parameter("phi")
    qc.ry(theta + phi, 0)  # Parameter expression
    qc.measure_all()

    with pytest.raises(CircuitValidationError, match=r"Circuit contains unbound parameters"):
        ddsim_backend.run(qc)


def test_backend_run_with_parameter_values_dict(ddsim_backend: QDMIBackend) -> None:
    """Backend should bind parameters using parameter_values with dict format."""
    theta = Parameter("theta")
    qc = QuantumCircuit(2)
    qc.ry(theta, 0)
    qc.measure_all()

    # Pass parameter values as a dict
    job = ddsim_backend.run(qc, parameter_values=[{theta: 1.5708}], shots=100)
    result = job.result()
    assert result.success
    counts = result.get_counts()
    assert sum(counts.values()) == 100


def test_backend_run_with_parameter_values_list(ddsim_backend: QDMIBackend) -> None:
    """Backend should bind parameters using parameter_values with list format."""
    theta = Parameter("theta")
    qc = QuantumCircuit(2)
    qc.ry(theta, 0)
    qc.measure_all()

    # Pass parameter values as a list (in order of circuit.parameters)
    job = ddsim_backend.run(qc, parameter_values=[[1.5708]], shots=100)
    result = job.result()
    assert result.success
    counts = result.get_counts()
    assert sum(counts.values()) == 100


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


def test_backend_run_parameter_binding_failure(ddsim_backend: QDMIBackend) -> None:
    """Backend should raise error when parameters remain unbound after partial binding."""
    theta = Parameter("theta")
    phi = Parameter("phi")
    qc = QuantumCircuit(2)
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    qc.measure_all()

    # Pass incomplete parameter dict (missing phi) - Qiskit allows partial binding
    # but we'll catch the remaining unbound parameters
    with pytest.raises(CircuitValidationError, match="Circuit contains unbound parameters"):
        ddsim_backend.run(qc, parameter_values=[{theta: 0.5}])


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


def test_backend_run_missing_parameter_values(ddsim_backend: QDMIBackend) -> None:
    """Backend should raise error when circuits have parameters but no parameter_values provided."""
    theta = Parameter("theta")
    qc = QuantumCircuit(2)
    qc.ry(theta, 0)
    qc.measure_all()

    # Don't provide parameter_values
    with pytest.raises(
        CircuitValidationError, match=r"Circuit contains unbound parameters.*Provide `parameter_values`"
    ):
        ddsim_backend.run(qc)


def test_backend_named_circuit_results_queryable_by_name(ddsim_backend: QDMIBackend) -> None:
    """Backend should preserve circuit name and allow querying results by name."""
    qc = QuantumCircuit(2, name="my_circuit")
    qc.cz(0, 1)
    qc.measure_all()

    job = ddsim_backend.run(qc, shots=100)
    result = job.result()

    # Circuit name should be preserved in metadata
    assert result.results is not None
    header = result.results[0].header
    # Support both dict-style (pre-Qiskit 2.0) and object-style (post-Qiskit 2.0) access
    circuit_name = header["name"] if isinstance(header, dict) else header.name
    assert circuit_name == "my_circuit"

    # Should be able to query results by circuit name
    counts = result.get_counts("my_circuit")
    assert sum(counts.values()) == 100


def test_backend_unnamed_circuit_results_queryable_by_generated_name(ddsim_backend: QDMIBackend) -> None:
    """Backend should generate a name for unnamed circuits and allow querying results by it."""
    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    qc.measure_all()

    job = ddsim_backend.run(qc, shots=100)
    result = job.result()

    # Should have a generated name
    assert result.results is not None
    header = result.results[0].header
    # Support both dict-style (pre-Qiskit 2.0) and object-style (post-Qiskit 2.0) access
    if isinstance(header, dict):
        assert "name" in header
        circuit_name = header["name"]
    else:
        assert hasattr(header, "name")
        circuit_name = header.name

    # Should be able to query results by the generated name
    counts = result.get_counts(circuit_name)
    assert sum(counts.values()) == 100


def test_job_status(ddsim_backend: QDMIBackend) -> None:
    """Job should be in DONE status after backend.run() completes."""
    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    qc.measure_all()

    job = ddsim_backend.run(qc, shots=100)
    _ = job.result()  # Wait for job to complete
    assert job.status() == JobStatus.DONE


def test_job_result_success_and_shots(ddsim_backend: QDMIBackend) -> None:
    """Job result should contain success status and shot count for each circuit."""
    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    qc.measure_all()

    job = ddsim_backend.run(qc, shots=100)
    result = job.result()

    assert result.success is True
    assert result.results is not None
    assert len(result.results) == 1
    assert result.results[0].shots == 100
    assert result.results[0].success is True


def test_job_get_counts_default(ddsim_backend: QDMIBackend) -> None:
    """result().get_counts() without arguments should return counts for first circuit."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure_all()

    job = ddsim_backend.run(qc, shots=100)
    counts = job.result().get_counts()

    assert sum(counts.values()) == 100


def test_job_submit_raises_error(ddsim_backend: QDMIBackend) -> None:
    """Calling submit() on a job should raise NotImplementedError."""
    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    qc.measure_all()

    job = ddsim_backend.run(qc, shots=100)
    with pytest.raises(NotImplementedError, match="You should never have to submit jobs"):
        job.submit()


def test_backend_supports_cz_gate(ddsim_backend: QDMIBackend) -> None:
    """Backend executes CZ gate circuits and returns counts."""
    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    qc.measure_all()

    job = ddsim_backend.run(qc, shots=100)
    counts = job.result().get_counts()
    assert sum(counts.values()) == 100


def test_backend_supports_multicontrolled_gates(ddsim_backend: QDMIBackend) -> None:
    """Backend executes multi-controlled gates (e.g., MCX) and returns counts."""
    qc = QuantumCircuit(3)
    qc.mcx([0, 1], 2)  # Toffoli gate
    qc.mcp(1.5708, [0, 1], 2)  # Multi-controlled Phase
    qc.measure_all()

    job = ddsim_backend.run(qc, shots=100)
    counts = job.result().get_counts()
    assert sum(counts.values()) == 100


def test_backend_openqasm3_translation_works_for_native_gates(ddsim_backend: QDMIBackend) -> None:
    """Ensures the backend can run circuits with gates that are not natively supported by OpenQASM 3.

    The DDSIM backend defines support for `mcx` gates, which are not native to OpenQASM3.
    Qiskit's OpenQASM3 exporter has problems providing proper definitions for such gates,
    which we work around by declaring the device's basis gates in the export call.
    This test ensures that this workaround is effective and that the backend can successfully run such circuits.
    """
    qc = QuantumCircuit(6)
    qc.mcx([0, 1, 2, 3, 4], 5)
    qc.measure_all()

    job = ddsim_backend.run(qc, shots=100)
    counts = job.result().get_counts()
    assert sum(counts.values()) == 100


def test_zoned_operation_rejected_at_backend_init() -> None:
    """Backend rejects devices exposing zoned operations."""
    devices, _errors = qdmi.DeviceManager().open_all()
    for device in devices.values():
        if device.name().startswith("MQT NA"):
            with pytest.raises(UnsupportedDeviceError, match="cannot be represented in Qiskit's Target model"):
                QDMIBackend(device)
            return
    pytest.skip("NA device not available")
