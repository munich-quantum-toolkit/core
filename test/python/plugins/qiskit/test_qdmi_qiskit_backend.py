# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for QiskitBackend."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.providers import JobStatus

from mqt.core import fomac
from mqt.core.plugins.qiskit import (
    CircuitValidationError,
    QDMIProvider,
    UnsupportedOperationError,
)

if TYPE_CHECKING:
    from test.python.plugins.qiskit.conftest import MockQDMIDevice

    from mqt.core.plugins.qiskit import QiskitBackend

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*Device operation.*cannot be mapped to a Qiskit gate.*:UserWarning"),
    pytest.mark.filterwarnings("ignore:Device does not define a measurement operation.*:UserWarning"),
]


def test_backend_instantiation(mock_backend: QiskitBackend) -> None:
    """Backend exposes target qubit count."""
    assert mock_backend.target.num_qubits > 0


def test_single_circuit_run_counts(mock_backend: QiskitBackend) -> None:
    """Running a circuit yields counts with specified shots."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure(0, 0)
    qc.measure(1, 1)
    job = mock_backend.run(qc, shots=256)
    counts = job.result().get_counts()

    # Verify total shots
    assert sum(counts.values()) == 256

    # Verify all keys are valid 2-bit binary strings
    for key in counts:
        assert len(key) == 2
        assert all(bit in {"0", "1"} for bit in key)


def test_unsupported_operation(mock_backend: QiskitBackend) -> None:
    """Unsupported operation raises UnsupportedOperationError."""
    qc = QuantumCircuit(1, 1)
    qc.x(0)  # 'x' not supported by mock device
    qc.measure(0, 0)
    with pytest.raises(UnsupportedOperationError):
        mock_backend.run(qc)


def test_backend_via_provider() -> None:
    """Backend can be obtained through provider."""
    provider = QDMIProvider()
    backend = provider.get_backend("MQT Core DDSIM QDMI Device")
    assert backend.target.num_qubits > 0
    assert backend.provider is provider


def test_provider_get_backend_by_name() -> None:
    """Provider can get backend by name."""
    provider = QDMIProvider()
    backend = provider.get_backend("MQT Core DDSIM QDMI Device")
    assert backend.name == "MQT Core DDSIM QDMI Device"


@pytest.mark.filterwarnings("ignore:Skipping device:UserWarning")
def test_provider_backends_list() -> None:
    """Provider can list all backends."""
    provider = QDMIProvider()
    backends = provider.backends()
    assert len(backends) > 0
    assert all(hasattr(b, "target") for b in backends)


def test_backend_max_circuits(mock_backend: QiskitBackend) -> None:
    """Backend max_circuits should return None (no limit)."""
    assert mock_backend.max_circuits is None


def test_backend_options(mock_backend: QiskitBackend) -> None:
    """Backend options should be accessible."""
    assert mock_backend.options.shots == 1024


def test_backend_run_with_shots_option(mock_backend: QiskitBackend) -> None:
    """Backend run should accept shots option."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = mock_backend.run(qc, shots=500)
    result = job.result()
    assert sum(result.get_counts().values()) == 500


def test_backend_run_with_invalid_shots_type(mock_backend: QiskitBackend) -> None:
    """Backend run should raise CircuitValidationError for invalid shots type."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    with pytest.raises(CircuitValidationError, match="Invalid 'shots' value"):
        mock_backend.run(qc, shots="invalid")


def test_backend_run_with_negative_shots(mock_backend: QiskitBackend) -> None:
    """Backend run should raise CircuitValidationError for negative shots."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    with pytest.raises(CircuitValidationError, match="'shots' must be >= 0"):
        mock_backend.run(qc, shots=-100)


def test_backend_circuit_with_parameters(mock_backend: QiskitBackend) -> None:
    """Backend should handle parameterized circuits correctly.

    Unbound parameters should raise CircuitValidationError, while bound parameters
    should execute successfully.
    """
    qc = QuantumCircuit(2, 2)
    theta = Parameter("theta")
    qc.ry(theta, 0)
    qc.measure([0, 1], [0, 1])

    # Unbound parameters should raise an error
    with pytest.raises(CircuitValidationError, match="Circuit contains unbound parameters"):
        mock_backend.run(qc)

    # Bound parameters should work
    qc_bound = qc.assign_parameters({theta: 1.5708})

    job = mock_backend.run(qc_bound, shots=100)
    result = job.result()
    assert result.success


def test_backend_circuit_with_parameter_expression(mock_backend: QiskitBackend) -> None:
    """Backend should raise CircuitValidationError for unbound parameter expressions."""
    qc = QuantumCircuit(2, 2)
    theta = Parameter("theta")
    phi = Parameter("phi")
    qc.ry(theta + phi, 0)  # Parameter expression
    qc.measure([0, 1], [0, 1])

    with pytest.raises(CircuitValidationError, match="Circuit contains unbound parameters"):
        mock_backend.run(qc)


def test_backend_circuit_with_barrier(mock_backend: QiskitBackend) -> None:
    """Backend should handle circuits with barriers."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.barrier()
    qc.measure([0, 1], [0, 1])

    job = mock_backend.run(qc, shots=100)
    result = job.result()
    assert result.success


def test_backend_circuit_with_single_qubit(mock_backend: QiskitBackend) -> None:
    """Backend should handle single-qubit circuits."""
    qc = QuantumCircuit(1, 1)
    qc.ry(1.5708, 0)
    qc.measure(0, 0)

    job = mock_backend.run(qc, shots=100)
    result = job.result()
    assert result.success


def test_backend_circuit_with_no_measurements(mock_backend: QiskitBackend) -> None:
    """Backend should handle circuits without measurements."""
    qc = QuantumCircuit(2)
    qc.cz(0, 1)

    job = mock_backend.run(qc, shots=100)
    result = job.result()
    assert result.success


def test_backend_named_circuit_results_queryable_by_name(mock_backend: QiskitBackend) -> None:
    """Backend should preserve circuit name and allow querying results by name."""
    qc = QuantumCircuit(2, 2, name="my_circuit")
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = mock_backend.run(qc, shots=100)
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


def test_backend_unnamed_circuit_results_queryable_by_generated_name(mock_backend: QiskitBackend) -> None:
    """Backend should generate a name for unnamed circuits and allow querying results by it."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = mock_backend.run(qc, shots=100)
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


def test_job_status(mock_backend: QiskitBackend) -> None:
    """Job should be in DONE status after backend.run() completes."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = mock_backend.run(qc, shots=100)
    assert job.status() == JobStatus.DONE


def test_job_result_success_and_shots(mock_backend: QiskitBackend) -> None:
    """Job result should contain success status and shot count for each circuit."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = mock_backend.run(qc, shots=100)
    result = job.result()

    assert result.success is True
    assert result.results is not None
    assert len(result.results) == 1
    assert result.results[0].shots == 100
    assert result.results[0].success is True


def test_job_get_counts_default(mock_backend: QiskitBackend) -> None:
    """result().get_counts() without arguments should return counts for first circuit."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = mock_backend.run(qc, shots=100)
    counts = job.result().get_counts()

    assert sum(counts.values()) == 100


def test_job_submit_raises_error(mock_backend: QiskitBackend) -> None:
    """Calling submit() on a job should raise NotImplementedError."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = mock_backend.run(qc, shots=100)
    with pytest.raises(NotImplementedError, match="You should never have to submit jobs"):
        job.submit()


def test_backend_warns_on_unmappable_operation(
    monkeypatch: pytest.MonkeyPatch, mock_qdmi_device_factory: type[MockQDMIDevice]
) -> None:
    """Backend should warn when device operation cannot be mapped to a Qiskit gate."""
    # Create mock device with an unmappable operation
    mock_device = mock_qdmi_device_factory(
        name="Test Device",
        num_qubits=2,
        operations=["cz", "custom_unmappable_gate", "measure"],
    )

    # Monkeypatch fomac.devices to return mock device
    def mock_devices() -> list[MockQDMIDevice]:
        return [mock_device]

    monkeypatch.setattr(fomac, "devices", mock_devices)

    # Creating backend should trigger warning about unmappable operation
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        provider = QDMIProvider()
        provider.get_backend("Test Device")

        # Check that the warning was raised
        assert len(w) > 0, "Expected at least one warning to be raised"
        warning_messages = [str(warning.message) for warning in w]
        assert any(
            "custom_unmappable_gate" in msg and "cannot be mapped to a Qiskit gate" in msg for msg in warning_messages
        ), f"Expected warning about custom_unmappable_gate, got: {warning_messages}"


def test_backend_warns_on_missing_measurement_operation(
    monkeypatch: pytest.MonkeyPatch, mock_qdmi_device_factory: type[MockQDMIDevice]
) -> None:
    """Backend should warn when device does not define a measurement operation."""
    # Create mock device without measure operation
    mock_device = mock_qdmi_device_factory(
        name="Test Device",
        num_qubits=2,
        operations=["cz"],  # No measure operation
    )

    # Monkeypatch fomac.devices to return mock device
    def mock_devices() -> list[MockQDMIDevice]:
        return [mock_device]

    monkeypatch.setattr(fomac, "devices", mock_devices)

    # Creating backend should trigger warning about missing measurement operation
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        provider = QDMIProvider()
        provider.get_backend("Test Device")

        # Check that the warning was raised
        assert len(w) > 0, "Expected at least one warning to be raised"
        warning_messages = [str(warning.message) for warning in w]
        assert any("does not define a measurement operation" in msg for msg in warning_messages), (
            f"Expected warning about missing measurement operation, got: {warning_messages}"
        )
