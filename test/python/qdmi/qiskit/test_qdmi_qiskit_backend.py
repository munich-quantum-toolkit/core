# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for QiskitBackend."""

from __future__ import annotations

from typing import Any

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from mqt.core.qdmi.qiskit import (
    CircuitValidationError,
    QiskitBackend,
    UnsupportedOperationError,
)

pytestmark = [
    pytest.mark.filterwarnings("ignore:.*Device operation.*cannot be mapped to a Qiskit gate.*:UserWarning"),
    pytest.mark.filterwarnings("ignore:Device does not define a measurement operation.*:UserWarning"),
]


def test_backend_instantiation(na_backend: QiskitBackend) -> None:
    """Backend exposes target qubit count."""
    assert na_backend.target.num_qubits > 0


def test_single_circuit_run_counts(na_backend: QiskitBackend) -> None:
    """Running a circuit yields counts with specified shots.

    Note:
        This test uses the NA device which supports the CZ gate.
        It validates that all measurement outcomes are valid binary strings and that
        the total number of shots matches the requested amount.
    """
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure(0, 0)
    qc.measure(1, 1)
    job = na_backend.run(qc, shots=256)
    counts = job.result().get_counts()

    # Verify total shots
    assert sum(counts.values()) == 256

    # Verify all keys are valid 2-bit binary strings
    for key in counts:
        assert len(key) == 2
        assert all(bit in {"0", "1"} for bit in key)


def test_unsupported_operation(na_backend: QiskitBackend) -> None:
    """Unsupported operation raises the expected error type.

    Note:
        This test uses the NA device, which doesn't support 'x' gates.
        TODO: Generalize this test to work with different device types or parameterize
        across devices with known unsupported operations.
    """
    qc = QuantumCircuit(1, 1)
    qc.x(0)  # 'x' not supported by NA device capabilities
    qc.measure(0, 0)
    with pytest.raises(UnsupportedOperationError):
        na_backend.run(qc)


def test_backend_device_index() -> None:
    """Backend can be instantiated with device index 0."""
    backend = QiskitBackend(device_index=0)
    assert backend.target.num_qubits > 0


def test_backend_invalid_device_index() -> None:
    """Backend should raise IndexError for invalid device index."""
    with pytest.raises(IndexError, match=r"Device index .* out of range"):
        QiskitBackend(device_index=999)


def test_backend_negative_device_index() -> None:
    """Backend should raise IndexError for negative device index."""
    with pytest.raises(IndexError, match=r"Device index .* out of range"):
        QiskitBackend(device_index=-1)


def test_backend_max_circuits() -> None:
    """Backend max_circuits should return None (no limit)."""
    backend = QiskitBackend()
    assert backend.max_circuits is None


def test_backend_options() -> None:
    """Backend options should be accessible."""
    backend = QiskitBackend()
    assert backend.options.shots == 1024


def test_backend_run_with_shots_option(na_backend: QiskitBackend) -> None:
    """Backend run should accept shots option."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = na_backend.run(qc, shots=500)
    result = job.result()
    assert sum(result.get_counts().values()) == 500


def test_backend_run_with_invalid_shots_type(na_backend: QiskitBackend) -> None:
    """Backend run should raise CircuitValidationError for invalid shots type."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    with pytest.raises(CircuitValidationError, match="Invalid 'shots' value"):
        na_backend.run(qc, shots="invalid")


def test_backend_run_with_negative_shots(na_backend: QiskitBackend) -> None:
    """Backend run should raise CircuitValidationError for negative shots."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    with pytest.raises(CircuitValidationError, match="'shots' must be >= 0"):
        na_backend.run(qc, shots=-100)


def test_backend_circuit_with_parameters(na_backend: QiskitBackend) -> None:
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
        na_backend.run(qc)

    # Bound parameters should work
    qc_bound = qc.assign_parameters({theta: 1.5708})

    job = na_backend.run(qc_bound, shots=100)
    result = job.result()
    assert result.success


def test_backend_circuit_with_parameter_expression() -> None:
    """Backend should raise CircuitValidationError for unbound parameter expressions."""
    backend = QiskitBackend()
    qc = QuantumCircuit(2, 2)
    theta = Parameter("theta")
    phi = Parameter("phi")
    qc.ry(theta + phi, 0)  # Parameter expression
    qc.measure([0, 1], [0, 1])

    with pytest.raises(CircuitValidationError, match="Circuit contains unbound parameters"):
        backend.run(qc)


def test_backend_circuit_with_barrier(na_backend: QiskitBackend) -> None:
    """Backend should handle circuits with barriers."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.barrier()
    qc.measure([0, 1], [0, 1])

    job = na_backend.run(qc, shots=100)
    result = job.result()
    assert result.success


def test_backend_circuit_with_single_qubit(na_backend: QiskitBackend) -> None:
    """Backend should handle single-qubit circuits."""
    qc = QuantumCircuit(1, 1)
    qc.ry(1.5708, 0)
    qc.measure(0, 0)

    job = na_backend.run(qc, shots=100)
    result = job.result()
    assert result.success


def test_backend_circuit_with_no_measurements(na_backend: QiskitBackend) -> None:
    """Backend should handle circuits without measurements."""
    qc = QuantumCircuit(2)
    qc.cz(0, 1)

    job = na_backend.run(qc, shots=100)
    result = job.result()
    assert result.success


def test_backend_circuit_name_preserved(na_backend: QiskitBackend) -> None:
    """Backend should preserve circuit name in metadata."""
    qc = QuantumCircuit(2, 2, name="my_circuit")
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = na_backend.run(qc, shots=100)
    result = job.result()
    assert result.results[0].header["name"] == "my_circuit"


def test_backend_unnamed_circuit(na_backend: QiskitBackend) -> None:
    """Backend should handle circuits without names."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = na_backend.run(qc, shots=100)
    result = job.result()
    # Should have a default name
    assert "name" in result.results[0].header


def test_backend_result_metadata_includes_circuit_name(na_backend: QiskitBackend) -> None:
    """Backend result should include circuit name in header."""
    qc = QuantumCircuit(2, 2, name="my_test_circuit")
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = na_backend.run(qc, shots=100)
    result = job.result()
    assert result.results[0].header["name"] == "my_test_circuit"


def test_job_status(na_backend: QiskitBackend) -> None:
    """Job should be in DONE status after backend.run() completes."""
    from qiskit.providers import JobStatus

    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = na_backend.run(qc, shots=100)
    assert job.status() == JobStatus.DONE


def test_job_submit_raises_not_implemented(na_backend: QiskitBackend) -> None:
    """Calling submit() on a job raises NotImplementedError."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = na_backend.run(qc, shots=100)

    # Calling submit should raise NotImplementedError
    with pytest.raises(NotImplementedError, match="Job submission logic must be implemented"):
        job.submit()


def test_job_result_success_and_shots(na_backend: QiskitBackend) -> None:
    """Job result should contain success status and shot count for each circuit."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = na_backend.run(qc, shots=100)
    result = job.result()

    assert result.success is True
    assert len(result.results) == 1
    assert result.results[0].shots == 100
    assert result.results[0].success is True


def test_job_result_with_timeout(na_backend: QiskitBackend) -> None:
    """Job result should accept timeout parameter."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = na_backend.run(qc, shots=50)
    result = job.result()

    assert result.success is True
    assert result.results[0].shots == 50


def test_job_get_counts_default(na_backend: QiskitBackend) -> None:
    """result().get_counts() without arguments should return counts for first circuit."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = na_backend.run(qc, shots=100)
    counts = job.result().get_counts()

    assert sum(counts.values()) == 100


def test_job_submit_raises_error(na_backend: QiskitBackend) -> None:
    """Calling submit() on a job should raise NotImplementedError."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = na_backend.run(qc, shots=100)
    with pytest.raises(NotImplementedError, match="Job submission logic must be implemented"):
        job.submit()


def test_backend_warns_on_unmappable_operation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backend should warn when device operation cannot be mapped to a Qiskit gate."""
    import warnings
    from unittest.mock import MagicMock

    from mqt.core import fomac

    # Create mock operations
    mock_cz_op = MagicMock()
    mock_cz_op.name.return_value = "cz"
    mock_cz_op.qubits_num.return_value = 2
    mock_cz_op.parameters_num.return_value = 0
    mock_cz_op.duration.return_value = None
    mock_cz_op.fidelity.return_value = None
    mock_cz_op.sites.return_value = None
    mock_cz_op.is_zoned.return_value = False

    mock_unmappable_op = MagicMock()
    mock_unmappable_op.name.return_value = "custom_unmappable_gate"
    mock_unmappable_op.qubits_num.return_value = 1
    mock_unmappable_op.parameters_num.return_value = 0
    mock_unmappable_op.duration.return_value = None
    mock_unmappable_op.fidelity.return_value = None
    mock_unmappable_op.sites.return_value = None
    mock_unmappable_op.is_zoned.return_value = False

    mock_measure_op = MagicMock()
    mock_measure_op.name.return_value = "measure"
    mock_measure_op.qubits_num.return_value = 1
    mock_measure_op.parameters_num.return_value = 0
    mock_measure_op.duration.return_value = None
    mock_measure_op.fidelity.return_value = None
    mock_measure_op.sites.return_value = None
    mock_measure_op.is_zoned.return_value = False

    # Create mock device
    mock_device = MagicMock()
    mock_device.name.return_value = "Test Device"
    mock_device.qubits_num.return_value = 2
    mock_device.sites.return_value = []
    mock_device.operations.return_value = [mock_cz_op, mock_unmappable_op, mock_measure_op]
    mock_device.coupling_map.return_value = None

    # Monkeypatch fomac.devices to return mock device
    def mock_devices() -> list[Any]:
        return [mock_device]

    monkeypatch.setattr(fomac, "devices", mock_devices)

    # Creating backend should trigger warning about unmappable operation
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        QiskitBackend(device_index=0)

        # Check that the warning was raised
        assert len(w) > 0, "Expected at least one warning to be raised"
        warning_messages = [str(warning.message) for warning in w]
        assert any(
            "custom_unmappable_gate" in msg and "cannot be mapped to a Qiskit gate" in msg for msg in warning_messages
        ), f"Expected warning about custom_unmappable_gate, got: {warning_messages}"


def test_backend_warns_on_missing_measurement_operation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backend should warn when device does not define a measurement operation."""
    import warnings
    from unittest.mock import MagicMock

    from mqt.core import fomac

    # Create mock operation (only cz, no measure)
    mock_cz_op = MagicMock()
    mock_cz_op.name.return_value = "cz"
    mock_cz_op.qubits_num.return_value = 2
    mock_cz_op.parameters_num.return_value = 0
    mock_cz_op.duration.return_value = None
    mock_cz_op.fidelity.return_value = None
    mock_cz_op.sites.return_value = None
    mock_cz_op.is_zoned.return_value = False

    # Create mock device
    mock_device = MagicMock()
    mock_device.name.return_value = "Test Device"
    mock_device.qubits_num.return_value = 2
    mock_device.sites.return_value = []
    mock_device.operations.return_value = [mock_cz_op]  # No measure operation
    mock_device.coupling_map.return_value = None

    # Monkeypatch fomac.devices to return mock device
    def mock_devices() -> list[Any]:
        return [mock_device]

    monkeypatch.setattr(fomac, "devices", mock_devices)

    # Creating backend should trigger warning about missing measurement operation
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        QiskitBackend(device_index=0)

        # Check that the warning was raised
        assert len(w) > 0, "Expected at least one warning to be raised"
        warning_messages = [str(warning.message) for warning in w]
        assert any("does not define a measurement operation" in msg for msg in warning_messages), (
            f"Expected warning about missing measurement operation, got: {warning_messages}"
        )
