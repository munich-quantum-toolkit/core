# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for QiskitBackend."""

from __future__ import annotations

import importlib.util

import pytest

from mqt.core.qdmi.qiskit import (
    TranslationError,
    UnsupportedOperationError,
)

_qiskit_present = importlib.util.find_spec("qiskit") is not None

pytestmark = pytest.mark.skipif(not _qiskit_present, reason="qiskit not installed")

if _qiskit_present:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    from mqt.core import fomac
    from mqt.core.qdmi.qiskit import QiskitBackend


@pytest.fixture
def na_backend() -> QiskitBackend:
    """Fixture providing a QiskitBackend configured with the NA device.

    Returns:
        QiskitBackend instance configured with the MQT NA Default QDMI Device.

    Raises:
        RuntimeError: If the MQT NA Default QDMI Device is not found.

    Note:
        This fixture is used for tests that rely on specific NA device characteristics.
        In the future, these tests should be generalized or parameterized across device types.
    """
    devices_list = list(fomac.devices())
    for idx, device in enumerate(devices_list):
        if device.name() == "MQT NA Default QDMI Device":
            return QiskitBackend(device_index=idx)
    msg = "MQT NA Default QDMI Device not found"
    raise RuntimeError(msg)


def test_backend_instantiation(na_backend: QiskitBackend) -> None:
    """Backend exposes capabilities hash and target qubit count."""
    assert na_backend.capabilities_hash
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
    counts = job.get_counts()

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
    """Backend run should raise TranslationError for invalid shots type."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    with pytest.raises(TranslationError, match="Invalid 'shots' value"):
        na_backend.run(qc, shots="invalid")


def test_backend_run_with_negative_shots(na_backend: QiskitBackend) -> None:
    """Backend run should raise TranslationError for negative shots."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    with pytest.raises(TranslationError, match="'shots' must be >= 0"):
        na_backend.run(qc, shots=-100)


def test_backend_run_multiple_circuits(na_backend: QiskitBackend) -> None:
    """Backend run should handle multiple circuits."""
    qc1 = QuantumCircuit(2, 2)
    qc1.cz(0, 1)
    qc1.measure([0, 1], [0, 1])

    qc2 = QuantumCircuit(2, 2)
    qc2.cz(0, 1)
    qc2.measure([0, 1], [0, 1])

    job = na_backend.run([qc1, qc2], shots=100)
    result = job.result()

    assert len(result.results) == 2


def test_backend_circuit_with_unbound_parameter() -> None:
    """Backend should raise TranslationError for unbound parameters."""
    backend = QiskitBackend()
    qc = QuantumCircuit(2, 2)
    theta = Parameter("theta")
    qc.ry(theta, 0)
    qc.measure([0, 1], [0, 1])

    with pytest.raises(TranslationError, match="Circuit contains unbound parameters"):
        backend.run(qc)


def test_backend_circuit_with_bound_parameter() -> None:
    """Backend should handle circuits with bound parameters."""
    backend = QiskitBackend()
    qc = QuantumCircuit(2, 2)
    theta = Parameter("theta")
    qc.ry(theta, 0)
    qc.measure([0, 1], [0, 1])

    # Bind parameter
    qc_bound = qc.assign_parameters({theta: 1.5708})

    job = backend.run(qc_bound, shots=100)
    result = job.result()
    assert result.success


def test_backend_circuit_with_parameter_expression() -> None:
    """Backend should raise TranslationError for unbound parameter expressions."""
    backend = QiskitBackend()
    qc = QuantumCircuit(2, 2)
    theta = Parameter("theta")
    phi = Parameter("phi")
    qc.ry(theta + phi, 0)  # Parameter expression
    qc.measure([0, 1], [0, 1])

    with pytest.raises(TranslationError, match="Circuit contains unbound parameters"):
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


def test_backend_target_operation_names(na_backend: QiskitBackend) -> None:
    """Backend target should expose operation names."""
    op_names = na_backend.target.operation_names

    # Check for measure if device provides it
    if "measure" in na_backend._capabilities.operations:  # noqa: SLF001
        assert "measure" in op_names

    # NA device should have at least some of these operations
    assert "cz" in op_names or "ry" in op_names or "rz" in op_names


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
    """Backend result metadata should include circuit name."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = na_backend.run(qc, shots=100)
    result = job.result()
    assert "circuit_name" in result.results[0].metadata


def test_backend_result_metadata_includes_capabilities_hash(na_backend: QiskitBackend) -> None:
    """Backend result metadata should include capabilities hash."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = na_backend.run(qc, shots=100)
    result = job.result()
    assert "capabilities_hash" in result.results[0].metadata
