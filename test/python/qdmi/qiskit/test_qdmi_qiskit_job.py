# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for QiskitJob."""

from __future__ import annotations

import importlib.util

import pytest

_qiskit_present = importlib.util.find_spec("qiskit") is not None

pytestmark = pytest.mark.skipif(not _qiskit_present, reason="qiskit not installed")

if _qiskit_present:
    from qiskit import QuantumCircuit
    from qiskit.providers import JobStatus

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


def test_job_status(na_backend: QiskitBackend) -> None:
    """Job should report DONE status immediately."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = na_backend.run(qc, shots=100)
    assert job.status() == JobStatus.DONE


def test_job_result(na_backend: QiskitBackend) -> None:
    """Job result should contain experiment results for each circuit."""
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
    """Job result should accept timeout parameter (even though it's unused)."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = na_backend.run(qc, shots=50)
    result = job.result(timeout=10)

    assert result.success is True
    assert result.results[0].shots == 50


def test_job_get_counts_default(na_backend: QiskitBackend) -> None:
    """get_counts() without arguments should return counts for first circuit."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = na_backend.run(qc, shots=100)
    counts = job.get_counts()

    assert sum(counts.values()) == 100


def test_job_get_counts_by_index(na_backend: QiskitBackend) -> None:
    """get_counts(idx) should return counts for the specified circuit index."""
    qc1 = QuantumCircuit(2, 2)
    qc1.cz(0, 1)
    qc1.measure([0, 1], [0, 1])

    qc2 = QuantumCircuit(2, 2)
    qc2.ry(1.5708, 0)
    qc2.measure([0, 1], [0, 1])

    job = na_backend.run([qc1, qc2], shots=100)
    counts0 = job.get_counts(0)
    counts1 = job.get_counts(1)

    assert sum(counts0.values()) == 100
    assert sum(counts1.values()) == 100


def test_job_get_counts_by_circuit_object(na_backend: QiskitBackend) -> None:
    """get_counts(circuit) should return counts for that specific circuit."""
    qc1 = QuantumCircuit(2, 2)
    qc1.cz(0, 1)
    qc1.measure([0, 1], [0, 1])

    qc2 = QuantumCircuit(2, 2)
    qc2.ry(1.5708, 0)
    qc2.measure([0, 1], [0, 1])

    job = na_backend.run([qc1, qc2], shots=100)
    counts = job.get_counts(qc2)

    assert sum(counts.values()) == 100


def test_job_get_counts_circuit_not_found(na_backend: QiskitBackend) -> None:
    """get_counts() should raise ValueError if circuit is not in the job."""
    qc1 = QuantumCircuit(2, 2)
    qc1.cz(0, 1)
    qc1.measure([0, 1], [0, 1])

    qc2 = QuantumCircuit(2, 2)  # Different circuit not in job
    qc2.ry(1.5708, 0)
    qc2.measure([0, 1], [0, 1])

    job = na_backend.run(qc1, shots=100)

    with pytest.raises(ValueError, match="Circuit not found in job"):
        job.get_counts(qc2)


def test_job_submit_noop(na_backend: QiskitBackend) -> None:
    """submit() should be a no-op since execution is synchronous."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = na_backend.run(qc, shots=100)
    job.submit()  # Should not raise


def test_job_multiple_circuits(na_backend: QiskitBackend) -> None:
    """Job with multiple circuits should have results for each."""
    circuits = []
    for _i in range(3):
        qc = QuantumCircuit(2, 2)
        qc.cz(0, 1)
        qc.measure([0, 1], [0, 1])
        circuits.append(qc)

    job = na_backend.run(circuits, shots=50)
    result = job.result()

    assert len(result.results) == 3
    for exp_result in result.results:
        assert exp_result.shots == 50


def test_job_result_metadata(na_backend: QiskitBackend) -> None:
    """Job result should include metadata from backend execution."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = na_backend.run(qc, shots=100)
    result = job.result()

    # Check that metadata is included
    assert result.results[0].metadata is not None
    assert "circuit_name" in result.results[0].metadata
    assert "capabilities_hash" in result.results[0].metadata
