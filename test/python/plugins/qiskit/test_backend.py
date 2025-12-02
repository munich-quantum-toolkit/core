# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for QDMIBackend."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, NoReturn, Protocol, cast
from unittest.mock import MagicMock

import pytest
from qiskit import QuantumCircuit, qasm2, qasm3
from qiskit.circuit import Parameter
from qiskit.providers import JobStatus
from qiskit.transpiler.target import InstructionProperties

from mqt.core import fomac
from mqt.core.plugins.qiskit import (
    CircuitValidationError,
    QDMIBackend,
    QDMIProvider,
    TranslationError,
    UnsupportedFormatError,
    UnsupportedOperationError,
)
from mqt.core.plugins.qiskit.exceptions import UnsupportedDeviceError

if TYPE_CHECKING:
    from test.python.plugins.qiskit.conftest import CircuitCase, MockQDMIDevice

    class _FomacDeviceLike(Protocol):  # pragma: no cover - typing helper
        def name(self) -> str: ...

        def version(self) -> str: ...

        def qubits_num(self) -> int: ...

        def operations(self) -> list[object]: ...

        def coupling_map(self) -> object: ...

    SiteSpecificDevice = _FomacDeviceLike
    MisconfiguredDevice = _FomacDeviceLike
    ZonedDevice = _FomacDeviceLike


pytestmark = [
    pytest.mark.filterwarnings("ignore:.*Device operation.*cannot be mapped to a Qiskit gate.*:UserWarning"),
    pytest.mark.filterwarnings("ignore:Device does not define a measurement operation.*:UserWarning"),
]


def test_backend_instantiation(mock_backend: QDMIBackend) -> None:
    """Backend exposes target qubit count."""
    assert mock_backend.target.num_qubits > 0


def test_backend_runs_variety_of_circuits(mock_backend: QDMIBackend, backend_circuit_case: CircuitCase) -> None:
    """Backend executes multiple circuit shapes and shot counts consistently."""
    circuit, shots, expect_measurements = backend_circuit_case
    job = mock_backend.run(circuit, shots=shots)
    result = job.result()
    assert result.success is True

    if expect_measurements:
        counts = result.get_counts()
        assert sum(counts.values()) == shots
        for key in counts:
            assert all(bit in {"0", "1"} for bit in key)


def test_unsupported_operation(mock_backend: QDMIBackend) -> None:
    """Unsupported operation raises UnsupportedOperationError."""
    qc = QuantumCircuit(1, 1)
    qc.x(0)  # 'x' not supported by mock device
    qc.measure(0, 0)
    with pytest.raises(UnsupportedOperationError):
        mock_backend.run(qc)


def test_backend_max_circuits(mock_backend: QDMIBackend) -> None:
    """Backend max_circuits should return None (no limit)."""
    assert mock_backend.max_circuits is None


def test_backend_options(mock_backend: QDMIBackend) -> None:
    """Backend options should be accessible."""
    assert mock_backend.options.shots == 1024


def test_backend_run_with_invalid_shots_type(mock_backend: QDMIBackend) -> None:
    """Backend run should raise CircuitValidationError for invalid shots type."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    with pytest.raises(CircuitValidationError, match="Invalid 'shots' value"):
        mock_backend.run(qc, shots="invalid")


def test_backend_run_with_negative_shots(mock_backend: QDMIBackend) -> None:
    """Backend run should raise CircuitValidationError for negative shots."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    with pytest.raises(CircuitValidationError, match="'shots' must be >= 0"):
        mock_backend.run(qc, shots=-100)


def test_backend_circuit_with_parameters(mock_backend: QDMIBackend) -> None:
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


def test_backend_circuit_with_parameter_expression(mock_backend: QDMIBackend) -> None:
    """Backend should raise CircuitValidationError for unbound parameter expressions."""
    qc = QuantumCircuit(2, 2)
    theta = Parameter("theta")
    phi = Parameter("phi")
    qc.ry(theta + phi, 0)  # Parameter expression
    qc.measure([0, 1], [0, 1])

    with pytest.raises(CircuitValidationError, match="Circuit contains unbound parameters"):
        mock_backend.run(qc)


def test_backend_named_circuit_results_queryable_by_name(mock_backend: QDMIBackend) -> None:
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


def test_backend_unnamed_circuit_results_queryable_by_generated_name(mock_backend: QDMIBackend) -> None:
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


def test_job_status(mock_backend: QDMIBackend) -> None:
    """Job should be in DONE status after backend.run() completes."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = mock_backend.run(qc, shots=100)
    assert job.status() == JobStatus.DONE


def test_job_result_success_and_shots(mock_backend: QDMIBackend) -> None:
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


def test_job_get_counts_default(mock_backend: QDMIBackend) -> None:
    """result().get_counts() without arguments should return counts for first circuit."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = mock_backend.run(qc, shots=100)
    counts = job.result().get_counts()

    assert sum(counts.values()) == 100


def test_job_submit_raises_error(mock_backend: QDMIBackend) -> None:
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


@pytest.mark.parametrize(
    (
        "gate_name",
        "test_cases",
    ),
    [
        (
            "h",
            [
                ((0,), 100.0, 0.99),
                ((1,), 110.0, 0.98),
                ((2,), 120.0, 0.97),
            ],
        ),
        (
            "cz",
            [
                ((0, 1), 205.0, 0.975),
                ((1, 2), 220.0, 0.965),
            ],
        ),
    ],
)
def test_backend_with_site_specific_properties(
    site_specific_device: SiteSpecificDevice, gate_name: str, test_cases: list[tuple[tuple[int, ...], float, float]]
) -> None:
    """Backend should honor site-specific duration and fidelity metadata."""
    backend = QDMIBackend(device=cast("fomac.Device", site_specific_device))

    for qarg, expected_duration, expected_fidelity in test_cases:
        props = backend.target[gate_name][qarg]
        assert props is not None
        assert props.duration == expected_duration
        expected_error = 1.0 - expected_fidelity
        assert abs(props.error - expected_error) < 1e-10


def test_backend_qasm_conversion_no_supported_formats(
    monkeypatch: pytest.MonkeyPatch, mock_qdmi_device_factory: type[MockQDMIDevice]
) -> None:
    """Backend should raise UnsupportedFormatError when device has no supported program formats."""
    # Create mock device with no supported formats
    device = mock_qdmi_device_factory(
        name="No Format Device",
        num_qubits=2,
        operations=["cz", "measure"],
    )
    monkeypatch.setattr(device, "supported_program_formats", list)
    monkeypatch.setattr(fomac, "devices", lambda: [device])

    # Create backend and try to run a circuit
    provider = QDMIProvider()
    backend = provider.get_backend("No Format Device")

    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    with pytest.raises(UnsupportedFormatError, match="No supported program formats found"):
        backend.run(qc, shots=100)


@pytest.mark.parametrize(
    ("qasm_module_name", "program_format"),
    [
        ("qasm3", fomac.ProgramFormat.QASM3),
        ("qasm2", fomac.ProgramFormat.QASM2),
    ],
)
def test_backend_qasm_conversion_failure(
    monkeypatch: pytest.MonkeyPatch,
    mock_qdmi_device_factory: type[MockQDMIDevice],
    qasm_module_name: str,
    program_format: fomac.ProgramFormat,
) -> None:
    """Backend should raise TranslationError when QASM conversion fails."""
    qasm_module = qasm3 if qasm_module_name == "qasm3" else qasm2

    # Create a device that only supports the specified format
    device = mock_qdmi_device_factory(
        name=f"{qasm_module_name.upper()} Only Device",
        num_qubits=2,
        operations=["cz", "measure"],
    )
    monkeypatch.setattr(device, "supported_program_formats", lambda: [program_format])
    monkeypatch.setattr(fomac, "devices", lambda: [device])

    # Create backend
    provider = QDMIProvider()
    backend = provider.get_backend(f"{qasm_module_name.upper()} Only Device")

    # Monkeypatch qasm dumps to raise an exception
    def failing_dumps(circuit: object) -> NoReturn:  # noqa: ARG001
        msg = f"Simulated {qasm_module_name.upper()} conversion failure"
        raise ValueError(msg)

    monkeypatch.setattr(qasm_module, "dumps", failing_dumps)

    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    with pytest.raises(TranslationError, match=f"Failed to convert circuit to {qasm_module_name.upper()}"):
        backend.run(qc, shots=100)


def test_backend_unsupported_format_error(
    monkeypatch: pytest.MonkeyPatch, mock_qdmi_device_factory: type[MockQDMIDevice]
) -> None:
    """Backend should raise UnsupportedFormatError when device supports unknown format."""
    # Create a device that supports a format other than QASM2/QASM3 (QPY)
    device = mock_qdmi_device_factory(
        name="Unknown Format Device",
        num_qubits=2,
        operations=["cz", "measure"],
    )
    monkeypatch.setattr(device, "supported_program_formats", lambda: [fomac.ProgramFormat.QPY])
    monkeypatch.setattr(fomac, "devices", lambda: [device])

    # Create backend
    provider = QDMIProvider()
    backend = provider.get_backend("Unknown Format Device")

    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    with pytest.raises(
        UnsupportedFormatError, match="No conversion from Qiskit to any of the supported program formats"
    ):
        backend.run(qc, shots=100)


def test_backend_supports_cz_gate(mock_backend: QDMIBackend) -> None:
    """Backend executes CZ gate circuits and returns counts."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = mock_backend.run(qc, shots=100)
    counts = job.result().get_counts()
    assert sum(counts.values()) == 100


def test_map_operation_returns_none_for_unknown(mock_backend: QDMIBackend) -> None:
    """Unknown FoMaC operations cannot be mapped to Qiskit gates."""
    assert mock_backend._map_operation_to_gate("unknown_gate") is None  # noqa: SLF001
    assert mock_backend._map_operation_to_gate("custom_op") is None  # noqa: SLF001
    assert mock_backend._map_operation_to_gate("") is None  # noqa: SLF001


@pytest.mark.parametrize(
    ("gate_name", "expected_qargs"),
    [
        ("h", [(0,), (1,), (2,)]),
        ("cz", [(0, 1), (1, 2)]),
    ],
)
def test_backend_operation_properties(
    site_specific_device: SiteSpecificDevice, gate_name: str, expected_qargs: list[tuple[int, ...]]
) -> None:
    """Target exposes per-qarg instruction properties for site-aware operations."""
    backend = QDMIBackend(device=cast("fomac.Device", site_specific_device))
    props_map = backend.target[gate_name]
    assert sorted(props_map.keys()) == sorted(expected_qargs)

    for qarg in expected_qargs:
        props = props_map[qarg]
        assert isinstance(props, InstructionProperties)
        assert props.duration is not None
        assert props.error is not None


def test_misconfigured_device_coupling_map_without_operation_sites(
    misconfigured_coupling_device: MisconfiguredDevice,
) -> None:
    """Devices with coupling maps must provide site pairs for two-qubit operations."""
    backend = QDMIBackend(device=cast("fomac.Device", misconfigured_coupling_device))

    mock_op = MagicMock()
    mock_op.name.return_value = "custom_2q"
    mock_op.qubits_num.return_value = 2
    mock_op.site_pairs.return_value = None
    mock_op.is_zoned.return_value = False

    with pytest.raises(UnsupportedOperationError, match="misconfigured device"):
        backend._get_operation_qargs(mock_op)  # noqa: SLF001


def test_zoned_operation_rejected_at_backend_init(zoned_operation_device: ZonedDevice) -> None:
    """Backend rejects devices exposing zoned operations."""
    with pytest.raises(UnsupportedDeviceError, match="cannot be represented in Qiskit's Target model"):
        QDMIBackend(device=cast("fomac.Device", zoned_operation_device))
