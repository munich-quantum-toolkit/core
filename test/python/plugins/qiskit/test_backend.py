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
from typing import TYPE_CHECKING, Any, NoReturn
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
    from test.python.plugins.qiskit.conftest import MockQDMIDevice


pytestmark = [
    pytest.mark.filterwarnings("ignore:.*Device operation.*cannot be mapped to a Qiskit gate.*:UserWarning"),
    pytest.mark.filterwarnings("ignore:Device does not define a measurement operation.*:UserWarning"),
]


def test_backend_instantiation(mock_backend: QDMIBackend) -> None:
    """Backend exposes target qubit count."""
    assert mock_backend.target.num_qubits > 0


def test_single_circuit_run_counts(mock_backend: QDMIBackend) -> None:
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


def test_unsupported_operation(mock_backend: QDMIBackend) -> None:
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


def test_backend_max_circuits(mock_backend: QDMIBackend) -> None:
    """Backend max_circuits should return None (no limit)."""
    assert mock_backend.max_circuits is None


def test_backend_options(mock_backend: QDMIBackend) -> None:
    """Backend options should be accessible."""
    assert mock_backend.options.shots == 1024


def test_backend_run_with_shots_option(mock_backend: QDMIBackend) -> None:
    """Backend run should accept shots option."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure([0, 1], [0, 1])

    job = mock_backend.run(qc, shots=500)
    result = job.result()
    assert sum(result.get_counts().values()) == 500


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


def test_backend_circuit_with_barrier(mock_backend: QDMIBackend) -> None:
    """Backend should handle circuits with barriers."""
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.barrier()
    qc.measure([0, 1], [0, 1])

    job = mock_backend.run(qc, shots=100)
    result = job.result()
    assert result.success


def test_backend_circuit_with_single_qubit(mock_backend: QDMIBackend) -> None:
    """Backend should handle single-qubit circuits."""
    qc = QuantumCircuit(1, 1)
    qc.ry(1.5708, 0)
    qc.measure(0, 0)

    job = mock_backend.run(qc, shots=100)
    result = job.result()
    assert result.success


def test_backend_circuit_with_no_measurements(mock_backend: QDMIBackend) -> None:
    """Backend should handle circuits without measurements."""
    qc = QuantumCircuit(2)
    qc.cz(0, 1)

    job = mock_backend.run(qc, shots=100)
    result = job.result()
    assert result.success


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
    ("num_qubits_param", "gate_name", "qubit_count", "test_cases"),
    [
        # Single-qubit gate with site-specific properties
        (
            3,
            "h",
            1,
            [
                (0, 100.0, 0.99),  # qubit 0: duration=100, fidelity=0.99
                (1, 110.0, 0.98),  # qubit 1: duration=110, fidelity=0.98
                (2, 120.0, 0.97),  # qubit 2: duration=120, fidelity=0.97
            ],
        ),
        # Two-qubit gate with site-pair-specific properties
        (
            3,
            "cz",
            2,
            [
                ((0, 1), 205.0, 0.975),  # pair (0,1): duration=200+0*10+1*5=205, fidelity=0.98-1*0.005=0.975
                ((1, 2), 220.0, 0.965),  # pair (1,2): duration=200+1*10+2*5=220, fidelity=0.98-3*0.005=0.965
            ],
        ),
    ],
)
def test_backend_with_site_specific_properties(
    monkeypatch: pytest.MonkeyPatch,
    mock_qdmi_device_factory: type[MockQDMIDevice],
    num_qubits_param: int,
    gate_name: str,
    qubit_count: int,
    test_cases: list[tuple[tuple[int, int], float, float]],
) -> None:
    """Backend should handle operations with site-specific duration and fidelity.

    Tests both single-qubit and two-qubit operations with parameterization.
    """

    class MockSite:
        def __init__(self, idx: int) -> None:
            self._idx = idx

        def index(self) -> int:
            return self._idx

        @staticmethod
        def is_zone() -> bool:
            return False

    class MockOperationWithProperties:
        def __init__(self, name: str, num_qubits: int, site_data: list[Any]) -> None:
            self._name = name
            self._num_qubits = num_qubits
            self._site_data = site_data

        def name(self) -> str:
            return self._name

        def qubits_num(self) -> int:
            return self._num_qubits

        @staticmethod
        def parameters_num() -> int:
            return 0

        def sites(self) -> list[MockSite] | None:
            return self._site_data if self._num_qubits == 1 else None

        def site_pairs(self) -> list[tuple[MockSite, MockSite]] | None:
            return self._site_data if self._num_qubits == 2 else None

        def duration(self, sites: list[MockSite] | None = None) -> float | None:
            if not sites:
                return None
            if self._num_qubits == 1:
                return 100.0 + sites[0].index() * 10.0
            # Two-qubit
            return 200.0 + sites[0].index() * 10.0 + sites[1].index() * 5.0

        def fidelity(self, sites: list[MockSite] | None = None) -> float | None:
            if not sites:
                return None
            if self._num_qubits == 1:
                return 0.99 - sites[0].index() * 0.01
            # Two-qubit
            return 0.98 - (sites[0].index() + sites[1].index()) * 0.005

        @staticmethod
        def is_zoned() -> bool:
            return False

    # Prepare site data based on qubit count
    sites = [MockSite(i) for i in range(num_qubits_param)]
    if qubit_count == 1:
        site_data: list[Any] = list(sites)
    else:
        site_pairs: list[tuple[MockSite, MockSite]] = [
            (sites[0], sites[1]),
            (sites[1], sites[2]),
        ]
        site_data = list(site_pairs)

    class CustomMockDevice:
        @staticmethod
        def name() -> str:
            return "Site-Specific Device"

        @staticmethod
        def version() -> str:
            return "1.0.0"

        @staticmethod
        def qubits_num() -> int:
            return num_qubits_param

        @staticmethod
        def operations() -> list[Any]:
            return [
                MockOperationWithProperties(gate_name, qubit_count, site_data),
                mock_qdmi_device_factory.MockOperation("measure"),
            ]

        @staticmethod
        def coupling_map() -> None:
            return None

        @staticmethod
        def supported_program_formats() -> list[fomac.ProgramFormat]:
            return [fomac.ProgramFormat.QASM3]

    monkeypatch.setattr(fomac, "devices", lambda: [CustomMockDevice()])

    # Create backend and verify properties
    provider = QDMIProvider()
    backend = provider.get_backend("Site-Specific Device")
    assert backend.target.num_qubits == num_qubits_param

    # Verify site-specific properties for each test case
    for qarg, expected_duration, expected_fidelity in test_cases:
        props = backend.target[gate_name][qarg if qubit_count == 2 else (qarg,)]
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


def test_backend_operation_properties(mock_backend: QDMIBackend) -> None:
    """Backend registers expected operations including property maps."""
    expected_ops = {"h", "ry", "rz", "cz", "measure"}
    assert expected_ops.issubset(set(mock_backend.target.operation_names))

    for op_name in expected_ops:
        props_map = mock_backend.target[op_name]
        assert isinstance(props_map, dict)
        for qargs, props in props_map.items():
            assert qargs is None or isinstance(qargs, tuple)
            assert props is None or isinstance(props, InstructionProperties)


def test_get_operation_qargs_single_qubit(mock_backend: QDMIBackend) -> None:
    """Single-qubit ops without explicit sites are globally available."""
    mock_op = MagicMock()
    mock_op.name.return_value = "test_op"
    mock_op.qubits_num.return_value = 1
    mock_op.parameters_num.return_value = 0
    mock_op.duration.return_value = None
    mock_op.fidelity.return_value = None
    mock_op.sites.return_value = None
    mock_op.is_zoned.return_value = False

    qargs = mock_backend._get_operation_qargs(mock_op)  # noqa: SLF001

    assert qargs == [None]


def test_get_operation_qargs_two_qubit_without_coupling_map(mock_backend: QDMIBackend) -> None:
    """Two-qubit ops without site pairs or coupling map are global."""
    assert mock_backend._device.coupling_map() is None  # noqa: SLF001

    mock_op = MagicMock()
    mock_op.name.return_value = "test_2q"
    mock_op.qubits_num.return_value = 2
    mock_op.parameters_num.return_value = 0
    mock_op.duration.return_value = None
    mock_op.fidelity.return_value = None
    mock_op.site_pairs.return_value = None
    mock_op.is_zoned.return_value = False

    qargs = mock_backend._get_operation_qargs(mock_op)  # noqa: SLF001

    assert qargs == [None]


def test_get_operation_qargs_with_specific_sites(mock_backend: QDMIBackend) -> None:
    """Target reflects backend operations including qargs."""
    assert mock_backend.target.operation_names

    for op_name in mock_backend.target.operation_names:
        qargs = mock_backend.target.qargs_for_operation_name(op_name)
        if qargs is None:
            continue
        for qarg in qargs:
            assert isinstance(qarg, tuple)
            assert all(isinstance(idx, int) for idx in qarg)


def test_get_operation_qargs_two_qubit_operation_with_subset_of_coupling_map(mock_backend: QDMIBackend) -> None:
    """Two-qubit ops respect specific qargs when defined."""
    two_qubit_ops = [
        op_name
        for op_name in mock_backend.target.operation_names
        if mock_backend.target.operation_from_name(op_name).num_qubits == 2
    ]
    assert two_qubit_ops

    for op_name in two_qubit_ops:
        qargs = mock_backend.target.qargs_for_operation_name(op_name)
        if not qargs:
            continue
        for qarg in qargs:
            assert len(qarg) == 2
            assert all(isinstance(idx, int) for idx in qarg)


def test_misconfigured_device_coupling_map_without_operation_sites() -> None:
    """Devices with coupling map need site pairs for two-qubit ops."""

    class MockDeviceWithCouplingMap:
        class MockSite:
            def __init__(self, idx: int) -> None:
                self._idx = idx

            def index(self) -> int:
                return self._idx

            @staticmethod
            def is_zone() -> bool:
                return False

        def __init__(self) -> None:
            self._sites = [self.MockSite(i) for i in range(5)]
            self._coupling_map = [(self._sites[0], self._sites[1]), (self._sites[1], self._sites[2])]

        @staticmethod
        def name() -> str:
            return "Mock Device With Coupling"

        @staticmethod
        def version() -> str:
            return "1.0.0"

        @staticmethod
        def qubits_num() -> int:
            return 5

        def regular_sites(self) -> list[MockSite]:
            return self._sites

        @staticmethod
        def operations() -> list[object]:
            return []

        def coupling_map(self) -> list[tuple[MockSite, MockSite]]:
            return self._coupling_map

    device = MockDeviceWithCouplingMap()
    backend = QDMIBackend(device=device)  # type: ignore[arg-type]

    mock_op = MagicMock()
    mock_op.name.return_value = "custom_2q"
    mock_op.qubits_num.return_value = 2
    mock_op.site_pairs.return_value = None
    mock_op.is_zoned.return_value = False

    with pytest.raises(UnsupportedOperationError, match="misconfigured device"):
        backend._get_operation_qargs(mock_op)  # noqa: SLF001


def test_zoned_operation_rejected_at_backend_init() -> None:
    """Backend rejects devices exposing zoned operations."""

    class MockZonedDevice:
        class MockZonedOp:
            @staticmethod
            def name() -> str:
                return "zoned_op"

            @staticmethod
            def is_zoned() -> bool:
                return True

        def __init__(self) -> None:
            self._ops = [self.MockZonedOp()]

        @staticmethod
        def name() -> str:
            return "Mock Zoned Device"

        @staticmethod
        def version() -> str:
            return "1.0.0"

        def operations(self) -> list[MockZonedOp]:
            return self._ops

        @staticmethod
        def qubits_num() -> int:
            return 5

        @staticmethod
        def regular_sites() -> list[object]:
            return []

    device = MockZonedDevice()

    with pytest.raises(UnsupportedDeviceError, match="cannot be represented in Qiskit's Target model"):
        QDMIBackend(device=device)  # type: ignore[arg-type]
