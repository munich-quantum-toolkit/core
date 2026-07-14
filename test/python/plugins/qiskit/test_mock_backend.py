# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Backend tests using a mock QDMI device implementation."""

from __future__ import annotations

import json
import re
import secrets
import string
import warnings
from typing import TYPE_CHECKING, NoReturn

import numpy as np
import pytest
from qiskit import qasm2, qasm3
from qiskit.circuit import Clbit, Parameter, QuantumCircuit

from mqt.core import fomac
from mqt.core.plugins.qiskit import (
    MoveGate,
    QDMIBackend,
    QDMIProvider,
    TranslationError,
    UnsupportedFormatError,
    UnsupportedOperationError,
    qiskit_to_iqm_json,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class MockQDMIDevice:
    """Mock QDMI device for testing with configurable properties and job execution.

    This class implements the FoMaC device interface for testing purposes,
    providing configurable device properties and mock job execution.
    """

    class MockSite:
        """Mock device site."""

        def __init__(self, idx: int) -> None:
            """Initialize mock site with index."""
            self._index = idx

        def index(self) -> int:
            """Return site index."""
            return self._index

        def name(self) -> str:
            """Return site name."""
            return f"site_{self._index}"

        @staticmethod
        def is_zone() -> bool:
            """Return whether site is a zone (always False for mock sites)."""
            return False

    class MockOperation:
        """Mock device operation."""

        def __init__(
            self,
            name: str,
            *,
            custom_duration: Callable[[list[MockQDMIDevice.MockSite]], float | None] | None = None,
            custom_fidelity: Callable[[list[MockQDMIDevice.MockSite]], float | None] | None = None,
            custom_sites: list[MockQDMIDevice.MockSite] | None = None,
            custom_site_pairs: list[tuple[MockQDMIDevice.MockSite, MockQDMIDevice.MockSite]] | None = None,
            zoned: bool = False,
        ) -> None:
            """Initialize mock operation with name and optional custom behavior."""
            self._name = name
            self._duration = custom_duration
            self._fidelity = custom_fidelity
            self._custom_sites = custom_sites
            self._custom_site_pairs = custom_site_pairs
            self._zoned = zoned
            # Determine qubit count and parameters based on operation name
            if name in {"h", "x", "y", "z", "s", "t", "measure", "sx", "id", "i"}:
                self._qubits = 1
                self._params = 0
            elif name in {"ry", "rz", "rx", "p", "phase"}:
                self._qubits = 1
                self._params = 1
            elif name in {"cz", "cx", "cnot", "cy", "ch", "swap", "iswap", "move"}:
                self._qubits = 2
                self._params = 0
            elif name in {"rxx", "ryy", "rzz", "rzx"}:
                self._qubits = 2
                self._params = 1
            else:
                self._qubits = 1
                self._params = 0

        def name(self) -> str:
            """Return operation name."""
            return self._name

        def qubits_num(self) -> int:
            """Return number of qubits for operation."""
            return self._qubits

        def parameters_num(self) -> int:
            """Return number of parameters for operation."""
            return self._params

        def duration(self, sites: list[MockQDMIDevice.MockSite] | None = None) -> float | None:
            """Return custom duration if defined for the provided sites."""
            if self._duration and sites:
                return self._duration(sites)
            return None

        def fidelity(self, sites: list[MockQDMIDevice.MockSite] | None = None) -> float | None:
            """Return custom fidelity if defined for the provided sites."""
            if self._fidelity and sites:
                return self._fidelity(sites)
            return None

        def sites(self) -> list[MockQDMIDevice.MockSite] | None:
            """Return the list of allowed single-qubit sites, if any."""
            return self._custom_sites

        def site_pairs(self) -> list[tuple[MockQDMIDevice.MockSite, MockQDMIDevice.MockSite]] | None:
            """Return the list of allowed two-qubit site pairs, if any."""
            return self._custom_site_pairs

        def is_zoned(self) -> bool:
            """Return True if the operation is marked as zoned."""
            return self._zoned

    class MockJob:
        """Mock FoMaC job with simulated results."""

        def __init__(self, num_clbits: int, shots: int) -> None:
            """Initialize mock job with number of classical bits and shots."""
            self._num_clbits = num_clbits
            self._shots = shots
            alphabet = string.ascii_lowercase + string.digits
            self._id = "mock-job-" + "".join(secrets.choice(alphabet) for _ in range(8))
            self._status = fomac.Job.Status.DONE
            self._counts: dict[str, int] | None = None

        @property
        def id(self) -> str:
            """The job ID."""
            return self._id

        @property
        def num_shots(self) -> int:
            """The number of shots."""
            return self._shots

        def check(self) -> fomac.Job.Status:
            """Return job status."""
            return self._status

        def wait(self) -> None:
            """Wait for job completion (no-op for mock)."""

        def get_counts(self) -> dict[str, int]:
            """Get measurement counts with uniform random distribution.

            Returns:
                Dictionary mapping measurement outcomes to counts.
            """
            if self._num_clbits == 0:
                return {"": self._shots}

            if self._counts is None:
                # Generate random counts with uniform distribution
                num_outcomes = 2**self._num_clbits
                outcomes = [format(i, f"0{self._num_clbits}b") for i in range(num_outcomes)]

                # Distribute shots randomly among outcomes
                counts_list = [0] * num_outcomes
                for _ in range(self._shots):
                    counts_list[secrets.randbelow(num_outcomes)] += 1

                # Create dictionary, including only non-zero counts
                self._counts = {
                    outcome: count for outcome, count in zip(outcomes, counts_list, strict=True) if count > 0
                }

            return self._counts

        def cancel(self) -> None:
            """Cancel job (no-op for mock)."""

    def __init__(
        self,
        name: str = "Mock QDMI Device",
        version: str = "1.0.0",
        num_qubits: int = 5,
        operations: Sequence[str] | None = None,
        coupling_map: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        """Initialize a mock QDMI device.

        Args:
            name: Device name.
            version: Device version.
            num_qubits: Number of qubits.
            operations: List of operation names. Defaults to common gates.
            coupling_map: Coupling map as list of (control, target) pairs. None means all-to-all.
        """
        self._name = name
        self._version = version
        self._num_qubits = num_qubits
        self._sites = [self.MockSite(i) for i in range(num_qubits)]

        if operations is None:
            operations = ["h", "cz", "ry", "rz", "measure"]
        self._operations = [self.MockOperation(op) for op in operations]

        if coupling_map is not None:
            self._coupling_map: list[tuple[MockQDMIDevice.MockSite, MockQDMIDevice.MockSite]] | None = [
                (self._sites[ctrl], self._sites[tgt]) for ctrl, tgt in coupling_map
            ]
        else:
            self._coupling_map = None

    def name(self) -> str:
        """Return device name."""
        return self._name

    def version(self) -> str:
        """Return device version."""
        return self._version

    def qubits_num(self) -> int:
        """Return number of qubits."""
        return self._num_qubits

    def sites(self) -> list[MockSite]:
        """Return list of device sites."""
        return self._sites

    def regular_sites(self) -> list[MockSite]:
        """Return list of regular sites (qubits)."""
        return self._sites

    @staticmethod
    def zones() -> list[MockSite]:
        """Return list of zones."""
        return []

    def operations(self) -> list[MockOperation]:
        """Return list of device operations."""
        return self._operations

    def coupling_map(self) -> list[tuple[MockSite, MockSite]] | None:
        """Return device coupling map or None if all-to-all."""
        return self._coupling_map

    @staticmethod
    def supported_program_formats() -> list[fomac.ProgramFormat]:
        """Return list of supported program formats."""
        return [fomac.ProgramFormat.QASM2, fomac.ProgramFormat.QASM3]

    def submit_job(self, program: str, program_format: fomac.ProgramFormat, num_shots: int) -> MockJob:  # noqa: ARG002
        """Submit a mock job to the device.

        Args:
            program: The program string to parse for classical bit count.
            program_format: The program format (unused in mock).
            num_shots: Number of shots to simulate.

        Returns:
            A mock job with simulated results.
        """
        # Parse the number of classical bits from a QASM program.

        # Look for "creg <name>[<size>];" pattern in QASM2
        matches_qasm2 = re.findall(r"creg\s+\w+\[(\d+)]", program)
        count_qasm2 = sum(int(m) for m in matches_qasm2)

        # Look for "bit[<size>] <name>;" pattern in QASM3
        matches_qasm3_arrays = re.findall(r"\bbit\[(\d+)]\s+\w+\s*;", program)
        count_qasm3_arrays = sum(int(m) for m in matches_qasm3_arrays)

        # Look for scalar-bit declarations in QASM3, including declarations with
        # optional initializer expressions.
        matches_qasm3_scalars = re.findall(r"\bbit(?!\s*\[)\s+\w+\s*(?:=\s*[^;]+)?;", program)
        count_qasm3_scalars = len(matches_qasm3_scalars)

        num_clbits = count_qasm2 + count_qasm3_arrays + count_qasm3_scalars
        return self.MockJob(num_clbits=num_clbits, shots=num_shots)


@pytest.fixture
def mock_qdmi_device_factory() -> type[MockQDMIDevice]:
    """Factory fixture for creating custom MockQDMIDevice instances.

    Returns:
        The MockQDMIDevice class that can be called to create instances.

    Note:
        Use this fixture when you need to create custom mock device instances
        with specific configurations (operations, coupling maps, etc.) for testing.

    Example:
        def test_custom_device(mock_qdmi_device_factory):
            device = mock_qdmi_device_factory(
                name="Custom Device",
                num_qubits=2,
                operations=["h", "cx"]
            )
    """
    return MockQDMIDevice


def _patch_session_devices(monkeypatch: pytest.MonkeyPatch, devices: list[MockQDMIDevice]) -> None:
    """Helper to monkeypatch fomac.Session.get_devices to return the given devices list."""

    def _mock_get_devices(_self: object) -> list[MockQDMIDevice]:
        return devices

    monkeypatch.setattr(fomac.Session, "get_devices", _mock_get_devices)


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

    # Use helper to patch Session.get_devices
    _patch_session_devices(monkeypatch, [mock_device])

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


def test_backend_exposes_move_operation(
    monkeypatch: pytest.MonkeyPatch, mock_qdmi_device_factory: type[MockQDMIDevice]
) -> None:
    """Backend target should expose MOVE when the device reports it."""
    mock_device = mock_qdmi_device_factory(
        name="Test Device",
        num_qubits=2,
        operations=["move", "measure"],
    )
    _patch_session_devices(monkeypatch, [mock_device])

    provider = QDMIProvider()
    backend = provider.get_backend("Test Device")

    assert "move" in backend.target.operation_names


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

    # Use helper to patch Session.get_devices
    _patch_session_devices(monkeypatch, [mock_device])

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


def test_backend_exposes_move_gate(
    monkeypatch: pytest.MonkeyPatch, mock_qdmi_device_factory: type[MockQDMIDevice]
) -> None:
    """Backend exposes a device's 'move' operation as an opaque MoveGate in the Target."""
    mock_device = mock_qdmi_device_factory(
        name="Test Device with MOVE",
        num_qubits=2,
        operations=["move", "cz", "measure"],
    )

    _patch_session_devices(monkeypatch, [mock_device])

    provider = QDMIProvider()
    backend = provider.get_backend("Test Device with MOVE")

    assert "move" in backend.target.operation_names
    move_instruction = backend.target.operation_from_name("move")
    assert isinstance(move_instruction, MoveGate)
    assert move_instruction.num_qubits == 2


def test_backend_qasm_conversion_no_supported_formats(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Backend should raise UnsupportedFormatError when no supported program formats exist."""
    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    qc.measure_all()

    device = mock_qdmi_device_factory(num_qubits=2, operations=["cz", "measure"])
    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type]

    with pytest.raises(UnsupportedFormatError, match="No supported program formats found"):
        backend._convert_circuit(qc, [])  # noqa: SLF001


def test_backend_qasm3_conversion_success(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Backend should successfully convert circuit to QASM3."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    device = mock_qdmi_device_factory(num_qubits=2, operations=["h", "cx", "measure"])
    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type]

    program, fmt = backend._convert_circuit(qc, [fomac.ProgramFormat.QASM3])  # noqa: SLF001

    assert fmt == fomac.ProgramFormat.QASM3
    assert "OPENQASM 3" in program
    assert "h q[0]" in program
    assert "cx q[0], q[1]" in program


def test_backend_qasm2_conversion_success(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Backend should successfully convert circuit to QASM2."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    device = mock_qdmi_device_factory(num_qubits=2, operations=["h", "cx", "measure"])
    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type]

    program, fmt = backend._convert_circuit(qc, [fomac.ProgramFormat.QASM2])  # noqa: SLF001

    assert fmt == fomac.ProgramFormat.QASM2
    assert "OPENQASM 2.0" in program
    assert "h q[0]" in program
    assert "cx q[0],q[1]" in program


def test_backend_qasm3_preferred_over_qasm2(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Backend should prefer QASM3 over QASM2 when both are available."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.measure_all()

    device = mock_qdmi_device_factory(num_qubits=2, operations=["h", "measure"])
    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type]

    # When both formats are available, QASM3 should be chosen
    program, fmt = backend._convert_circuit(qc, [fomac.ProgramFormat.QASM2, fomac.ProgramFormat.QASM3])  # noqa: SLF001

    assert fmt == fomac.ProgramFormat.QASM3
    assert "OPENQASM 3" in program


def test_backend_uses_iqm_json_when_supported(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Test that backend uses IQM JSON format when supported."""
    device = mock_qdmi_device_factory(num_qubits=2, operations=["r", "cz", "measure"])

    submitted_format: fomac.ProgramFormat | None = None

    def mock_supported_formats() -> list[fomac.ProgramFormat]:
        return [fomac.ProgramFormat.IQM_JSON, fomac.ProgramFormat.QASM3]

    def mock_submit_job(program: str, program_format: fomac.ProgramFormat, num_shots: int) -> MockQDMIDevice.MockJob:  # noqa: ARG001
        nonlocal submitted_format
        submitted_format = program_format
        return device.MockJob(num_clbits=2, shots=num_shots)

    device.supported_program_formats = mock_supported_formats  # ty: ignore[invalid-assignment]
    device.submit_job = mock_submit_job  # ty: ignore[invalid-assignment]

    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type]
    qc = QuantumCircuit(2)
    qc.r(1.5708, 0.0, 0)
    qc.cz(0, 1)
    qc.measure_all()

    backend.run(qc, shots=100)

    assert submitted_format == fomac.ProgramFormat.IQM_JSON


def test_backend_iqm_json_preferred_over_qasm(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Test that IQM JSON takes priority over QASM formats."""
    device = mock_qdmi_device_factory(num_qubits=2, operations=["r", "cz", "measure"])

    submitted_format: fomac.ProgramFormat | None = None

    def mock_supported_formats() -> list[fomac.ProgramFormat]:
        return [fomac.ProgramFormat.QASM2, fomac.ProgramFormat.QASM3, fomac.ProgramFormat.IQM_JSON]

    def mock_submit_job(program: str, program_format: fomac.ProgramFormat, num_shots: int) -> MockQDMIDevice.MockJob:  # noqa: ARG001
        nonlocal submitted_format
        submitted_format = program_format
        return device.MockJob(num_clbits=2, shots=num_shots)

    device.supported_program_formats = mock_supported_formats  # ty: ignore[invalid-assignment]
    device.submit_job = mock_submit_job  # ty: ignore[invalid-assignment]

    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type]
    qc = QuantumCircuit(2)
    qc.r(1.5708, 0.0, 0)
    qc.cz(0, 1)
    qc.measure_all()

    backend.run(qc, shots=100)

    assert submitted_format == fomac.ProgramFormat.IQM_JSON


@pytest.mark.parametrize(
    ("qasm_module_name", "program_format"),
    [
        ("qasm3", fomac.ProgramFormat.QASM3),
        ("qasm2", fomac.ProgramFormat.QASM2),
    ],
)
def test_backend_qasm_conversion_failure(
    monkeypatch: pytest.MonkeyPatch,
    qasm_module_name: str,
    program_format: fomac.ProgramFormat,
    mock_qdmi_device_factory: type[MockQDMIDevice],
) -> None:
    """Backend should raise TranslationError when QASM conversion fails."""
    qasm_module = qasm3 if qasm_module_name == "qasm3" else qasm2

    # Monkeypatch qasm dumps to raise an exception
    def failing_dumps(circuit: object) -> NoReturn:  # noqa: ARG001
        msg = f"Simulated {qasm_module_name.upper()} conversion failure"
        raise ValueError(msg)

    monkeypatch.setattr(qasm_module, "dumps", failing_dumps)

    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    qc.measure_all()

    device = mock_qdmi_device_factory(num_qubits=2, operations=["cz", "measure"])
    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type]

    with pytest.raises(TranslationError, match=f"Failed to convert circuit to {qasm_module_name.upper()}"):
        backend._convert_circuit(qc, [program_format])  # noqa: SLF001


def test_backend_unsupported_format_error(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Backend should raise UnsupportedFormatError when only unsupported formats available."""
    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    qc.measure_all()

    device = mock_qdmi_device_factory(num_qubits=2, operations=["cz", "measure"])
    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type]

    # Test with QPY format which is not supported for conversion from Qiskit
    with pytest.raises(
        UnsupportedFormatError, match="No conversion from Qiskit to any of the supported program formats"
    ):
        backend._convert_circuit(qc, [fomac.ProgramFormat.QPY])  # noqa: SLF001


def test_map_operation_returns_none_for_unknown() -> None:
    """Unknown FoMaC operations cannot be mapped to Qiskit gates."""
    assert QDMIBackend._map_operation_to_gate("unknown_gate") is None  # noqa: SLF001
    assert QDMIBackend._map_operation_to_gate("custom_op") is None  # noqa: SLF001
    assert QDMIBackend._map_operation_to_gate("") is None  # noqa: SLF001


def test_map_operation_to_move_gate() -> None:
    """MOVE operations map to an opaque 2-qubit gate."""
    gate = QDMIBackend._map_operation_to_gate("move")  # noqa: SLF001
    assert gate is not None
    assert gate.name == "move"
    assert gate.num_qubits == 2


def test_map_qiskit_gate_to_operation_names() -> None:
    """Test the inverse gate name mapping function comprehensively."""
    # Basic gates map to themselves
    assert QDMIBackend._map_qiskit_gate_to_operation_names("x") == {"x"}  # noqa: SLF001
    assert QDMIBackend._map_qiskit_gate_to_operation_names("h") == {"h"}  # noqa: SLF001
    assert QDMIBackend._map_qiskit_gate_to_operation_names("cz") == {"cz"}  # noqa: SLF001

    # Aliases: gates with multiple naming conventions return all possible aliases
    id_names = QDMIBackend._map_qiskit_gate_to_operation_names("id")  # noqa: SLF001
    assert id_names == {"id", "i"}
    assert QDMIBackend._map_qiskit_gate_to_operation_names("i") == id_names  # noqa: SLF001

    cx_names = QDMIBackend._map_qiskit_gate_to_operation_names("cx")  # noqa: SLF001
    assert cx_names == {"cx", "cnot"}
    assert QDMIBackend._map_qiskit_gate_to_operation_names("cnot") == cx_names  # noqa: SLF001

    # Device-specific aliases: bidirectional consistency for R/PRX (IQM naming)
    r_names = QDMIBackend._map_qiskit_gate_to_operation_names("r")  # noqa: SLF001
    assert r_names == {"r", "prx"}
    assert QDMIBackend._map_qiskit_gate_to_operation_names("prx") == r_names  # noqa: SLF001

    p_names = QDMIBackend._map_qiskit_gate_to_operation_names("p")  # noqa: SLF001
    assert p_names == {"p", "phase"}
    assert QDMIBackend._map_qiskit_gate_to_operation_names("phase") == p_names  # noqa: SLF001

    # Case-insensitive matching
    assert QDMIBackend._map_qiskit_gate_to_operation_names("X") == {"x"}  # noqa: SLF001
    assert QDMIBackend._map_qiskit_gate_to_operation_names("CX") == {"cx", "cnot"}  # noqa: SLF001

    # MOVE operation is represented as a real gate for IQM devices
    assert QDMIBackend._map_qiskit_gate_to_operation_names("move") == {"move"}  # noqa: SLF001
    assert QDMIBackend._map_qiskit_gate_to_operation_names("MOVE") == {"move"}  # noqa: SLF001

    # Fallback for unknown gates (returns lowercase name)
    assert QDMIBackend._map_qiskit_gate_to_operation_names("unknown") == {"unknown"}  # noqa: SLF001
    assert QDMIBackend._map_qiskit_gate_to_operation_names("CUSTOM") == {"custom"}  # noqa: SLF001


def test_backend_validation_uses_inverse_mapping(
    monkeypatch: pytest.MonkeyPatch, mock_qdmi_device_factory: type[MockQDMIDevice]
) -> None:
    """Backend validation correctly uses inverse mapping to handle device-specific naming."""
    # Create a mock device that uses 'prx' instead of 'r' (like IQM devices)
    mock_device = mock_qdmi_device_factory(
        name="Test Device with PRX",
        num_qubits=2,
        operations=["prx", "cz", "measure"],  # Uses 'prx' instead of 'r'
    )

    # Use helper to patch Session.get_devices
    _patch_session_devices(monkeypatch, [mock_device])

    provider = QDMIProvider()
    backend = provider.get_backend("Test Device with PRX")

    # Create a circuit with the 'r' gate (Qiskit's name)
    qc = QuantumCircuit(2)
    theta = Parameter("theta")
    phi = Parameter("phi")
    qc.r(theta, phi, 0)  # Qiskit uses 'r'
    qc.cz(0, 1)
    qc.measure_all()

    # Bind parameters before running
    qc_bound = qc.assign_parameters({theta: 1.5708, phi: 0.0})

    # This should NOT raise UnsupportedOperationError because the inverse mapping
    # knows that Qiskit's 'r' can map to device's 'prx'
    job = backend.run(qc_bound, shots=100)
    assert job is not None


def test_qiskit_to_iqm_json_simple_circuit(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Test conversion of a simple circuit to IQM JSON."""
    device = mock_qdmi_device_factory(
        name="IQM Device",
        num_qubits=2,
        operations=["r", "cz", "measure", "barrier"],
    )

    qc = QuantumCircuit(2, 2)
    qc.r(1.5708, 0.0, 0)
    qc.cz(0, 1)
    qc.measure_all()

    json_str = qiskit_to_iqm_json(qc, device)  # ty: ignore[invalid-argument-type]
    program = json.loads(json_str)

    assert "name" in program
    assert "metadata" in program
    assert "instructions" in program
    assert isinstance(program["instructions"], list)
    assert len(program["instructions"]) == 5  # r, cz, barrier, measure, measure
    instr_names = [instr["name"] for instr in program["instructions"]]
    assert instr_names == ["prx", "cz", "barrier", "measure", "measure"]


def test_qiskit_to_iqm_json_prx_parameters(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Test that R gates are converted to PRX with correct parameters."""
    device = mock_qdmi_device_factory(num_qubits=1, operations=["r", "measure"])

    angle = np.pi / 2
    phase = np.pi / 4
    qc = QuantumCircuit(1, 1)
    qc.r(angle, phase, 0)
    qc.measure_all()

    json_str = qiskit_to_iqm_json(qc, device)  # ty: ignore[invalid-argument-type]
    program = json.loads(json_str)

    prx_instr = program["instructions"][0]
    assert prx_instr["name"] == "prx"
    assert "args" in prx_instr
    assert "angle_t" in prx_instr["args"]
    assert "phase_t" in prx_instr["args"]

    expected_angle_t = angle / (2 * np.pi)
    expected_phase_t = phase / (2 * np.pi)
    assert abs(prx_instr["args"]["angle_t"] - expected_angle_t) < 1e-10
    assert abs(prx_instr["args"]["phase_t"] - expected_phase_t) < 1e-10


def test_qiskit_to_iqm_json_barrier(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Test that barriers are correctly converted."""
    device = mock_qdmi_device_factory(num_qubits=3, operations=["barrier"])

    qc = QuantumCircuit(3)
    qc.barrier([0, 1, 2])

    json_str = qiskit_to_iqm_json(qc, device)  # ty: ignore[invalid-argument-type]
    program = json.loads(json_str)

    barrier_instr = program["instructions"][0]
    assert barrier_instr["name"] == "barrier"
    assert len(barrier_instr["locus"]) == 3
    assert barrier_instr["args"] == {}


def test_qiskit_to_iqm_json_cz_gate(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Test that CZ gates are correctly converted."""
    device = mock_qdmi_device_factory(num_qubits=2, operations=["cz"])

    qc = QuantumCircuit(2)
    qc.cz(0, 1)

    json_str = qiskit_to_iqm_json(qc, device)  # ty: ignore[invalid-argument-type]
    program = json.loads(json_str)

    cz_instr = program["instructions"][0]
    assert cz_instr["name"] == "cz"
    assert len(cz_instr["locus"]) == 2
    assert cz_instr["args"] == {}


def test_qiskit_to_iqm_json_move_gate(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Test that MOVE gates are correctly converted to IQM JSON."""
    device = mock_qdmi_device_factory(num_qubits=2, operations=["move"])

    qc = QuantumCircuit(2)
    qc.append(MoveGate(), [0, 1])

    json_str = qiskit_to_iqm_json(qc, device)  # ty: ignore[invalid-argument-type]
    program = json.loads(json_str)

    move_instr = program["instructions"][0]
    assert move_instr["name"] == "move"
    assert move_instr["locus"] == ["site_0", "site_1"]
    assert move_instr["args"] == {}


def test_qiskit_to_iqm_json_measure_keys(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Test that measurements generate correct keys."""
    device = mock_qdmi_device_factory(num_qubits=2, operations=["measure"])

    qc = QuantumCircuit(2, 2)
    qc.measure_all()

    json_str = qiskit_to_iqm_json(qc, device)  # ty: ignore[invalid-argument-type]
    program = json.loads(json_str)

    barr = program["instructions"][0]
    meas0 = program["instructions"][1]
    meas1 = program["instructions"][2]

    assert barr["name"] == "barrier"
    assert barr["args"] == {}
    assert meas0["name"] == "measure"
    assert "key" in meas0["args"]
    assert meas1["name"] == "measure"
    assert "key" in meas1["args"]
    assert meas0["args"]["key"] != meas1["args"]["key"]


def test_qiskit_to_iqm_json_unsupported_operation(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Test that unsupported operations raise UnsupportedOperationError."""
    device = mock_qdmi_device_factory(num_qubits=1, operations=[])

    qc = QuantumCircuit(1)
    qc.h(0)

    with pytest.raises(UnsupportedOperationError, match="not supported in IQM JSON format"):
        qiskit_to_iqm_json(qc, device)  # ty: ignore[invalid-argument-type]


def test_qiskit_to_iqm_json_circuit_name(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Test that circuit name is preserved in IQM JSON."""
    device = mock_qdmi_device_factory(num_qubits=1, operations=["measure"])

    qc = QuantumCircuit(1, 1, name="my_test_circuit")
    qc.measure_all()

    json_str = qiskit_to_iqm_json(qc, device)  # ty: ignore[invalid-argument-type]
    program = json.loads(json_str)

    assert program["name"] == "my_test_circuit"


def test_qiskit_to_iqm_json_unbound_parameters(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Test that circuits with unbound parameters raise UnsupportedOperationError."""
    device = mock_qdmi_device_factory(num_qubits=2, operations=["r", "cz", "measure"])

    # Create circuit with unbound parameters
    theta = Parameter("theta")
    phi = Parameter("phi")
    qc = QuantumCircuit(2, 2)
    qc.r(theta, phi, 0)
    qc.cz(0, 1)
    qc.measure_all()

    # Should raise UnsupportedOperationError with clear message
    with pytest.raises(UnsupportedOperationError) as exc_info:
        qiskit_to_iqm_json(qc, device)  # ty: ignore[invalid-argument-type]

    error_msg = str(exc_info.value)
    assert "unbound parameters" in error_msg.lower()
    assert "phi" in error_msg
    assert "theta" in error_msg
    assert "assign_parameters" in error_msg


def test_qiskit_to_iqm_json_bound_parameters(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Test that circuits with bound parameters work correctly."""
    device = mock_qdmi_device_factory(num_qubits=2, operations=["r", "cz", "measure"])

    # Create circuit with parameters
    theta = Parameter("theta")
    phi = Parameter("phi")
    qc = QuantumCircuit(2, 2)
    qc.r(theta, phi, 0)
    qc.cz(0, 1)
    qc.measure_all()

    # Bind parameters
    qc_bound = qc.assign_parameters({theta: np.pi / 2, phi: 0.0})

    # Should convert successfully
    json_str = qiskit_to_iqm_json(qc_bound, device)  # ty: ignore[invalid-argument-type]
    program = json.loads(json_str)

    assert "instructions" in program
    # r, cz, barrier (from measure_all), measure, measure
    assert len(program["instructions"]) == 5

    # Check PRX instruction has correct parameters
    prx_instr = program["instructions"][0]
    assert prx_instr["name"] == "prx"
    expected_angle_t = (np.pi / 2) / (2 * np.pi)
    expected_phase_t = 0.0 / (2 * np.pi)
    assert abs(prx_instr["args"]["angle_t"] - expected_angle_t) < 1e-10
    assert abs(prx_instr["args"]["phase_t"] - expected_phase_t) < 1e-10


def test_qiskit_to_iqm_json_unregistered_classical_bit(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Test that measurements to unregistered classical bits raise TranslationError."""
    device = mock_qdmi_device_factory(num_qubits=2, operations=["cz", "measure"])

    # Create circuit with unregistered classical bit
    qc = QuantumCircuit(2)
    standalone_clbit = Clbit()
    qc.add_bits([standalone_clbit])
    qc.cz(0, 1)
    qc.measure(0, standalone_clbit)

    # Should raise TranslationError with clear message
    with pytest.raises(TranslationError) as exc_info:
        qiskit_to_iqm_json(qc, device)  # ty: ignore[invalid-argument-type]

    error_msg = str(exc_info.value)
    assert "unregistered classical bit" in error_msg.lower()
    assert "ClassicalRegister" in error_msg


def test_qiskit_to_iqm_json_registered_classical_bit(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Test that measurements to registered classical bits work correctly."""
    device = mock_qdmi_device_factory(num_qubits=2, operations=["cz", "measure"])

    # Create circuit with registered classical bits (standard approach)
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure(0, 0)
    qc.measure(1, 1)

    # Should convert successfully
    json_str = qiskit_to_iqm_json(qc, device)  # ty: ignore[invalid-argument-type]
    program = json.loads(json_str)

    assert "instructions" in program
    assert len(program["instructions"]) == 3  # cz, measure, measure

    # Check measurements have keys
    measure_instrs = [instr for instr in program["instructions"] if instr["name"] == "measure"]
    assert len(measure_instrs) == 2
    assert "key" in measure_instrs[0]["args"]
    assert "key" in measure_instrs[1]["args"]
    assert measure_instrs[0]["args"]["key"] != measure_instrs[1]["args"]["key"]
