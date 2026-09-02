# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Backend tests using a mock QDMI device implementation."""

from __future__ import annotations

import re
import secrets
import string
import warnings
from typing import TYPE_CHECKING, ClassVar, NoReturn

import pytest
from qiskit import qasm2, qasm3
from qiskit.circuit import ClassicalRegister, Clbit, Gate, Parameter, QuantumCircuit

from mqt.core.plugins.qiskit import (
    QDMIBackend,
    QDMIProvider,
    TranslationError,
    UnsupportedFormatError,
    program_serializer,
    register_program_serializer,
    unregister_program_serializer,
)
from mqt.core.qdmi import Job as QDMIJobHandle
from mqt.core.qdmi import ProgramFormat

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from qiskit.circuit import Instruction


class MockQDMIDevice:
    """Mock QDMI device for testing with configurable properties and job execution.

    This class implements the QDMI device interface for testing purposes,
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
            elif name in {"cz", "cx", "cnot", "cy", "ch", "swap", "iswap", "hop"}:
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
        """Mock QDMI job with simulated results."""

        def __init__(self, num_clbits: int, shots: int) -> None:
            """Initialize mock job with number of classical bits and shots."""
            self._num_clbits = num_clbits
            self._shots = shots
            alphabet = string.ascii_lowercase + string.digits
            self._id = "mock-job-" + "".join(secrets.choice(alphabet) for _ in range(8))
            self._status = QDMIJobHandle.Status.DONE
            self._counts: dict[str, int] | None = None

        @property
        def id(self) -> str:
            """The job ID."""
            return self._id

        @property
        def num_shots(self) -> int:
            """The number of shots."""
            return self._shots

        def check(self) -> QDMIJobHandle.Status:
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
    def supported_program_formats() -> list[ProgramFormat]:
        """Return list of supported program formats."""
        return [ProgramFormat.QASM2, ProgramFormat.QASM3]

    def submit_job(self, program: str, program_format: ProgramFormat, num_shots: int) -> MockJob:  # ruff:ignore[unused-method-argument]
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


def _patch_registered_devices(monkeypatch: pytest.MonkeyPatch, devices: list[MockQDMIDevice]) -> None:
    """Make the driver functions expose the given mock devices."""
    device_ids = [f"test.device.{index}" for index in range(len(devices))]
    devices_by_id = dict(zip(device_ids, devices, strict=True))
    monkeypatch.setattr("mqt.core.plugins.qiskit.provider.registered_device_ids", lambda: device_ids)
    monkeypatch.setattr(
        "mqt.core.plugins.qiskit.backend.open_device",
        lambda device_id, **_kwargs: devices_by_id[device_id],
    )


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

    # Use helper to patch registered driver devices
    _patch_registered_devices(monkeypatch, [mock_device])

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

    # Use helper to patch registered driver devices
    _patch_registered_devices(monkeypatch, [mock_device])

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


def test_backend_warns_on_device_native_operation(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Backend skips a device operation that no Qiskit gate represents."""
    mock_device = mock_qdmi_device_factory(
        name="Test Device with a device-native operation",
        num_qubits=2,
        operations=["hop", "cz", "measure"],
    )

    with pytest.warns(UserWarning, match="'hop' cannot be mapped to a Qiskit gate"):
        backend = QDMIBackend(device=mock_device)  # ty: ignore[invalid-argument-type]

    assert "hop" not in backend.target.operation_names


def test_subclass_extra_gates_appear_in_target(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """A subclass represents a device-native operation through _EXTRA_GATES."""

    class HopGate(Gate):
        """An opaque two-qubit gate outside Qiskit's standard library."""

        def __init__(self) -> None:
            super().__init__("hop", 2, [])

    class HoppingBackend(QDMIBackend):
        """Backend for a device whose native gate set includes 'hop'."""

        _EXTRA_GATES: ClassVar[dict[str, Instruction | type[Instruction]]] = {"hop": HopGate()}

    mock_device = mock_qdmi_device_factory(
        name="Test Device with a device-native operation",
        num_qubits=2,
        operations=["hop", "cz", "measure"],
    )

    backend = HoppingBackend(device=mock_device)  # ty: ignore[invalid-argument-type]

    assert "hop" in backend.target.operation_names
    hop_instruction = backend.target.operation_from_name("hop")
    assert isinstance(hop_instruction, HopGate)
    assert hop_instruction.num_qubits == 2

    # The base class keeps its own mappings
    assert QDMIBackend._map_operation_to_gate("hop") is None  # ruff:ignore[private-member-access]


def _record_submissions(device: MockQDMIDevice) -> list[tuple[str | bytes, ProgramFormat]]:
    """Make a mock device record every submission instead of parsing the program.

    Args:
        device: The mock device to change.

    Returns:
        The list the device appends each (program, format) pair to.
    """
    submissions: list[tuple[str | bytes, ProgramFormat]] = []

    def submit_job(program: str | bytes, program_format: ProgramFormat, num_shots: int) -> MockQDMIDevice.MockJob:
        submissions.append((program, program_format))
        return device.MockJob(num_clbits=2, shots=num_shots)

    device.submit_job = submit_job  # ty: ignore[invalid-assignment]
    return submissions


def test_backend_serialization_without_supported_formats(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Backend should raise UnsupportedFormatError when the device reports no program format."""
    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    qc.measure_all()

    device = mock_qdmi_device_factory(num_qubits=2, operations=["cz", "measure"])
    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type]

    with pytest.raises(UnsupportedFormatError, match="reports no supported program formats"):
        backend._serialize_circuit(qc, [])  # ruff:ignore[private-member-access]


def test_backend_qasm3_serialization_success(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Backend should successfully serialize a circuit into OpenQASM 3."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    device = mock_qdmi_device_factory(num_qubits=2, operations=["h", "cx", "measure"])
    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type]

    program, fmt = backend._serialize_circuit(qc, [ProgramFormat.QASM3])  # ruff:ignore[private-member-access]

    assert fmt == ProgramFormat.QASM3
    assert isinstance(program, str)
    assert "OPENQASM 3" in program
    assert "h q[0]" in program
    assert "cx q[0], q[1]" in program


def test_backend_qasm3_zero_initializes_classical_bits(
    mock_qdmi_device_factory: type[MockQDMIDevice],
) -> None:
    """Initialize each register and loose bit before measurement."""
    one = ClassicalRegister(1, "one")
    many = ClassicalRegister(2, "many")
    loose = Clbit()
    qc = QuantumCircuit(2)
    qc.add_register(one, many)
    qc.add_bits([loose])
    qc.measure(0, many[0])

    device = mock_qdmi_device_factory(num_qubits=2, operations=["measure"])
    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type]

    program, fmt = backend._serialize_circuit(qc, [ProgramFormat.QASM3])  # ruff:ignore[private-member-access]

    assert fmt == ProgramFormat.QASM3
    assert isinstance(program, str)
    assert program.count("one = 0;") == 1
    assert program.count("many = 0;") == 1
    assert "one[0] = false;" not in program
    assert "many[0] = false;" not in program
    assert "many[1] = false;" not in program
    loose_initialization = re.search(r"(?m)^_bit\d+ = false;$", program)
    assert loose_initialization is not None
    measurement = "many[0] = measure q[0];"
    for initialization in ("one = 0;", "many = 0;", loose_initialization.group()):
        assert program.index(initialization) < program.index(measurement)


def test_backend_qasm2_serialization_success(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Backend should successfully serialize a circuit into OpenQASM 2."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    device = mock_qdmi_device_factory(num_qubits=2, operations=["h", "cx", "measure"])
    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type]

    program, fmt = backend._serialize_circuit(qc, [ProgramFormat.QASM2])  # ruff:ignore[private-member-access]

    assert fmt == ProgramFormat.QASM2
    assert isinstance(program, str)
    assert "OPENQASM 2.0" in program
    assert "h q[0]" in program
    assert "cx q[0],q[1]" in program


def test_backend_respects_format_preference(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """The preference order decides the format, not the order the device reports."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.measure_all()

    device = mock_qdmi_device_factory(num_qubits=2, operations=["h", "measure"])
    # The device reports OpenQASM 2 first, but OpenQASM 3 outranks it.
    device.supported_program_formats = lambda: [ProgramFormat.QASM2, ProgramFormat.QASM3]  # ty: ignore[invalid-assignment]
    submissions = _record_submissions(device)

    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type]
    backend.run(qc, shots=100)

    assert len(submissions) == 1
    program, fmt = submissions[0]
    assert fmt == ProgramFormat.QASM3
    assert isinstance(program, str)
    assert "OPENQASM 3" in program


@pytest.fixture
def registered_serializer() -> Iterator[ProgramFormat]:
    """Register a text program serializer for CUSTOM1 and remove it after the test.

    Yields:
        The program format the serializer is registered for.
    """

    def serializer(circuit: QuantumCircuit, backend: QDMIBackend) -> str:  # ruff:ignore[unused-function-argument]
        return f"CUSTOM1 program for {circuit.name}"

    register_program_serializer(ProgramFormat.CUSTOM1, serializer)
    yield ProgramFormat.CUSTOM1
    unregister_program_serializer(ProgramFormat.CUSTOM1)


def test_backend_uses_registered_serializer(
    mock_qdmi_device_factory: type[MockQDMIDevice], registered_serializer: ProgramFormat
) -> None:
    """Backend serializes through a registered serializer when the device supports its format."""
    device = mock_qdmi_device_factory(num_qubits=2, operations=["r", "cz", "measure"])
    device.supported_program_formats = lambda: [registered_serializer, ProgramFormat.QASM3]  # ty: ignore[invalid-assignment]
    submissions = _record_submissions(device)

    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type]
    qc = QuantumCircuit(2, name="bell")
    qc.r(1.5708, 0.0, 0)
    qc.cz(0, 1)

    backend.run(qc, shots=100)

    assert submissions == [("CUSTOM1 program for bell", registered_serializer)]


def test_backend_prefers_registered_serializer_over_qasm(
    mock_qdmi_device_factory: type[MockQDMIDevice], registered_serializer: ProgramFormat
) -> None:
    """A registered serializer takes priority over the built-in OpenQASM serializers."""
    device = mock_qdmi_device_factory(num_qubits=2, operations=["r", "cz", "measure"])
    device.supported_program_formats = lambda: [ProgramFormat.QASM2, ProgramFormat.QASM3, registered_serializer]  # ty: ignore[invalid-assignment]
    submissions = _record_submissions(device)

    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type]
    qc = QuantumCircuit(2, name="bell")
    qc.r(1.5708, 0.0, 0)
    qc.cz(0, 1)
    qc.measure_all()

    backend.run(qc, shots=100)

    assert submissions == [("CUSTOM1 program for bell", registered_serializer)]


@pytest.fixture
def registered_binary_serializer() -> Iterator[tuple[ProgramFormat, bytes]]:
    """Register a binary program serializer for QPY and remove it after the test.

    Yields:
        The program format the serializer is registered for and the payload it
        returns.
    """
    payload = b"QPY\x00\x01binary program"

    def serializer(circuit: QuantumCircuit, backend: QDMIBackend) -> bytes:  # ruff:ignore[unused-function-argument]
        return payload

    register_program_serializer(ProgramFormat.QPY, serializer)
    yield ProgramFormat.QPY, payload
    unregister_program_serializer(ProgramFormat.QPY)


def test_backend_submits_binary_payload(
    mock_qdmi_device_factory: type[MockQDMIDevice], registered_binary_serializer: tuple[ProgramFormat, bytes]
) -> None:
    """A binary format reaches the device as the exact bytes the serializer returned."""
    fmt, payload = registered_binary_serializer
    device = mock_qdmi_device_factory(num_qubits=2, operations=["h", "cz", "measure"])
    device.supported_program_formats = lambda: [fmt, ProgramFormat.QASM3]  # ty: ignore[invalid-assignment]
    submissions = _record_submissions(device)

    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type]
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cz(0, 1)
    qc.measure_all()

    backend.run(qc, shots=100)

    assert submissions == [(payload, fmt)]


@pytest.fixture
def mistyped_serializer() -> Iterator[ProgramFormat]:
    """Register a serializer that returns bytes for a text format.

    Yields:
        The text program format the serializer is registered for.
    """

    def serializer(circuit: QuantumCircuit, backend: QDMIBackend) -> bytes:  # ruff:ignore[unused-function-argument]
        return b"not a string"

    register_program_serializer(ProgramFormat.CUSTOM2, serializer)
    yield ProgramFormat.CUSTOM2
    unregister_program_serializer(ProgramFormat.CUSTOM2)


def test_backend_rejects_wrong_payload_type(
    mock_qdmi_device_factory: type[MockQDMIDevice], mistyped_serializer: ProgramFormat
) -> None:
    """A serializer that returns the wrong payload type for its format fails."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.measure_all()

    device = mock_qdmi_device_factory(num_qubits=2, operations=["h", "measure"])
    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type]

    with pytest.raises(TranslationError, match="returned bytes, but CUSTOM2 requires str"):
        backend._serialize_circuit(qc, [mistyped_serializer])  # ruff:ignore[private-member-access]


@pytest.fixture
def replaced_qasm3_serializer() -> Iterator[str]:
    """Replace the built-in OpenQASM 3 serializer and restore it after the test.

    Yields:
        The program the replacement returns.
    """
    program = "OPENQASM 3.0; // replaced"
    original = program_serializer(ProgramFormat.QASM3)
    assert original is not None

    def serializer(circuit: QuantumCircuit, backend: QDMIBackend) -> str:  # ruff:ignore[unused-function-argument]
        return program

    register_program_serializer(ProgramFormat.QASM3, serializer, replace=True)
    yield program
    register_program_serializer(ProgramFormat.QASM3, original, replace=True)


def test_backend_uses_replaced_qasm3_serializer(
    mock_qdmi_device_factory: type[MockQDMIDevice], replaced_qasm3_serializer: str
) -> None:
    """Replacing the built-in OpenQASM 3 serializer changes what the backend submits."""
    device = mock_qdmi_device_factory(num_qubits=2, operations=["h", "measure"])
    submissions = _record_submissions(device)

    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type]
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.measure_all()

    backend.run(qc, shots=100)

    assert submissions == [(replaced_qasm3_serializer, ProgramFormat.QASM3)]


def test_backend_rejects_device_without_program_payload(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """A device that only accepts CALIBRATION has no format a circuit can go into."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.measure_all()

    device = mock_qdmi_device_factory(num_qubits=2, operations=["h", "measure"])
    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type]

    with pytest.raises(UnsupportedFormatError, match="No program serializer for any format the device supports"):
        backend._serialize_circuit(qc, [ProgramFormat.CALIBRATION])  # ruff:ignore[private-member-access]


@pytest.mark.parametrize(
    ("qasm_module_name", "program_format"),
    [
        ("qasm3", ProgramFormat.QASM3),
        ("qasm2", ProgramFormat.QASM2),
    ],
)
def test_backend_qasm_serialization_failure(
    monkeypatch: pytest.MonkeyPatch,
    qasm_module_name: str,
    program_format: ProgramFormat,
    mock_qdmi_device_factory: type[MockQDMIDevice],
) -> None:
    """Backend should raise TranslationError when OpenQASM serialization fails."""
    qasm_module = qasm3 if qasm_module_name == "qasm3" else qasm2

    # Monkeypatch qasm dumps to raise an exception
    def failing_dumps(circuit: object) -> NoReturn:  # ruff:ignore[unused-function-argument]
        msg = f"Simulated {qasm_module_name.upper()} conversion failure"
        raise ValueError(msg)

    monkeypatch.setattr(qasm_module, "dumps", failing_dumps)

    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    qc.measure_all()

    device = mock_qdmi_device_factory(num_qubits=2, operations=["cz", "measure"])
    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type]

    with pytest.raises(TranslationError, match=f"Failed to serialize the circuit to {qasm_module_name.upper()}"):
        backend._serialize_circuit(qc, [program_format])  # ruff:ignore[private-member-access]


def test_backend_unsupported_format_error(mock_qdmi_device_factory: type[MockQDMIDevice]) -> None:
    """Backend should raise UnsupportedFormatError when no supported format has a serializer."""
    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    qc.measure_all()

    device = mock_qdmi_device_factory(num_qubits=2, operations=["cz", "measure"])
    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type]

    # MQT Core ships no QPY serializer, and no package registered one
    with pytest.raises(UnsupportedFormatError, match="No program serializer for any format the device supports"):
        backend._serialize_circuit(qc, [ProgramFormat.QPY])  # ruff:ignore[private-member-access]


def test_map_operation_returns_none_for_unknown() -> None:
    """Unknown QDMI operations cannot be mapped to Qiskit gates."""
    assert QDMIBackend._map_operation_to_gate("unknown_gate") is None  # ruff:ignore[private-member-access]
    assert QDMIBackend._map_operation_to_gate("custom_op") is None  # ruff:ignore[private-member-access]
    assert QDMIBackend._map_operation_to_gate("") is None  # ruff:ignore[private-member-access]


def test_map_qiskit_gate_to_operation_names() -> None:
    """Test the inverse gate name mapping function comprehensively."""
    # Basic gates map to themselves
    assert QDMIBackend._map_qiskit_gate_to_operation_names("x") == {"x"}  # ruff:ignore[private-member-access]
    assert QDMIBackend._map_qiskit_gate_to_operation_names("h") == {"h"}  # ruff:ignore[private-member-access]
    assert QDMIBackend._map_qiskit_gate_to_operation_names("cz") == {"cz"}  # ruff:ignore[private-member-access]

    # Aliases: gates with multiple naming conventions return all possible aliases
    id_names = QDMIBackend._map_qiskit_gate_to_operation_names("id")  # ruff:ignore[private-member-access]
    assert id_names == {"id", "i"}
    assert QDMIBackend._map_qiskit_gate_to_operation_names("i") == id_names  # ruff:ignore[private-member-access]

    cx_names = QDMIBackend._map_qiskit_gate_to_operation_names("cx")  # ruff:ignore[private-member-access]
    assert cx_names == {"cx", "cnot"}
    assert QDMIBackend._map_qiskit_gate_to_operation_names("cnot") == cx_names  # ruff:ignore[private-member-access]

    # Device-specific aliases: bidirectional consistency for R/PRX
    r_names = QDMIBackend._map_qiskit_gate_to_operation_names("r")  # ruff:ignore[private-member-access]
    assert r_names == {"r", "prx"}
    assert QDMIBackend._map_qiskit_gate_to_operation_names("prx") == r_names  # ruff:ignore[private-member-access]

    p_names = QDMIBackend._map_qiskit_gate_to_operation_names("p")  # ruff:ignore[private-member-access]
    assert p_names == {"p", "phase"}
    assert QDMIBackend._map_qiskit_gate_to_operation_names("phase") == p_names  # ruff:ignore[private-member-access]

    # Case-insensitive matching
    assert QDMIBackend._map_qiskit_gate_to_operation_names("X") == {"x"}  # ruff:ignore[private-member-access]
    assert QDMIBackend._map_qiskit_gate_to_operation_names("CX") == {"cx", "cnot"}  # ruff:ignore[private-member-access]

    # An operation without a Qiskit gate maps to itself
    assert QDMIBackend._map_qiskit_gate_to_operation_names("hop") == {"hop"}  # ruff:ignore[private-member-access]
    assert QDMIBackend._map_qiskit_gate_to_operation_names("HOP") == {"hop"}  # ruff:ignore[private-member-access]

    # Fallback for unknown gates (returns lowercase name)
    assert QDMIBackend._map_qiskit_gate_to_operation_names("unknown") == {"unknown"}  # ruff:ignore[private-member-access]
    assert QDMIBackend._map_qiskit_gate_to_operation_names("CUSTOM") == {"custom"}  # ruff:ignore[private-member-access]


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

    # Use helper to patch registered driver devices
    _patch_registered_devices(monkeypatch, [mock_device])

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
