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
import struct
from io import BytesIO
from math import pi
from typing import TYPE_CHECKING, NoReturn, cast
from unittest.mock import Mock

import pytest
from qiskit import qpy
from qiskit.circuit import (
    ClassicalRegister,
    Gate,
    QuantumCircuit,
)
from qiskit.circuit.library import CU3Gate, CUGate
from qiskit.quantum_info import Operator, SparsePauliOp

from mqt.core.mlir import CompiledProgram, serialize_program
from mqt.core.plugins.qiskit import (
    CircuitValidationError,
    QDMIBackend,
    TranslationError,
    UnsupportedOperationError,
)
from mqt.core.plugins.qiskit.exceptions import UnsupportedDeviceError
from mqt.core.qdmi import Job as QDMIJobHandle
from mqt.core.qdmi import ProgramFormat

pytestmark = pytest.mark.usefixtures("mock_device_snapshot")

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from qiskit.providers import Options

    from mqt.core.typing import QDMIJobParameters


class MockQDMIDevice:
    """Mock QDMI device for testing with configurable properties and job execution.

    This class implements the QDMI device interface for testing purposes,
    providing configurable device properties and mock job execution.
    """

    @staticmethod
    def query_custom_property(*_args: object) -> None:
        """No optional device metadata is advertised."""
        return

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
            elif name in {"r", "prx"}:
                self._qubits = 1
                self._params = 2
            elif name in {"cu", "cu3"}:
                self._qubits = 2
                self._params = 4 if name == "cu" else 3
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

        @staticmethod
        def query_custom_property(_property: object, _value_type: type[bytes]) -> bytes | None:
            """Model an operation without custom metadata.

            Returns:
                None, as for an absent QDMI custom property.
            """
            return None

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
            self.num_clbits = num_clbits
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

        @staticmethod
        def get_program_status(_program_index: int = 0) -> None:
            """This mock exposes only an aggregate job outcome."""
            return

        @staticmethod
        def wait() -> bool:
            """Return immediately for this completed job."""
            return True

        def get_counts(self, program_index: int = 0) -> dict[str, int]:  # ruff:ignore[unused-method-argument] Match the indexed result interface.
            """Get measurement counts with uniform random distribution.

            Returns:
                Dictionary mapping measurement outcomes to counts.
            """
            if self.num_clbits == 0:
                return {"": self._shots}

            if self._counts is None:
                # Generate random counts with uniform distribution
                num_outcomes = 2**self.num_clbits
                outcomes = [format(i, f"0{self.num_clbits}b") for i in range(num_outcomes)]

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

        def get_shots(self, program_index: int = 0) -> list[str]:
            """Raise unless the test device implements individual shots.

            Raises:
                NotImplementedError: This device only supports counts.
            """
            raise NotImplementedError

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
            operations = ["h", *(["cz"] if num_qubits >= 2 else []), "ry", "rz", "measure"]
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
        return [ProgramFormat.QASM3, ProgramFormat.QASM2]

    def try_submit_job(self, *_args: object, **_kwargs: object) -> None:
        """Reject native groups before submission to exercise independent jobs."""

    def submit_job(self, program: str, program_format: ProgramFormat, num_shots: int) -> MockJob:  # ruff:ignore[unused-method-argument]
        """Submit a mock job to the device.

        Args:
            program: The program parsed for classical bit count.
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
        matches_qasm3_arrays = re.findall(r"\bbit\[(\d+)]\s+\w+\s*(?:=\s*[^;]+)?;", program)
        count_qasm3_arrays = sum(int(m) for m in matches_qasm3_arrays)

        # Look for scalar-bit declarations in QASM3, including declarations with
        # optional initializer expressions.
        matches_qasm3_scalars = re.findall(r"\bbit(?!\s*\[)\s+\w+\s*(?:=\s*[^;]+)?;", program)
        count_qasm3_scalars = len(matches_qasm3_scalars)

        num_clbits = count_qasm2 + count_qasm3_arrays + count_qasm3_scalars
        return self.MockJob(num_clbits=num_clbits, shots=num_shots)


def _patch_client_devices(monkeypatch: pytest.MonkeyPatch, devices: list[MockQDMIDevice]) -> None:
    """Make the driver functions expose the given mock devices."""
    device_ids = [f"test.device.{index}" for index in range(len(devices))]
    devices_by_id = dict(zip(device_ids, devices, strict=True))
    monkeypatch.setattr(
        "mqt.core.plugins.qiskit.provider.QDMIProvider.device_ids",
        staticmethod(lambda: device_ids),
    )
    monkeypatch.setattr(
        "mqt.core.plugins.qiskit.backend.open_device",
        lambda device_id, **_kwargs: devices_by_id[device_id],
    )


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

    device.submit_job = submit_job
    return submissions


def test_backend_serialization_without_supported_formats() -> None:
    """Reject a device without a usable payload when taking its snapshot."""
    device = MockQDMIDevice(num_qubits=2, operations=["cz", "measure", "x"])
    device.supported_program_formats = list
    with pytest.raises(UnsupportedDeviceError, match="no supported program formats"):
        QDMIBackend(device)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("formats", "expected"),
    [
        ([ProgramFormat.QASM2], ProgramFormat.QASM2),
        ([ProgramFormat.QASM3], ProgramFormat.QASM3),
    ],
)
def test_backend_qasm_serialization(formats: list[ProgramFormat], expected: ProgramFormat) -> None:
    """Core preserves operations in each advertised OpenQASM format."""
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    device = MockQDMIDevice(num_qubits=2, operations=["h", "cx", "measure"])
    device.supported_program_formats = lambda: formats
    submissions = _record_submissions(device)
    QDMIBackend(device).run(circuit, shots=100)  # ty: ignore[invalid-argument-type]
    assert len(submissions) == 1
    program, fmt = submissions[0]
    assert fmt == expected
    assert isinstance(program, str)
    assert f"OPENQASM {2 if fmt == ProgramFormat.QASM2 else 3}" in program
    sites = ("q[0]", "q[1]") if fmt == ProgramFormat.QASM2 else ("$0", "$1")
    assert f"h {sites[0]}" in program
    assert f"cx {sites[0]},{sites[1]}" in program.replace(", ", ",")


@pytest.mark.parametrize("angle", [pi + 1e-9, 1e-12])
@pytest.mark.parametrize("program_format", [ProgramFormat.QASM2, ProgramFormat.QASM3])
def test_qasm_preserves_float_parameters(angle: float, program_format: ProgramFormat) -> None:
    """Executable serialization must not replace nearby values by constants."""
    device = MockQDMIDevice(operations=["rx", "measure"])
    device.supported_program_formats = lambda: [program_format]
    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type]
    circuit = QuantumCircuit(1)
    circuit.rx(angle, 0)
    program = serialize_program(circuit, target=backend._environment)  # ruff:ignore[private-member-access] Test the chosen snapshot.
    assert isinstance(program, CompiledProgram)
    assert isinstance(program.payload, str)
    match = re.search(r"rx\(([^)]+)\)", program.payload)
    assert match is not None
    assert float(match[1]).hex() == float(angle).hex()


@pytest.mark.parametrize("gate", [CU3Gate(0.1, 0.2, 0.3), CUGate(0.1, 0.2, 0.3, 0.7)])
@pytest.mark.parametrize("fmt", [ProgramFormat.QASM3, ProgramFormat.QIR_BASE_STRING, ProgramFormat.QPY])
def test_controlled_u_signatures_are_distinct(
    gate: CU3Gate | CUGate, fmt: ProgramFormat, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Advertising one controlled-U signature must not enable the other."""
    device = MockQDMIDevice(num_qubits=2, operations=[gate.name, "measure", "x"])
    monkeypatch.setattr(device, "supported_program_formats", lambda: [fmt])
    submissions = _record_submissions(device)
    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type]
    advertised = backend.target.operation_from_name(gate.name)
    assert isinstance(advertised, type(gate))
    assert len(advertised.params) == len(gate.params)
    circuit = QuantumCircuit(2)
    circuit.append(gate, [0, 1])
    backend.run(circuit, shots=1)
    if fmt == ProgramFormat.QIR_BASE_STRING:
        payload, _ = submissions[0]
        assert isinstance(payload, str)
        call = next(line for line in payload.splitlines() if f"call void @__quantum__qis__{gate.name}__body" in line)
        values = re.findall(r"double ([^,)]+)", call)
        angles = [
            struct.unpack(">d", bytes.fromhex(value[3:]))[0] if value.startswith("f0x") else float(value)
            for value in values
        ]
        assert angles == gate.params
    elif fmt == ProgramFormat.QPY:
        payload, _ = submissions[0]
        assert isinstance(payload, bytes)
        restored = qpy.load(BytesIO(payload))[0]
        assert restored.data[0].operation.name == gate.name
        assert restored.data[0].operation.params == gate.params
    other = CUGate(0.1, 0.2, 0.3, 0.7) if gate.name == "cu3" else CU3Gate(0.1, 0.2, 0.3)
    circuit.clear()
    circuit.append(other, [0, 1])
    with pytest.raises(UnsupportedOperationError, match="not advertised"):
        backend.run(circuit, shots=1)


@pytest.mark.parametrize("program_format", [ProgramFormat.QASM2, ProgramFormat.QASM3, ProgramFormat.QPY])
def test_native_serialization_rejects_invalid_native_placements_before_submitting_batch(
    monkeypatch: pytest.MonkeyPatch, program_format: ProgramFormat
) -> None:
    """Validate the whole batch against ordered native pairs before submission."""
    device = MockQDMIDevice(num_qubits=2, operations=["cx", "measure", "x"])
    operation = device.operations()[0]
    monkeypatch.setattr(operation, "site_pairs", lambda: [(device.sites()[0], device.sites()[1])])
    monkeypatch.setattr(device, "supported_program_formats", lambda: [program_format])
    submissions = _record_submissions(device)
    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type] Device boundary double.
    valid = QuantumCircuit(2)
    valid.cx(0, 1)
    valid.measure_all()
    invalid = QuantumCircuit(2)
    invalid.cx(1, 0)
    invalid.measure_all()

    with pytest.raises(UnsupportedOperationError, match=r"native device qubits \(1, 0\)"):
        backend.run([valid, invalid], shots=2)
    assert not submissions
    backend.run(valid, shots=2)
    assert len(submissions) == 1


@pytest.mark.parametrize("program_format", [ProgramFormat.QASM2, ProgramFormat.QASM3, ProgramFormat.QPY])
def test_native_serialization_rejects_excess_native_width(
    monkeypatch: pytest.MonkeyPatch, program_format: ProgramFormat
) -> None:
    """Native serialization cannot address qubits beyond the device width."""
    device = MockQDMIDevice(num_qubits=2, operations=["measure", "x"])
    monkeypatch.setattr(device, "supported_program_formats", lambda: [program_format])
    submissions = _record_submissions(device)
    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type] Device boundary double.
    circuit = QuantumCircuit(3)
    circuit.measure_all()
    with pytest.raises(CircuitValidationError, match="native device has 2"):
        backend.run(circuit, shots=2)
    assert not submissions


def test_native_sparse_site_addresses(monkeypatch: pytest.MonkeyPatch) -> None:
    """A disabled site does not shift physical wires or shrink their address range."""
    device = MockQDMIDevice(num_qubits=2, operations=["measure", "x"])
    sites = [MockQDMIDevice.MockSite(0), MockQDMIDevice.MockSite(2)]
    monkeypatch.setattr(device, "sites", lambda: sites)
    monkeypatch.setattr(device.operations()[0], "sites", lambda: sites)
    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type] Device boundary double.
    circuit = QuantumCircuit(3, 1)
    circuit.measure(2, 0)
    assert backend.target.num_qubits == 3
    backend.validate_circuit(circuit, native=True)
    circuit.measure(1, 0)
    with pytest.raises(UnsupportedOperationError, match=r"native device qubits \(1,\)"):
        backend.validate_circuit(circuit, native=True)


@pytest.mark.parametrize("program_format", [ProgramFormat.QASM2, ProgramFormat.QASM3, ProgramFormat.QPY])
def test_builtin_preflight_maps_control_flow_operands(
    monkeypatch: pytest.MonkeyPatch, program_format: ProgramFormat
) -> None:
    """Validate block-local operands against their enclosing native qubits."""
    device = MockQDMIDevice(num_qubits=3, operations=["cx", "measure", "x"])
    monkeypatch.setattr(device, "supported_program_formats", lambda: [program_format])
    monkeypatch.setattr(device.operations()[0], "site_pairs", lambda: [(device.sites()[2], device.sites()[0])])
    submissions = _record_submissions(device)
    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type] Device boundary double.
    body = QuantumCircuit(2)
    body.cx(0, 1)
    circuit = QuantumCircuit(3, 1)
    circuit.if_else((circuit.clbits[0], True), body, QuantumCircuit(2), [2, 0], [])
    backend.run(circuit, shots=2)
    assert len(submissions) == 1

    invalid = QuantumCircuit(3, 1)
    invalid.if_else((invalid.clbits[0], True), body, QuantumCircuit(2), [0, 2], [])
    with pytest.raises(UnsupportedOperationError, match=r"native device qubits \(0, 2\)"):
        backend.run(invalid, shots=2)
    assert len(submissions) == 1


def test_backend_qasm_serialization_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backend should raise TranslationError when OpenQASM serialization fails."""

    # Exercise the shared compiler import error boundary.
    def failing_dumps(circuit: object, **_kwargs: object) -> NoReturn:  # ruff:ignore[unused-function-argument]
        msg = "Simulated conversion failure"
        raise ValueError(msg)

    monkeypatch.setattr("mqt.core.plugins.qiskit.backend.serialize_program", failing_dumps)

    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    qc.measure_all()

    device = MockQDMIDevice(num_qubits=2, operations=["cz", "measure", "x"])
    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type]

    with pytest.raises(TranslationError, match="Failed to serialize circuit") as error:
        backend.run(qc)
    assert isinstance(error.value.__cause__, ValueError)
    assert str(error.value.__cause__) == "Simulated conversion failure"


def test_backend_submits_qpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """QPY submission preserves the circuit and metadata within the version cap."""
    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    qc.measure_all()
    qc.metadata = {"experiment": "QPY submission"}

    device = MockQDMIDevice(num_qubits=2, operations=["cz", "measure", "x"])
    monkeypatch.setattr(device, "supported_program_formats", lambda: [ProgramFormat.QPY])
    submissions = _record_submissions(device)
    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type]

    job = backend.run(qc, shots=2)
    assert len(submissions) == 1
    payload, fmt = submissions[0]
    assert fmt == ProgramFormat.QPY
    assert isinstance(payload, bytes)
    assert qpy.get_qpy_version(BytesIO(payload)) == 17
    restored = qpy.load(BytesIO(payload))
    assert len(restored) == 1
    assert Operator(cast("QuantumCircuit", restored[0].remove_final_measurements(inplace=False))).equiv(
        Operator(cast("QuantumCircuit", qc.remove_final_measurements(inplace=False)))
    )
    assert restored[0].count_ops()["measure"] == 2
    result = job.result()
    assert result.results is not None
    assert result.results[0].header["metadata"] == qc.metadata


def test_backend_qpy_serialization_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """QPY failures are reported as translation errors without submitting a job."""
    circuit = QuantumCircuit(1)
    circuit.x(0)
    monkeypatch.setattr(qpy, "dump", Mock(side_effect=ValueError("QPY encoding failed")))
    device = MockQDMIDevice(num_qubits=1, operations=["x", "measure"])
    monkeypatch.setattr(device, "supported_program_formats", lambda: [ProgramFormat.QPY])
    submissions = _record_submissions(device)
    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type] Device boundary double.
    with pytest.raises(TranslationError, match="Failed to serialize circuit"):
        backend.run(circuit, shots=2)
    assert not submissions


def test_backend_validation_uses_inverse_mapping() -> None:
    """Qiskit's R gate is accepted for the device spelling PRX."""
    backend = QDMIBackend(MockQDMIDevice(num_qubits=2, operations=["prx", "cz", "measure"]))  # ty: ignore[invalid-argument-type]
    circuit = QuantumCircuit(2)
    circuit.r(1.5708, 0.0, 0)
    circuit.cz(0, 1)
    circuit.measure_all()
    assert backend.run(circuit, shots=100).result().success


class ExecutionOptionsBackend(QDMIBackend):
    """Test device-specific option encoding without vendor dependencies."""

    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.update_options(execution_mode="default")
        return options

    def _job_parameters(self, options: Mapping[str, object]) -> QDMIJobParameters:  # ruff:ignore[no-self-use]
        mode = options["execution_mode"]
        assert options == {"execution_mode": mode}
        if mode not in {"default", "selected"}:
            msg = "Invalid execution_mode"
            raise CircuitValidationError(msg)
        return {"custom1": str(mode)}


def test_execution_options_defaults_overrides_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Encode resolved options for each circuit without mutating defaults."""
    device = MockQDMIDevice(num_qubits=1)
    backend = ExecutionOptionsBackend(device)  # ty: ignore[invalid-argument-type]
    submit = Mock(return_value=device.MockJob(num_clbits=1, shots=3))
    monkeypatch.setattr(device, "submit_job", submit)
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    backend.set_options(execution_mode="selected")
    backend.run([circuit, circuit], shots=3)
    assert submit.call_count == 2
    assert all(call.kwargs["custom1"] == "selected" for call in submit.call_args_list)
    backend.run(circuit, execution_mode="default")
    assert submit.call_args.kwargs["custom1"] == "default"
    assert backend.options.execution_mode == "selected"
    submit.reset_mock()
    with pytest.raises(CircuitValidationError, match="Invalid execution_mode"):
        backend.run([circuit, circuit], execution_mode="invalid")
    with pytest.raises(CircuitValidationError, match="Unsupported execution options: typo"):
        backend.run(circuit, typo=True)
    submit.assert_not_called()


def test_execution_options_forward_exact_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward custom bytes unchanged through the backend submission hook."""
    device = MockQDMIDevice(num_qubits=1)
    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type]
    submit = Mock(return_value=device.MockJob(num_clbits=1, shots=1))
    monkeypatch.setattr(device, "submit_job", submit)
    monkeypatch.setattr(QDMIBackend, "_job_parameters", lambda _self, _options: {"custom1": b"\x00\xff"})
    backend.run(QuantumCircuit(1), shots=1)
    assert submit.call_args.kwargs["custom1"] == b"\x00\xff"


def test_execution_options_survive_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A replacement keeps submitted options despite backend or caller mutations."""
    device = MockQDMIDevice(num_qubits=1)
    backend = ExecutionOptionsBackend(device)  # ty: ignore[invalid-argument-type]
    failed = device.MockJob(num_clbits=1, shots=3)
    failed._status = QDMIJobHandle.Status.FAILED  # ruff:ignore[private-member-access] Simulate a failed job.
    submit = Mock(side_effect=[failed, device.MockJob(num_clbits=1, shots=3)])
    monkeypatch.setattr(device, "submit_job", submit)
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    backend.set_options(execution_mode="selected")
    parameters: QDMIJobParameters = {"custom1": "selected"}
    monkeypatch.setattr(backend, "_job_parameters", lambda _options: parameters)
    job = backend.run(circuit, shots=3, max_retries=1)
    backend.set_options(execution_mode="default")
    parameters["custom1"] = "changed"
    job.result()
    assert [call.kwargs["custom1"] for call in submit.call_args_list] == ["selected", "selected"]


@pytest.mark.parametrize("primitive", ["sampler", "estimator"])
def test_primitives_forward_backend_execution_options(monkeypatch: pytest.MonkeyPatch, primitive: str) -> None:
    """Native primitives preserve backend defaults; sampler accepts run overrides."""
    device = MockQDMIDevice(num_qubits=1)
    backend = ExecutionOptionsBackend(device)  # ty: ignore[invalid-argument-type]
    backend.set_options(execution_mode="selected")
    job = device.MockJob(num_clbits=1, shots=4)
    monkeypatch.setattr(job, "get_shots", lambda _program_index=0: ["0"] * 4)
    submit = Mock(return_value=job)
    monkeypatch.setattr(device, "submit_job", submit)
    circuit = QuantumCircuit(1)
    if primitive == "sampler":
        circuit.measure_all()
        backend.sampler().run([circuit], shots=4).result()
        assert submit.call_args.kwargs["custom1"] == "selected"
        sampler = backend.sampler(run_options={"execution_mode": "default"})
        sampler.run([circuit], shots=4).result()
        expected = "default"
    else:
        backend.estimator().run([(circuit, SparsePauliOp("Z"))], precision=0.5).result()
        expected = "selected"
    assert submit.call_args.kwargs["custom1"] == expected


def test_declared_options_without_encoding_are_rejected() -> None:
    """A subclass must implement encoding for every additional declared option."""
    device = MockQDMIDevice(num_qubits=1)
    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type]
    backend.options.update_options(unmapped=True)
    with pytest.raises(CircuitValidationError, match="Unsupported execution options: unmapped"):
        backend.run(QuantumCircuit(1))


def test_qasm3_rejects_empty_register_before_submission() -> None:
    """The compiler's root-register limits are enforced before creating a job."""
    device = MockQDMIDevice(num_qubits=1, operations=["measure", "x"])
    submissions = _record_submissions(device)
    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type] Device boundary double.
    circuit = QuantumCircuit(1)
    circuit.add_register(ClassicalRegister(0, "empty"))
    with pytest.raises(TranslationError, match="does not support empty classical registers"):
        backend.run(circuit)
    assert not submissions


def test_native_opaque_gate_validation() -> None:
    """An opaque native gate remains usable without a Qiskit transpilation mapping."""
    device = MockQDMIDevice(num_qubits=2, operations=["hop", "measure", "x"])
    backend = QDMIBackend(device)  # ty: ignore[invalid-argument-type] Device boundary double.
    assert "hop" in backend.target.operation_names
    circuit = QuantumCircuit(2)
    circuit.append(Gate("hop", 2, []), [0, 1])
    backend.validate_circuit(circuit, native=True)
