# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared fixtures for QDMI Qiskit backend tests."""

from __future__ import annotations

import itertools
import re
import secrets
import string
from typing import TYPE_CHECKING

import pytest
from qiskit import QuantumCircuit

from mqt.core import fomac
from mqt.core.plugins.qiskit import QDMIBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def _parse_num_clbits_from_qasm(program: str) -> int:
    """Parse the number of classical bits from a QASM program.

    Args:
        program: The QASM program string (QASM2 or QASM3 format).

    Returns:
        The number of classical bits declared in the program.

    Raises:
        ValueError: If the number of classical bits cannot be determined.
    """
    # Look for "creg <name>[<size>];" pattern in QASM2
    match = re.search(r"creg\s+\w+\[(\d+)]", program)
    if match:
        return int(match.group(1))

    # Look for "bit[<size>] <name>;" pattern in QASM3
    match = re.search(r"bit\[(\d+)]", program)
    if match:
        return int(match.group(1))

    msg = "Could not parse number of classical bits from QASM program."
    raise ValueError(msg)


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
            elif name in {"cz", "cx", "cnot", "cy", "ch", "swap", "iswap"}:
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
            """Return job ID."""
            return self._id

        @property
        def num_shots(self) -> int:
            """Return number of shots."""
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
                    outcome: count for outcome, count in zip(outcomes, counts_list, strict=False) if count > 0
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
        num_clbits = _parse_num_clbits_from_qasm(program)
        return self.MockJob(num_clbits=num_clbits, shots=num_shots)


class MockQDMIDeviceWrapper:
    """Wrapper that adds mock job execution to any FoMaC device.

    This is useful for wrapping real devices (like the NA device) to provide
    mock job execution for testing without actual hardware.
    """

    def __init__(self, device: fomac.Device) -> None:
        """Initialize the wrapper.

        Args:
            device: The real FoMaC device to wrap.
        """
        self._device = device

    def __getattr__(self, name: str) -> object:
        """Delegate attribute access to the wrapped device.

        Returns:
            The attribute value from the wrapped device.
        """
        return getattr(self._device, name)

    def submit_job(  # noqa: PLR6301
        self,
        program: str,
        program_format: fomac.ProgramFormat,  # noqa: ARG002
        num_shots: int,
    ) -> MockQDMIDevice.MockJob:
        """Submit a mock job (overrides the real device's submit_job).

        Args:
            program: The program string to parse for classical bit count.
            program_format: The program format (unused in mock).
            num_shots: Number of shots to simulate.

        Returns:
            A mock job with simulated results.
        """
        num_clbits = _parse_num_clbits_from_qasm(program)
        return MockQDMIDevice.MockJob(num_clbits=num_clbits, shots=num_shots)


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


@pytest.fixture
def mock_qdmi_device() -> MockQDMIDevice:
    """Fixture providing a generic mock QDMI device for unit tests.

    Returns:
        Mock device with 5 qubits and common gates (h, cz, ry, rz, measure).

    Note:
        This fixture is intended for generic unit tests that don't depend on
        specific device characteristics.
    """
    return MockQDMIDevice()


@pytest.fixture
def mock_backend(mock_qdmi_device: MockQDMIDevice) -> QDMIBackend:
    """Fixture providing a QiskitBackend with a generic mock device.

    Returns:
        QDMIBackend instance configured with a mock device.

    Note:
        This fixture is intended for generic unit tests that don't depend on
        specific device characteristics. The mock device supports common gates
        (h, cz, ry, rz, measure) on 5 qubits.
    """
    return QDMIBackend(device=mock_qdmi_device)  # type: ignore[arg-type]


def _single_qubit_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(1, 1)
    qc.ry(1.5708, 0)
    qc.measure_all()
    return qc


def _two_qubit_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.cz(0, 1)
    qc.measure_all()
    return qc


def _two_qubit_barrier_circuit() -> QuantumCircuit:
    qc = _two_qubit_circuit()
    qc.barrier()
    return qc


def _two_qubit_circuit_without_measurements() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.cz(0, 1)
    return qc


CircuitCase = tuple[QuantumCircuit, int, bool]


@pytest.fixture(
    params=[
        (_single_qubit_circuit, 100, True),
        (_two_qubit_circuit, 256, True),
        (_two_qubit_circuit, 500, True),
        (_two_qubit_barrier_circuit, 100, True),
        (_two_qubit_circuit_without_measurements, 100, False),
    ]
)
def backend_circuit_case(request: pytest.FixtureRequest) -> CircuitCase:
    """Provide reusable backend circuit cases.

    Returns:
        Tuple containing a prepared circuit, the shots to execute, and whether counts are expected.
    """
    builder, shots, expect_meas = request.param
    return builder(), shots, expect_meas


def _build_site_specific_operation(
    mock_cls: type[MockQDMIDevice],
    name: str,
    num_qubits: int,
    sites: list[MockQDMIDevice.MockSite],
) -> MockQDMIDevice.MockOperation:
    if num_qubits == 1:

        def single_duration(target_sites: list[MockQDMIDevice.MockSite]) -> float:
            site = target_sites[0]
            return 100.0 + site.index() * 10.0

        def single_fidelity(target_sites: list[MockQDMIDevice.MockSite]) -> float:
            site = target_sites[0]
            return 0.99 - site.index() * 0.01

        return mock_cls.MockOperation(
            name,
            custom_duration=single_duration,
            custom_fidelity=single_fidelity,
            custom_sites=sites,
        )

    def pair_duration(pair: list[MockQDMIDevice.MockSite]) -> float:
        left, right = pair
        return 200.0 + left.index() * 10.0 + right.index() * 5.0

    def pair_fidelity(pair: list[MockQDMIDevice.MockSite]) -> float:
        left, right = pair
        return 0.98 - (left.index() + right.index()) * 0.005

    site_pairs = list(itertools.pairwise(sites))
    return mock_cls.MockOperation(
        name,
        custom_duration=pair_duration,
        custom_fidelity=pair_fidelity,
        custom_site_pairs=site_pairs,
    )


@pytest.fixture
def site_specific_device(mock_qdmi_device_factory: type[MockQDMIDevice]) -> MockQDMIDevice:
    """Device with operations carrying site-specific duration/fidelity metadata.

    Returns:
        Mock device exposing site-specific `h` and `cz` operations.
    """
    mock_cls = mock_qdmi_device_factory
    sites = [mock_cls.MockSite(i) for i in range(3)]
    operations = [
        _build_site_specific_operation(mock_cls, "h", 1, sites),
        _build_site_specific_operation(mock_cls, "cz", 2, sites),
        mock_cls.MockOperation("measure"),
    ]
    device = mock_cls(name="Site-Specific Device", num_qubits=3)
    device._operations = operations  # noqa: SLF001
    device._sites = sites  # noqa: SLF001
    return device


@pytest.fixture
def misconfigured_coupling_device(mock_qdmi_device_factory: type[MockQDMIDevice]) -> MockQDMIDevice:
    """Device declaring a coupling map but no site pairs for operations.

    Returns:
        Mock device lacking site-pair metadata for two-qubit ops.
    """
    mock_cls = mock_qdmi_device_factory
    return mock_cls(name="Misconfigured Device", num_qubits=5, operations=[], coupling_map=[(0, 1), (1, 2)])


@pytest.fixture
def zoned_operation_device(mock_qdmi_device_factory: type[MockQDMIDevice]) -> MockQDMIDevice:
    """Device exposing an operation marked as zoned.

    Returns:
        Mock device containing a zoned-only operation for validation tests.
    """
    mock_cls = mock_qdmi_device_factory
    zoned_op = mock_cls.MockOperation("zoned_op", zoned=True)
    device = mock_cls(name="Zoned Device", num_qubits=5)
    device._operations = [zoned_op]  # noqa: SLF001
    return device
