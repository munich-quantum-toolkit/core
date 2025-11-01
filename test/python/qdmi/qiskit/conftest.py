# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared fixtures for QDMI Qiskit backend tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mqt.core import fomac
from mqt.core.qdmi.qiskit import QDMIProvider

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mqt.core.qdmi.qiskit import QiskitBackend


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

        def __init__(self, name: str) -> None:
            """Initialize mock operation with name and infer properties."""
            self._name = name
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

        @staticmethod
        def duration() -> None:
            """Return operation duration (always None for mock)."""
            return

        @staticmethod
        def fidelity() -> None:
            """Return operation fidelity (always None for mock)."""
            return

        @staticmethod
        def sites() -> None:
            """Return specific sites for operation (always None for mock)."""
            return

        @staticmethod
        def is_zoned() -> bool:
            """Return whether operation is zoned (always False for mock)."""
            return False

    class MockJob:
        """Mock FoMaC job with simulated results."""

        def __init__(self, num_clbits: int, shots: int) -> None:
            """Initialize mock job with number of classical bits and shots."""
            import secrets
            import string

            self._num_clbits = num_clbits
            self._shots = shots
            alphabet = string.ascii_lowercase + string.digits
            self._id = "mock-job-" + "".join(secrets.choice(alphabet) for _ in range(8))
            self._status = fomac.JobStatus.DONE
            self._counts: dict[str, int] | None = None

        @property
        def id(self) -> str:
            """Return job ID."""
            return self._id

        @property
        def num_shots(self) -> int:
            """Return number of shots."""
            return self._shots

        def check(self) -> fomac.JobStatus:
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
                import secrets

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
        num_qubits: int = 5,
        operations: Sequence[str] | None = None,
        coupling_map: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        """Initialize a mock QDMI device.

        Args:
            name: Device name.
            num_qubits: Number of qubits.
            operations: List of operation names. Defaults to common gates.
            coupling_map: Coupling map as list of (control, target) pairs. None means all-to-all.
        """
        self._name = name
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

    def qubits_num(self) -> int:
        """Return number of qubits."""
        return self._num_qubits

    def sites(self) -> list[MockSite]:
        """Return list of device sites."""
        return self._sites

    def operations(self) -> list[MockOperation]:
        """Return list of device operations."""
        return self._operations

    def coupling_map(self) -> list[tuple[MockSite, MockSite]] | None:
        """Return device coupling map or None if all-to-all."""
        return self._coupling_map

    def submit_job(self, program: str, program_format: fomac.ProgramFormat, num_shots: int) -> MockJob:  # noqa: ARG002
        """Submit a mock job to the device.

        Args:
            program: The program string to parse for classical bit count.
            program_format: The program format (unused in mock).
            num_shots: Number of shots to simulate.

        Returns:
            A mock job with simulated results.
        """
        # Parse the program to determine the number of classical bits
        import re

        # Look for "creg <name>[<size>];" pattern in QASM2
        match = re.search(r"creg\s+\w+\[(\d+)]", program)
        if match:
            num_clbits = int(match.group(1))
        else:
            # Look for "bit[<size>] <name>;" pattern in QASM3
            match = re.search(r"bit\[(\d+)]", program)
            num_clbits = int(match.group(1)) if match else 2

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
        import re

        # Parse the program to determine the number of classical bits
        match = re.search(r"creg\s+\w+\[(\d+)]", program)
        if match:
            num_clbits = int(match.group(1))
        else:
            match = re.search(r"bit\[(\d+)]", program)
            num_clbits = int(match.group(1)) if match else 2

        return MockQDMIDevice.MockJob(num_clbits=num_clbits, shots=num_shots)


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
def mock_backend(mock_qdmi_device: MockQDMIDevice, monkeypatch: pytest.MonkeyPatch) -> QiskitBackend:
    """Fixture providing a QiskitBackend with a generic mock device.

    Returns:
        QiskitBackend instance configured with a mock device.

    Note:
        This fixture is intended for generic unit tests that don't depend on
        specific device characteristics. The mock device supports common gates
        (h, cz, ry, rz, measure) on 5 qubits.
    """

    # Monkeypatch fomac.devices() to return mock device
    def mock_devices() -> list[MockQDMIDevice]:
        return [mock_qdmi_device]

    monkeypatch.setattr("mqt.core.fomac.devices", mock_devices)

    # Use provider pattern
    provider = QDMIProvider()
    return provider.get_backend("Mock QDMI Device")


@pytest.fixture
def mock_na_device() -> MockQDMIDeviceWrapper:
    """Fixture providing a mock-wrapped NA device for testing.

    Returns:
        MockQDMIDeviceWrapper wrapping the MQT NA Default QDMI Device.

    Note:
        This fixture provides a device that supports mock job execution,
        allowing tests to run without actual hardware or simulator support.
    """
    # Find NA device by name
    devices_list = list(fomac.devices())
    na_device = None
    for device in devices_list:
        if device.name() == "MQT NA Default QDMI Device":
            na_device = device
            break

    if na_device is None:
        pytest.skip("MQT NA Default QDMI Device not found in environment")

    return MockQDMIDeviceWrapper(na_device)
