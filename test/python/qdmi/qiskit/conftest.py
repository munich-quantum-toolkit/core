# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared fixtures for QDMI Qiskit backend tests."""

from __future__ import annotations

from typing import NoReturn

import pytest

from mqt.core import fomac
from mqt.core.qdmi.qiskit import QiskitBackend


def _get_na_device_index() -> int:
    """Find the index of the MQT NA Default QDMI Device.

    Returns:
        Index of the NA device in fomac.devices().
    """
    devices_list = list(fomac.devices())
    for idx, device in enumerate(devices_list):
        if device.name() == "MQT NA Default QDMI Device":
            return idx
    return _skip_na_device_not_found()


def _skip_na_device_not_found() -> NoReturn:
    """Skip the test when NA device is not found."""
    pytest.skip("MQT NA Default QDMI Device not found in environment")


@pytest.fixture
def na_backend(mock_na_device: MockFoMaCDevice, monkeypatch: pytest.MonkeyPatch) -> QiskitBackend:
    """Fixture providing a QiskitBackend configured with the NA device.

    Returns:
        QiskitBackend instance configured with the MQT NA Default QDMI Device (mocked).

    Note:
        This fixture is used for tests that rely on specific NA device characteristics.
        These tests verify site handling, zone filtering, and QDMI specification compliance
        specific to the NA device structure (100 qubits + 3 zone sites).
        The device's job execution is mocked to allow testing without actual hardware.
        In the future, these tests should be generalized or parameterized across device types.
    """

    # Monkeypatch fomac.devices() to return a mock device
    def mock_devices() -> list[MockFoMaCDevice]:
        return [mock_na_device]

    monkeypatch.setattr("mqt.core.fomac.devices", mock_devices)

    # Create backend with device_index=0
    return QiskitBackend(device_index=0)


@pytest.fixture
def na_device() -> fomac.Device:
    """Fixture providing the NA FoMaC device for direct inspection.

    Returns:
        The MQT NA Default QDMI Device.

    Note:
        This fixture is used for tests that need direct access to the FoMaC device
        to inspect raw device capabilities, site information, and other low-level
        properties specific to the NA device structure (100 qubits + 3 zone sites).
        In the future, these tests should be generalized or parameterized across device types.
    """
    return list(fomac.devices())[_get_na_device_index()]


class MockFoMaCDevice:
    """Mock wrapper around a FoMaC device that provides mock job execution.

    This class wraps a real FoMaC device and delegates all methods except submit_job,
    which returns a mock job with simulated results.
    """

    def __init__(self, device: fomac.Device) -> None:
        """Initialize the mock device wrapper.

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

    def submit_job(self, program: str, program_format: fomac.ProgramFormat, num_shots: int) -> MockFoMaCJob:  # noqa: ARG002
        """Submit a job (mocked) to the device.

        Args:
            program: The program string to parse for classical bit count.
            program_format: The program format (unused in mock).
            num_shots: Number of shots to simulate.

        Returns:
            A mock job with simulated results.
        """
        # Parse the program to determine the number of classical bits
        # For QASM programs, look for the classical register declaration
        num_clbits = self._parse_num_clbits_from_program(program)
        return MockFoMaCJob(num_clbits=num_clbits, shots=num_shots)

    @staticmethod
    def _parse_num_clbits_from_program(program: str) -> int:
        """Parse the number of classical bits from a QASM program.

        Args:
            program: The QASM program string.

        Returns:
            Number of classical bits declared in the program.
        """
        import re

        # Look for "creg <name>[<size>];" pattern in QASM2
        match = re.search(r"creg\s+\w+\[(\d+)]", program)
        if match:
            return int(match.group(1))

        # Look for "bit[<size>] <name>;" pattern in QASM3
        match = re.search(r"bit\[(\d+)]", program)
        if match:
            return int(match.group(1))

        # Default to a reasonable number if not found
        return 2


class MockFoMaCJob:
    """Mock implementation of a FoMaC Job for testing purposes.

    This class simulates a FoMaC job with random measurement results,
    allowing tests to run against devices that don't support actual job execution.
    """

    def __init__(self, num_clbits: int, shots: int) -> None:
        """Initialize a mock FoMaC job.

        Args:
            num_clbits: Number of classical bits for measurements.
            shots: Number of shots to simulate.
        """
        import secrets
        import string

        self._num_clbits = num_clbits
        self._shots = shots
        # Use secrets for cryptographically secure random ID
        alphabet = string.ascii_lowercase + string.digits
        self._id = "mock-job-" + "".join(secrets.choice(alphabet) for _ in range(8))
        self._status = fomac.JobStatus.DONE
        self._counts: dict[str, int] | None = None

    @property
    def id(self) -> str:
        """Get the job ID.

        Returns:
            The unique job identifier.
        """
        return self._id

    @property
    def num_shots(self) -> int:
        """Get the number of shots.

        Returns:
            The number of shots for this job.
        """
        return self._shots

    def check(self) -> fomac.JobStatus:
        """Get the job status (always DONE for mock jobs).

        Returns:
            The job status.
        """
        return self._status

    def wait(self) -> None:
        """Wait for the job to complete (no-op for mock jobs)."""

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
            self._counts = {outcome: count for outcome, count in zip(outcomes, counts_list, strict=False) if count > 0}

        return self._counts

    def cancel(self) -> None:
        """Cancel the job (no-op for mock jobs)."""


@pytest.fixture
def mock_na_device() -> MockFoMaCDevice:
    """Fixture providing a mock-wrapped NA device for testing.

    Returns:
        MockFoMaCDevice wrapping the MQT NA Default QDMI Device.

    Note:
        This fixture provides a device that supports mock job execution,
        allowing tests to run without actual hardware or simulator support.
    """
    real_device = list(fomac.devices())[_get_na_device_index()]
    return MockFoMaCDevice(real_device)
