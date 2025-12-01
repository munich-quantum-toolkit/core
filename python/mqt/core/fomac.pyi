# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from collections.abc import Iterable
from enum import Enum

__all__ = [
    "Device",
    "Job",
    "ProgramFormat",
    "devices",
]

class ProgramFormat(Enum):
    """Enumeration of program formats."""

    QASM2 = ...
    QASM3 = ...
    QIR_BASE_STRING = ...
    QIR_BASE_MODULE = ...
    QIR_ADAPTIVE_STRING = ...
    QIR_ADAPTIVE_MODULE = ...
    CALIBRATION = ...
    QPY = ...
    IQM_JSON = ...
    CUSTOM1 = ...
    CUSTOM2 = ...
    CUSTOM3 = ...
    CUSTOM4 = ...
    CUSTOM5 = ...

class Job:
    """A job represents a submitted quantum program execution."""

    class Status(Enum):
        """Enumeration of job status."""

        CREATED = ...
        SUBMITTED = ...
        QUEUED = ...
        RUNNING = ...
        DONE = ...
        CANCELED = ...
        FAILED = ...

    def check(self) -> Job.Status:
        """Returns the current status of the job."""
    def wait(self, timeout: int = 0) -> bool:
        """Waits for the job to complete.

        Args:
            timeout: The maximum time to wait in seconds. If 0, waits indefinitely.

        Returns:
            True if the job completed within the timeout, False otherwise.
        """
    def cancel(self) -> None:
        """Cancels the job."""
    def get_shots(self) -> list[str]:
        """Returns the raw shot results from the job."""
    def get_counts(self) -> dict[str, int]:
        """Returns the measurement counts from the job."""
    def get_dense_statevector(self) -> list[complex]:
        """Returns the dense statevector from the job (typically only available from simulator devices)."""
    def get_sparse_statevector(self) -> dict[str, complex]:
        """Returns the sparse statevector from the job (typically only available from simulator devices)."""
    def get_dense_probabilities(self) -> list[float]:
        """Returns the dense probabilities from the job (typically only available from simulator devices)."""
    def get_sparse_probabilities(self) -> dict[str, float]:
        """Returns the sparse probabilities from the job (typically only available from simulator devices)."""
    def __eq__(self, other: object) -> bool:
        """Checks if two jobs are equal."""
    def __ne__(self, other: object) -> bool:
        """Checks if two jobs are not equal."""
    @property
    def id(self) -> str:
        """Returns the job ID."""
    @property
    def program_format(self) -> ProgramFormat:
        """Returns the program format used for the job."""
    @property
    def program(self) -> str:
        """Returns the quantum program submitted for the job."""
    @property
    def num_shots(self) -> int:
        """Returns the number of shots for the job."""

class Device:
    """A device represents a quantum device with its properties and capabilities."""
    class Status(Enum):
        offline = ...
        idle = ...
        busy = ...
        error = ...
        maintenance = ...
        calibration = ...

    class Site:
        """A site represents a potential qubit location on a quantum device."""
        def index(self) -> int:
            """Returns the index of the site."""
        def t1(self) -> int | None:
            """Returns the T1 coherence time of the site."""
        def t2(self) -> int | None:
            """Returns the T2 coherence time of the site."""
        def name(self) -> str | None:
            """Returns the name of the site."""
        def x_coordinate(self) -> int | None:
            """Returns the x coordinate of the site."""
        def y_coordinate(self) -> int | None:
            """Returns the y coordinate of the site."""
        def z_coordinate(self) -> int | None:
            """Returns the z coordinate of the site."""
        def is_zone(self) -> bool:
            """Returns whether the site is a zone."""
        def x_extent(self) -> int | None:
            """Returns the x extent of the site."""
        def y_extent(self) -> int | None:
            """Returns the y extent of the site."""
        def z_extent(self) -> int | None:
            """Returns the z extent of the site."""
        def module_index(self) -> int | None:
            """Returns the index of the module the site belongs to."""
        def submodule_index(self) -> int | None:
            """Returns the index of the submodule the site belongs to."""
        def __eq__(self, other: object) -> bool:
            """Checks if two sites are equal."""
        def __ne__(self, other: object) -> bool:
            """Checks if two sites are not equal."""

    class Operation:
        """An operation represents a quantum operation that can be performed on a quantum device."""
        def name(self, sites: Iterable[Device.Site] = ..., params: Iterable[float] = ...) -> str:
            """Returns the name of the operation."""
        def qubits_num(self, sites: Iterable[Device.Site] = ..., params: Iterable[float] = ...) -> int | None:
            """Returns the number of qubits the operation acts on."""
        def parameters_num(self, sites: Iterable[Device.Site] = ..., params: Iterable[float] = ...) -> int:
            """Returns the number of parameters the operation has."""
        def duration(self, sites: Iterable[Device.Site] = ..., params: Iterable[float] = ...) -> int | None:
            """Returns the duration of the operation."""
        def fidelity(self, sites: Iterable[Device.Site] = ..., params: Iterable[float] = ...) -> float | None:
            """Returns the fidelity of the operation."""
        def interaction_radius(self, sites: Iterable[Device.Site] = ..., params: Iterable[float] = ...) -> int | None:
            """Returns the interaction radius of the operation."""
        def blocking_radius(self, sites: Iterable[Device.Site] = ..., params: Iterable[float] = ...) -> int | None:
            """Returns the blocking radius of the operation."""
        def idling_fidelity(self, sites: Iterable[Device.Site] = ..., params: Iterable[float] = ...) -> float | None:
            """Returns the idling fidelity of the operation."""
        def is_zoned(self) -> bool:
            """Returns whether the operation is zoned."""
        def sites(self) -> list[Device.Site] | None:
            """Returns the list of sites the operation can be performed on."""
        def site_pairs(self) -> list[tuple[Device.Site, Device.Site]] | None:
            """Returns the list of site pairs the local 2-qubit operation can be performed on."""
        def mean_shuttling_speed(self, sites: Iterable[Device.Site] = ..., params: Iterable[float] = ...) -> int | None:
            """Returns the mean shuttling speed of the operation."""
        def __eq__(self, other: object) -> bool:
            """Checks if two operations are equal."""
        def __ne__(self, other: object) -> bool:
            """Checks if two operations are not equal."""

    def name(self) -> str:
        """Returns the name of the device."""
    def version(self) -> str:
        """Returns the version of the device."""
    def status(self) -> Device.Status:
        """Returns the current status of the device."""
    def library_version(self) -> str:
        """Returns the version of the library used to define the device."""
    def qubits_num(self) -> int:
        """Returns the number of qubits available on the device."""
    def sites(self) -> list[Site]:
        """Returns the list of all sites (zone and regular sites) available on the device."""
    def regular_sites(self) -> list[Site]:
        """Returns the list of regular sites (without zone sites) available on the device."""
    def zones(self) -> list[Site]:
        """Returns the list of zone sites (without regular sites) available on the device."""
    def operations(self) -> list[Operation]:
        """Returns the list of operations supported by the device."""
    def coupling_map(self) -> list[tuple[Site, Site]] | None:
        """Returns the coupling map of the device as a list of site pairs."""
    def needs_calibration(self) -> int | None:
        """Returns whether the device needs calibration."""
    def length_unit(self) -> str | None:
        """Returns the unit of length used by the device."""
    def length_scale_factor(self) -> float | None:
        """Returns the scale factor for length used by the device."""
    def duration_unit(self) -> str | None:
        """Returns the unit of duration used by the device."""
    def duration_scale_factor(self) -> float | None:
        """Returns the scale factor for duration used by the device."""
    def min_atom_distance(self) -> int | None:
        """Returns the minimum atom distance on the device."""
    def supported_program_formats(self) -> list[ProgramFormat]:
        """Returns the list of program formats supported by the device."""
    def submit_job(self, program: str, program_format: ProgramFormat, num_shots: int) -> Job:
        """Submits a job to the device."""
    def __eq__(self, other: object) -> bool:
        """Checks if two devices are equal."""
    def __ne__(self, other: object) -> bool:
        """Checks if two devices are not equal."""

def devices() -> list[Device]:
    """Returns a list of available devices."""
