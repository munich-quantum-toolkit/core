# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

import enum
from collections.abc import Sequence

class Session:
    """A FoMaC session for managing QDMI devices.

    Allows creating isolated sessions with independent authentication settings.
    All authentication parameters are optional and can be provided as keyword arguments to the constructor.
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        auth_file: str | None = None,
        auth_url: str | None = None,
        username: str | None = None,
        password: str | None = None,
        project_id: str | None = None,
        custom1: str | None = None,
        custom2: str | None = None,
        custom3: str | None = None,
        custom4: str | None = None,
        custom5: str | None = None,
    ) -> None:
        """Create a new FoMaC session with optional authentication.

        Args:
            token: Authentication token
            auth_file: Path to file containing authentication information
            auth_url: URL to authentication server
            username: Username for authentication
            password: Password for authentication
            project_id: Project ID for session
            custom1: Custom configuration parameter 1
            custom2: Custom configuration parameter 2
            custom3: Custom configuration parameter 3
            custom4: Custom configuration parameter 4
            custom5: Custom configuration parameter 5

        Raises:
            RuntimeError: If auth_file does not exist
            RuntimeError: If auth_url has invalid format

        Example:
            >>> from mqt.core.fomac import Session
            >>> # Session without authentication
            >>> session = Session()
            >>> devices = session.get_devices()
            >>>
            >>> # Session with token authentication
            >>> session = Session(token="my_secret_token")
            >>> devices = session.get_devices()
            >>>
            >>> # Session with file-based authentication
            >>> session = Session(auth_file="/path/to/auth.json")
            >>> devices = session.get_devices()
            >>>
            >>> # Session with multiple parameters
            >>> session = Session(
            ...     auth_url="https://auth.example.com", username="user", password="pass", project_id="project-123"
            ... )
            >>> devices = session.get_devices()
        """

    def get_devices(self) -> list[Device]:
        """Get available devices from this session.

        Returns:
            List of available devices.
        """

class Job:
    """A job represents a submitted quantum program execution."""

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

    def get_dense_probabilities(self) -> list[float]:
        """Returns the dense probabilities from the job (typically only available from simulator devices)."""

    def get_sparse_statevector(self) -> dict[str, complex]:
        """Returns the sparse statevector from the job (typically only available from simulator devices)."""

    def get_sparse_probabilities(self) -> dict[str, float]:
        """Returns the sparse probabilities from the job (typically only available from simulator devices)."""

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

    def __eq__(self, arg: object, /) -> bool: ...
    def __ne__(self, arg: object, /) -> bool: ...

    class Status(enum.Enum):
        """Enumeration of job status."""

        CREATED = 0

        SUBMITTED = 1

        QUEUED = 2

        RUNNING = 3

        DONE = 4

        CANCELED = 5

        FAILED = 6

class ProgramFormat(enum.Enum):
    """Enumeration of program formats."""

    QASM2 = 0

    QASM3 = 1

    QIR_BASE_STRING = 2

    QIR_BASE_MODULE = 3

    QIR_ADAPTIVE_STRING = 4

    QIR_ADAPTIVE_MODULE = 5

    CALIBRATION = 6

    QPY = 7

    IQM_JSON = 8

    CUSTOM1 = 999999995

    CUSTOM2 = 999999996

    CUSTOM3 = 999999997

    CUSTOM4 = 999999998

    CUSTOM5 = 999999999

class Device:
    """A device represents a quantum device with its properties and capabilities."""

    class Status(enum.Enum):
        """Enumeration of device status."""

        OFFLINE = 0

        IDLE = 1

        BUSY = 2

        ERROR = 3

        MAINTENANCE = 4

        CALIBRATION = 5

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

    def sites(self) -> list[Device.Site]:
        """Returns the list of all sites (zone and regular sites) available on the device."""

    def regular_sites(self) -> list[Device.Site]:
        """Returns the list of regular sites (without zone sites) available on the device."""

    def zones(self) -> list[Device.Site]:
        """Returns the list of zone sites (without regular sites) available on the device."""

    def operations(self) -> list[Device.Operation]:
        """Returns the list of operations supported by the device."""

    def coupling_map(self) -> list[tuple[Device.Site, Device.Site]] | None:
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

    def __eq__(self, arg: object, /) -> bool: ...
    def __ne__(self, arg: object, /) -> bool: ...

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

        def __eq__(self, arg: object, /) -> bool: ...
        def __ne__(self, arg: object, /) -> bool: ...

    class Operation:
        """An operation represents a quantum operation that can be performed on a quantum device."""

        def name(self, sites: Sequence[Device.Site] = ..., params: Sequence[float] = ...) -> str:
            """Returns the name of the operation."""

        def qubits_num(self, sites: Sequence[Device.Site] = ..., params: Sequence[float] = ...) -> int | None:
            """Returns the number of qubits the operation acts on."""

        def parameters_num(self, sites: Sequence[Device.Site] = ..., params: Sequence[float] = ...) -> int:
            """Returns the number of parameters the operation has."""

        def duration(self, sites: Sequence[Device.Site] = ..., params: Sequence[float] = ...) -> int | None:
            """Returns the duration of the operation."""

        def fidelity(self, sites: Sequence[Device.Site] = ..., params: Sequence[float] = ...) -> float | None:
            """Returns the fidelity of the operation."""

        def interaction_radius(self, sites: Sequence[Device.Site] = ..., params: Sequence[float] = ...) -> int | None:
            """Returns the interaction radius of the operation."""

        def blocking_radius(self, sites: Sequence[Device.Site] = ..., params: Sequence[float] = ...) -> int | None:
            """Returns the blocking radius of the operation."""

        def idling_fidelity(self, sites: Sequence[Device.Site] = ..., params: Sequence[float] = ...) -> float | None:
            """Returns the idling fidelity of the operation."""

        def is_zoned(self) -> bool:
            """Returns whether the operation is zoned."""

        def sites(self) -> list[Device.Site] | None:
            """Returns the list of sites the operation can be performed on."""

        def site_pairs(self) -> list[tuple[Device.Site, Device.Site]] | None:
            """Returns the list of site pairs the local 2-qubit operation can be performed on."""

        def mean_shuttling_speed(self, sites: Sequence[Device.Site] = ..., params: Sequence[float] = ...) -> int | None:
            """Returns the mean shuttling speed of the operation."""

        def __eq__(self, arg: object, /) -> bool: ...
        def __ne__(self, arg: object, /) -> bool: ...

def add_dynamic_device_library(
    library_path: str,
    prefix: str,
    *,
    base_url: str | None = None,
    token: str | None = None,
    auth_file: str | None = None,
    auth_url: str | None = None,
    username: str | None = None,
    password: str | None = None,
    custom1: str | None = None,
    custom2: str | None = None,
    custom3: str | None = None,
    custom4: str | None = None,
    custom5: str | None = None,
) -> Device:
    """Load a dynamic device library into the QDMI driver.

    This function loads a shared library (.so, .dll, or .dylib) that implements a QDMI device interface and makes it available for use in sessions.

    Args:
        library_path: Path to the shared library file to load.
        prefix: Function prefix used by the library (e.g., "MY_DEVICE").
        base_url: Optional base URL for the device API endpoint.
        token: Optional authentication token.
        auth_file: Optional path to authentication file.
        auth_url: Optional authentication server URL.
        username: Optional username for authentication.
        password: Optional password for authentication.
        custom1: Optional custom configuration parameter 1.
        custom2: Optional custom configuration parameter 2.
        custom3: Optional custom configuration parameter 3.
        custom4: Optional custom configuration parameter 4.
        custom5: Optional custom configuration parameter 5.

    Returns:
        Device: The newly loaded device that can be used to create backends.

    Raises:
        RuntimeError: If library loading fails or configuration is invalid.

    Examples:
        Load a device library with configuration:

        >>> import mqt.core.fomac as fomac
        >>> device = fomac.add_dynamic_device_library(
        ...     "/path/to/libmy_device.so", "MY_DEVICE", base_url="http://localhost:8080", custom1="API_V2"
        ... )

        Now the device can be used directly:

        >>> from mqt.core.plugins.qiskit import QDMIBackend
        >>> backend = QDMIBackend(device=device)
    """
