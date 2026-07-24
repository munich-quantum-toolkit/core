# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

import enum
import os
from collections.abc import Sequence
from typing import overload

class Session:
    """A FoMaC session for managing QDMI devices.

    Allows creating isolated sessions with separate authentication settings.
    All authentication parameters are optional and can be provided as keyword arguments to the constructor.
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        auth_file: str | os.PathLike | None = None,
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

    def check(self) -> Status:
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

    @overload
    def query_custom_property(self, custom_property: CustomProperty, value_type: type[str]) -> str | None: ...
    @overload
    def query_custom_property(self, custom_property: CustomProperty, value_type: type[bool]) -> bool | None: ...
    @overload
    def query_custom_property(self, custom_property: CustomProperty, value_type: type[int]) -> int | None: ...
    @overload
    def query_custom_property(self, custom_property: CustomProperty, value_type: type[float]) -> float | None: ...
    @overload
    def query_custom_property(self, custom_property: CustomProperty, value_type: type[bytes]) -> bytes | None: ...
    @overload
    def query_custom_property(
        self, custom_property: CustomProperty, value_type: type[str | bool | int | float | bytes]
    ) -> str | bool | int | float | bytes | None:
        """Query an implementation-defined custom job property.

        The caller must provide the type documented by the device implementation.
        Use ``bytes`` to retrieve the value without interpretation. Returns ``None``
        when the custom slot is unsupported.
        """

    @overload
    def get_custom_result(self, custom_property: CustomProperty, value_type: type[str]) -> str | None: ...
    @overload
    def get_custom_result(self, custom_property: CustomProperty, value_type: type[bool]) -> bool | None: ...
    @overload
    def get_custom_result(self, custom_property: CustomProperty, value_type: type[int]) -> int | None: ...
    @overload
    def get_custom_result(self, custom_property: CustomProperty, value_type: type[float]) -> float | None: ...
    @overload
    def get_custom_result(self, custom_property: CustomProperty, value_type: type[bytes]) -> bytes | None: ...
    @overload
    def get_custom_result(
        self, custom_property: CustomProperty, value_type: type[str | bool | int | float | bytes]
    ) -> str | bool | int | float | bytes | None:
        """Return an implementation-defined custom job result.

        The caller must provide the type documented by the device implementation.
        Use ``bytes`` to retrieve the value without interpretation. Returns ``None``
        when the custom slot is unsupported.
        """

    @property
    def id(self) -> str:
        """The job ID."""

    @property
    def program_format(self) -> ProgramFormat:
        """The format of the submitted program."""

    @property
    def program(self) -> str:
        """The submitted program."""

    @property
    def num_shots(self) -> int:
        """The number of shots."""

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

class CustomProperty(enum.Enum):
    """An implementation-defined custom property or result slot."""

    CUSTOM1 = 1

    CUSTOM2 = 2

    CUSTOM3 = 3

    CUSTOM4 = 4

    CUSTOM5 = 5

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

    def status(self) -> Status:
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

    def child_devices(self) -> list[Device]:
        """Returns the direct child devices managed by this device."""

    @overload
    def query_custom_property(self, custom_property: CustomProperty, value_type: type[str]) -> str | None: ...
    @overload
    def query_custom_property(self, custom_property: CustomProperty, value_type: type[bool]) -> bool | None: ...
    @overload
    def query_custom_property(self, custom_property: CustomProperty, value_type: type[int]) -> int | None: ...
    @overload
    def query_custom_property(self, custom_property: CustomProperty, value_type: type[float]) -> float | None: ...
    @overload
    def query_custom_property(self, custom_property: CustomProperty, value_type: type[bytes]) -> bytes | None: ...
    @overload
    def query_custom_property(
        self, custom_property: CustomProperty, value_type: type[str | bool | int | float | bytes]
    ) -> str | bool | int | float | bytes | None:
        """Query an implementation-defined custom device property.

        The caller must provide the type documented by the device implementation.
        Use ``bytes`` to retrieve the value without interpretation. Returns ``None``
        when the custom slot is unsupported.
        """

    def submit_job(
        self,
        program: str,
        program_format: ProgramFormat,
        num_shots: int,
        *,
        custom1: str | bool | float | None = None,
        custom2: str | bool | float | None = None,
        custom3: str | bool | float | None = None,
        custom4: str | bool | float | None = None,
        custom5: str | bool | float | None = None,
    ) -> Job:
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

        @overload
        def query_custom_property(self, custom_property: CustomProperty, value_type: type[str]) -> str | None: ...
        @overload
        def query_custom_property(self, custom_property: CustomProperty, value_type: type[bool]) -> bool | None: ...
        @overload
        def query_custom_property(self, custom_property: CustomProperty, value_type: type[int]) -> int | None: ...
        @overload
        def query_custom_property(self, custom_property: CustomProperty, value_type: type[float]) -> float | None: ...
        @overload
        def query_custom_property(self, custom_property: CustomProperty, value_type: type[bytes]) -> bytes | None: ...
        @overload
        def query_custom_property(
            self, custom_property: CustomProperty, value_type: type[str | bool | int | float | bytes]
        ) -> str | bool | int | float | bytes | None:
            """Query an implementation-defined custom site property.

            The caller must provide the type documented by the device implementation.
            Use ``bytes`` to retrieve the value without interpretation. Returns ``None``
            when the custom slot is unsupported.
            """

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

        @overload
        def query_custom_property(
            self,
            custom_property: CustomProperty,
            value_type: type[str],
            sites: Sequence[Device.Site] = ...,
            params: Sequence[float] = ...,
        ) -> str | None: ...
        @overload
        def query_custom_property(
            self,
            custom_property: CustomProperty,
            value_type: type[bool],
            sites: Sequence[Device.Site] = ...,
            params: Sequence[float] = ...,
        ) -> bool | None: ...
        @overload
        def query_custom_property(
            self,
            custom_property: CustomProperty,
            value_type: type[int],
            sites: Sequence[Device.Site] = ...,
            params: Sequence[float] = ...,
        ) -> int | None: ...
        @overload
        def query_custom_property(
            self,
            custom_property: CustomProperty,
            value_type: type[float],
            sites: Sequence[Device.Site] = ...,
            params: Sequence[float] = ...,
        ) -> float | None: ...
        @overload
        def query_custom_property(
            self,
            custom_property: CustomProperty,
            value_type: type[bytes],
            sites: Sequence[Device.Site] = ...,
            params: Sequence[float] = ...,
        ) -> bytes | None: ...
        @overload
        def query_custom_property(
            self,
            custom_property: CustomProperty,
            value_type: type[str | bool | int | float | bytes],
            sites: Sequence[Device.Site] = ...,
            params: Sequence[float] = ...,
        ) -> str | bool | int | float | bytes | None:
            """Query an implementation-defined custom operation property.

            The caller must provide the type documented by the device implementation.
            Use ``bytes`` to retrieve the value without interpretation. Returns ``None``
            when the custom slot is unsupported.
            """

        def __eq__(self, arg: object, /) -> bool: ...
        def __ne__(self, arg: object, /) -> bool: ...

class DeviceDefinition:
    """A stable QDMI device registration that can be stored before loading."""

    def __init__(
        self,
        device_id: str,
        library_path: str,
        prefix: str,
        *,
        base_url: str | None = None,
        token: str | None = None,
        auth_file: str | os.PathLike | None = None,
        auth_url: str | None = None,
        username: str | None = None,
        password: str | None = None,
        custom1: str | None = None,
        custom2: str | None = None,
        custom3: str | None = None,
        custom4: str | None = None,
        custom5: str | None = None,
    ) -> None:
        """Create a device definition without loading its native library.

        Args:
            device_id: Stable identifier used by :func:`open_device`.
            library_path: Path to the shared QDMI device library.
            prefix: Function prefix used by the library (for example, ``MY_DEVICE``).
            base_url: Optional base URL for the device API endpoint.
            token: Optional authentication token.
            auth_file: Optional path to an authentication file.
            auth_url: Optional authentication server URL.
            username: Optional authentication username.
            password: Optional authentication password.
            custom1: Optional custom configuration parameter 1.
            custom2: Optional custom configuration parameter 2.
            custom3: Optional custom configuration parameter 3.
            custom4: Optional custom configuration parameter 4.
            custom5: Optional custom configuration parameter 5.
        """

    @property
    def device_id(self) -> str:
        """Stable identifier used to open the device."""

    @property
    def library_path(self) -> str:
        """Path to the native QDMI device library."""

    @property
    def prefix(self) -> str:
        """Prefix used for the QDMI device interface functions."""

def register_device(definition: DeviceDefinition, *, replace: bool = False) -> None:
    """Register a QDMI device definition without loading its library.

    Args:
        definition: Definition to validate and store.
        replace: Replace an existing definition if it has not been opened.

    Raises:
        ValueError: If the definition is invalid or its ID is already registered.
        RuntimeError: If replacing an already opened ID.
    """

def register_device_if_absent(definition: DeviceDefinition) -> bool:
    """Register a valid QDMI device definition if its ID is absent.

    An existing ID is the only ignored condition. Invalid definitions still raise.

    Args:
        definition: Definition to validate and store.

    Returns:
        bool: Whether the definition was inserted.

    Raises:
        ValueError: If the definition is invalid.
    """

def open_device(
    device_id: str,
    *,
    base_url: str | None = None,
    token: str | None = None,
    auth_file: str | os.PathLike | None = None,
    auth_url: str | None = None,
    username: str | None = None,
    password: str | None = None,
    custom1: str | None = None,
    custom2: str | None = None,
    custom3: str | None = None,
    custom4: str | None = None,
    custom5: str | None = None,
) -> Device:
    """Open a registered QDMI device by stable ID.

    Every call creates a fresh device session while keeping the stable registration
    unchanged. Opening the device loads trusted native device code.

    Args:
        device_id: Stable ID of a registered device.
        base_url: Optional base URL override for the device API endpoint.
        token: Optional authentication token override.
        auth_file: Optional authentication-file override.
        auth_url: Optional authentication server URL override.
        username: Optional authentication username override.
        password: Optional authentication password override.
        custom1: Optional custom configuration parameter 1 override.
        custom2: Optional custom configuration parameter 2 override.
        custom3: Optional custom configuration parameter 3 override.
        custom4: Optional custom configuration parameter 4 override.
        custom5: Optional custom configuration parameter 5 override.

    Returns:
        Device: The opened device, ready for direct backend construction.

    Raises:
        IndexError: If the ID is not registered.
        RuntimeError: If the device library cannot be loaded or initialized.
    """
