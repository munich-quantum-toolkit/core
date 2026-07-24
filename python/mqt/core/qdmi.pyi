# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

import enum
import os
import pathlib
from collections.abc import Sequence
from typing import overload

class SessionParameters:
    """Parameters for one QDMI device session."""

    def __init__(self) -> None:
        """Create empty session parameters."""

    @property
    def base_url(self) -> str | None:
        """Base URL of the device service."""

    @base_url.setter
    def base_url(self, arg: str | None) -> None: ...
    @property
    def token(self) -> str | None:
        """Authentication token."""

    @token.setter
    def token(self, arg: str | None) -> None: ...
    @property
    def auth_file(self) -> pathlib.Path | None:
        """Path to an authentication file."""

    @auth_file.setter
    def auth_file(self, arg: str | os.PathLike | None) -> None: ...
    @property
    def auth_url(self) -> str | None:
        """URL of the authentication service."""

    @auth_url.setter
    def auth_url(self, arg: str | None) -> None: ...
    @property
    def username(self) -> str | None:
        """Authentication username."""

    @username.setter
    def username(self, arg: str | None) -> None: ...
    @property
    def password(self) -> str | None:
        """Authentication password."""

    @password.setter
    def password(self, arg: str | None) -> None: ...
    @property
    def custom1(self) -> str | None:
        """First implementation-defined session parameter."""

    @custom1.setter
    def custom1(self, arg: str | None) -> None: ...
    @property
    def custom2(self) -> str | None:
        """Second implementation-defined session parameter."""

    @custom2.setter
    def custom2(self, arg: str | None) -> None: ...
    @property
    def custom3(self) -> str | None:
        """Third implementation-defined session parameter."""

    @custom3.setter
    def custom3(self, arg: str | None) -> None: ...
    @property
    def custom4(self) -> str | None:
        """Fourth implementation-defined session parameter."""

    @custom4.setter
    def custom4(self, arg: str | None) -> None: ...
    @property
    def custom5(self) -> str | None:
        """Fifth implementation-defined session parameter."""

    @custom5.setter
    def custom5(self, arg: str | None) -> None: ...

class DeviceDefinition:
    """A side-effect-free QDMI device registration."""

    def __init__(
        self,
        device_id: str,
        library: str | os.PathLike,
        prefix: str,
        *,
        enabled: bool = True,
        session: SessionParameters = ...,
    ) -> None:
        """Create a device definition without loading its library.

        Args:
            device_id: Stable identifier used for discovery and opening.
            library: Path to the native QDMI device library.
            prefix: Symbol prefix exported by the QDMI implementation.
            enabled: Whether the definition participates in discovery.
            session: Default parameters for sessions opened from this definition.
        """

    @property
    def device_id(self) -> str:
        """Stable device identifier."""

    @device_id.setter
    def device_id(self, arg: str, /) -> None: ...
    @property
    def library(self) -> pathlib.Path:
        """Path to the native QDMI device library."""

    @library.setter
    def library(self, arg: str | os.PathLike, /) -> None: ...
    @property
    def prefix(self) -> str:
        """Symbol prefix exported by the device library."""

    @prefix.setter
    def prefix(self, arg: str, /) -> None: ...
    @property
    def enabled(self) -> bool:
        """Whether this definition is enabled."""

    @enabled.setter
    def enabled(self, arg: bool, /) -> None: ...
    @property
    def session(self) -> SessionParameters:
        """Default parameters for newly opened sessions."""

    @session.setter
    def session(self, arg: SessionParameters, /) -> None: ...
    @property
    def source(self) -> pathlib.Path:
        """Configuration source that declared the definition."""

class ConfigOptions:
    """Controls QDMI configuration discovery."""

    def __init__(
        self,
        *,
        config_root: str | os.PathLike | None = None,
        explicit_file: str | os.PathLike | None = None,
        base_directory: str | os.PathLike | None = None,
        isolated: bool = False,
        inline_json: str | None = None,
        runtime_overrides: Sequence[DeviceDefinition] = [],
    ) -> None:
        """Create configuration discovery options.

        Args:
            config_root: Root containing relocatable built-in manifest fragments.
            explicit_file: Configuration file replacing system, user, and project discovery.
            base_directory: Base for relative paths in inline configuration.
            isolated: Exclude packaged built-in definitions when true.
            inline_json: JSON configuration layered above discovered files.
            runtime_overrides: Device definitions applied at highest precedence.
        """

class Job:
    """A submitted quantum program execution retaining its device session."""

    def check(self) -> Status:
        """Return the current QDMI job status."""

    def wait(self, timeout: int = 0) -> bool:
        """Waits for the job to complete.

        Args:
            timeout: The maximum time to wait in seconds. If 0, waits indefinitely.

        Returns:
            True if the job completed within the timeout, False otherwise.
        """

    def cancel(self) -> None:
        """Request cancellation of the job."""

    def get_shots(self) -> list[str]:
        """Return the raw shot results."""

    def get_counts(self) -> dict[str, int]:
        """Return measurement counts keyed by bit string."""

    def get_dense_statevector(self) -> list[complex]:
        """Return the dense state vector.

        This result is typically available only from simulator devices.
        """

    def get_dense_probabilities(self) -> list[float]:
        """Return the dense probability vector.

        This result is typically available only from simulator devices.
        """

    def get_sparse_statevector(self) -> dict[str, complex]:
        """Return the sparse state vector keyed by basis state.

        This result is typically available only from simulator devices.
        """

    def get_sparse_probabilities(self) -> dict[str, float]:
        """Return sparse probabilities keyed by basis state.

        This result is typically available only from simulator devices.
        """

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
        Use ``bytes`` to retrieve the value without interpretation.

        Args:
            custom_property: Custom property slot to query.
            value_type: Expected Python type of the property value.

        Returns:
            The typed property value, or ``None`` when the slot is unsupported.
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
        Use ``bytes`` to retrieve the value without interpretation.

        Args:
            custom_property: Custom result slot to retrieve.
            value_type: Expected Python type of the result value.

        Returns:
            The typed result value, or ``None`` when the slot is unsupported.
        """

    @property
    def id(self) -> str:
        """The provider-assigned job identifier."""

    @property
    def program_format(self) -> ProgramFormat:
        """The QDMI format of the submitted program."""

    @property
    def program(self) -> str:
        """The submitted program."""

    @property
    def num_shots(self) -> int:
        """The requested number of shots."""

    def __eq__(self, arg: object, /) -> bool:
        """Return whether two objects refer to the same job."""

    def __ne__(self, arg: object, /) -> bool:
        """Return whether two objects refer to different jobs."""

    class Status(enum.Enum):
        """Status values defined by QDMI."""

        CREATED = 0

        SUBMITTED = 1

        QUEUED = 2

        RUNNING = 3

        DONE = 4

        CANCELED = 5

        FAILED = 6

class ProgramFormat(enum.Enum):
    """Program formats defined by QDMI."""

    QASM2 = 0

    QASM3 = 1

    QIR_BASE_STRING = 2

    QIR_BASE_MODULE = 3

    QIR_ADAPTIVE_STRING = 4

    QIR_ADAPTIVE_MODULE = 5

    CALIBRATION = 6

    QPY = 7

    IQM_JSON = 8

    BATCH_JOB = 9

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
    """One initialized QDMI device session.

    The object owns the native library and session state required by its sites,
    operations, child devices, and jobs.
    """

    class Status(enum.Enum):
        """Status values defined by QDMI."""

        OFFLINE = 0

        IDLE = 1

        BUSY = 2

        ERROR = 3

        MAINTENANCE = 4

        CALIBRATION = 5

    def name(self) -> str:
        """Return the provider-reported device name."""

    def version(self) -> str:
        """Return the provider-reported device version."""

    def status(self) -> Status:
        """Return the current QDMI device status."""

    def library_version(self) -> str:
        """Return the provider library version."""

    def qubits_num(self) -> int:
        """Return the number of qubits available on the device."""

    def sites(self) -> list[Site]:
        """Return all regular sites and zones."""

    def regular_sites(self) -> list[Site]:
        """Return sites that are not zones."""

    def zones(self) -> list[Site]:
        """Return sites that represent zones."""

    def operations(self) -> list[Operation]:
        """Return operations supported by the device."""

    def coupling_map(self) -> list[tuple[Site, Site]] | None:
        """Return the optional coupling map as site pairs."""

    def needs_calibration(self) -> int | None:
        """Return the optional calibration requirement."""

    def length_unit(self) -> str | None:
        """Return the optional device length unit."""

    def length_scale_factor(self) -> float | None:
        """Return the optional length scale factor."""

    def duration_unit(self) -> str | None:
        """Return the optional device duration unit."""

    def duration_scale_factor(self) -> float | None:
        """Return the optional duration scale factor."""

    def min_atom_distance(self) -> int | None:
        """Return the optional minimum atom distance."""

    def supported_program_formats(self) -> list[ProgramFormat]:
        """Return the QDMI program formats accepted by the device."""

    def child_devices(self) -> list[Device]:
        """Return directly managed child devices."""

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
        Use ``bytes`` to retrieve the value without interpretation.

        Args:
            custom_property: Custom property slot to query.
            value_type: Expected Python type of the property value.

        Returns:
            The typed property value, or ``None`` when the slot is unsupported.
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
        """Submit a quantum program to the device.

        Args:
            program: Serialized program data.
            program_format: QDMI format of ``program``.
            num_shots: Number of requested executions.
            custom1: First implementation-defined job parameter.
            custom2: Second implementation-defined job parameter.
            custom3: Third implementation-defined job parameter.
            custom4: Fourth implementation-defined job parameter.
            custom5: Fifth implementation-defined job parameter.

        Returns:
            A job retaining the device session.
        """

    def __eq__(self, arg: object, /) -> bool:
        """Return whether two objects refer to the same device."""

    def __ne__(self, arg: object, /) -> bool:
        """Return whether two objects refer to different devices."""

    class Site:
        """A physical site or zone belonging to a device."""

        def index(self) -> int:
            """Return the provider-assigned site index."""

        def t1(self) -> int | None:
            """Return the optional T1 coherence time."""

        def t2(self) -> int | None:
            """Return the optional T2 coherence time."""

        def name(self) -> str | None:
            """Return the optional site name."""

        def x_coordinate(self) -> int | None:
            """Return the optional x coordinate."""

        def y_coordinate(self) -> int | None:
            """Return the optional y coordinate."""

        def z_coordinate(self) -> int | None:
            """Return the optional z coordinate."""

        def is_zone(self) -> bool:
            """Return whether this site represents a zone."""

        def x_extent(self) -> int | None:
            """Return the optional x extent of the zone."""

        def y_extent(self) -> int | None:
            """Return the optional y extent of the zone."""

        def z_extent(self) -> int | None:
            """Return the optional z extent of the zone."""

        def module_index(self) -> int | None:
            """Return the optional module index."""

        def submodule_index(self) -> int | None:
            """Return the optional submodule index."""

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
            Use ``bytes`` to retrieve the value without interpretation.

            Args:
                custom_property: Custom property slot to query.
                value_type: Expected Python type of the property value.

            Returns:
                The typed property value, or ``None`` when the slot is unsupported.
            """

        def __eq__(self, arg: object, /) -> bool:
            """Return whether two objects refer to the same site."""

        def __ne__(self, arg: object, /) -> bool:
            """Return whether two objects refer to different sites."""

    class Operation:
        """A quantum operation supported by a device."""

        def name(self, sites: Sequence[Device.Site] = ..., params: Sequence[float] = ...) -> str:
            """Return the operation name for the given sites and parameters."""

        def qubits_num(self, sites: Sequence[Device.Site] = ..., params: Sequence[float] = ...) -> int | None:
            """Return the optional operation arity."""

        def parameters_num(self, sites: Sequence[Device.Site] = ..., params: Sequence[float] = ...) -> int:
            """Return the number of operation parameters."""

        def duration(self, sites: Sequence[Device.Site] = ..., params: Sequence[float] = ...) -> int | None:
            """Return the optional duration for this operation instance."""

        def fidelity(self, sites: Sequence[Device.Site] = ..., params: Sequence[float] = ...) -> float | None:
            """Return the optional fidelity for this operation instance."""

        def interaction_radius(self, sites: Sequence[Device.Site] = ..., params: Sequence[float] = ...) -> int | None:
            """Return the optional interaction radius."""

        def blocking_radius(self, sites: Sequence[Device.Site] = ..., params: Sequence[float] = ...) -> int | None:
            """Return the optional blocking radius."""

        def idling_fidelity(self, sites: Sequence[Device.Site] = ..., params: Sequence[float] = ...) -> float | None:
            """Return the optional idling fidelity."""

        def is_zoned(self) -> bool:
            """Return whether the operation is restricted to zones."""

        def sites(self) -> list[Device.Site] | None:
            """Return sites on which the operation is available."""

        def site_pairs(self) -> list[tuple[Device.Site, Device.Site]] | None:
            """Return supported site pairs for a local two-site operation."""

        def mean_shuttling_speed(self, sites: Sequence[Device.Site] = ..., params: Sequence[float] = ...) -> int | None:
            """Return the optional mean shuttling speed."""

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
            Use ``bytes`` to retrieve the value without interpretation.

            Args:
                custom_property: Custom property slot to query.
                value_type: Expected Python type of the property value.
                sites: Sites for the operation instance.
                params: Parameters for the operation instance.

            Returns:
                The typed property value, or ``None`` when the slot is unsupported.
            """

        def __eq__(self, arg: object, /) -> bool:
            """Return whether two objects refer to the same operation."""

        def __ne__(self, arg: object, /) -> bool:
            """Return whether two objects refer to different operations."""

class OpenAllResult:
    """Devices and per-ID errors produced by bulk opening."""

    @property
    def devices(self) -> dict[str, Device]:
        """Successfully opened devices keyed by stable ID."""

    @property
    def errors(self) -> dict[str, str]:
        """Error messages for failed definitions keyed by stable ID."""

class DeviceManager:
    """Discover and lazily open QDMI devices.

    Definitions are discovered without loading native libraries. Opening a device
    creates an independent session while compatible devices may share a loaded
    library.
    """

    def __init__(self, options: ConfigOptions = ...) -> None:
        """Create a manager using the supplied discovery options."""

    @property
    def definitions(self) -> list[DeviceDefinition]:
        """A snapshot of the currently registered device definitions."""

    def register_device(self, definition: DeviceDefinition, *, replace: bool = False) -> None:
        """Register a complete device definition.

        Args:
            definition: Definition to register.
            replace: Replace an existing definition with the same ID.
        """

    def unregister_device(self, device_id: str) -> bool:
        """Remove a definition without invalidating opened devices.

        Returns:
            Whether a definition with the requested ID existed.
        """

    def open(self, device_id: str, *, session_overrides: SessionParameters = ...) -> Device:
        """Open one device by stable ID.

        The supplied session values override the definition defaults field by field.
        The native library is loaded only when this method is called.
        """

    def open_all(self, *, session_overrides: SessionParameters = ...) -> OpenAllResult:
        """Open a snapshot of all definitions independently.

        Failures are retained by device ID and do not prevent other definitions from
        opening.
        """
