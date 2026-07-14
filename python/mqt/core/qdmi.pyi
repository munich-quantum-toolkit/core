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

    def __init__(self) -> None: ...
    @property
    def base_url(self) -> str | None: ...
    @base_url.setter
    def base_url(self, arg: str | None) -> None: ...
    @property
    def token(self) -> str | None: ...
    @token.setter
    def token(self, arg: str | None) -> None: ...
    @property
    def auth_file(self) -> pathlib.Path | None: ...
    @auth_file.setter
    def auth_file(self, arg: str | os.PathLike | None) -> None: ...
    @property
    def auth_url(self) -> str | None: ...
    @auth_url.setter
    def auth_url(self, arg: str | None) -> None: ...
    @property
    def username(self) -> str | None: ...
    @username.setter
    def username(self, arg: str | None) -> None: ...
    @property
    def password(self) -> str | None: ...
    @password.setter
    def password(self, arg: str | None) -> None: ...
    @property
    def custom1(self) -> str | None: ...
    @custom1.setter
    def custom1(self, arg: str | None) -> None: ...
    @property
    def custom2(self) -> str | None: ...
    @custom2.setter
    def custom2(self, arg: str | None) -> None: ...
    @property
    def custom3(self) -> str | None: ...
    @custom3.setter
    def custom3(self, arg: str | None) -> None: ...
    @property
    def custom4(self) -> str | None: ...
    @custom4.setter
    def custom4(self, arg: str | None) -> None: ...
    @property
    def custom5(self) -> str | None: ...
    @custom5.setter
    def custom5(self, arg: str | None) -> None: ...

class DeviceDefinition:
    """A side-effect-free QDMI device registration."""

    def __init__(
        self,
        id: str,
        library: str | os.PathLike,
        prefix: str,
        *,
        abi: str = "qdmi-v1",
        enabled: bool = True,
        session: SessionParameters = ...,
    ) -> None: ...
    @property
    def id(self) -> str: ...
    @id.setter
    def id(self, arg: str, /) -> None: ...
    @property
    def library(self) -> pathlib.Path: ...
    @library.setter
    def library(self, arg: str | os.PathLike, /) -> None: ...
    @property
    def abi(self) -> str: ...
    @abi.setter
    def abi(self, arg: str, /) -> None: ...
    @property
    def prefix(self) -> str: ...
    @prefix.setter
    def prefix(self, arg: str, /) -> None: ...
    @property
    def enabled(self) -> bool: ...
    @enabled.setter
    def enabled(self, arg: bool, /) -> None: ...
    @property
    def session(self) -> SessionParameters: ...
    @session.setter
    def session(self, arg: SessionParameters, /) -> None: ...
    @property
    def source(self) -> pathlib.Path: ...

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
    ) -> None: ...

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

class DeviceManager:
    """Discovers and lazily opens QDMI devices."""

    def __init__(self, options: ConfigOptions = ...) -> None: ...
    @property
    def definitions(self) -> list[DeviceDefinition]: ...
    def register_device(self, definition: DeviceDefinition, *, replace: bool = False) -> None: ...
    def unregister_device(self, id: str) -> bool: ...
    def open(self, id: str, *, session_overrides: SessionParameters = ...) -> Device: ...
