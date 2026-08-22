# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""QDMI entities and access to MQT Core's QDMI driver."""

import enum
from collections.abc import Sequence
from typing import overload

from mqt.core.qdmi import driver as driver
from mqt.core.qdmi import slurm as slurm

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
    def program_bytes(self) -> bytes:
        """The exact bytes of the submitted program."""

    @property
    def num_shots(self) -> int:
        """The number of shots."""

    @property
    def queue_position(self) -> int | None:
        """The number of jobs ahead in the queue, or None if unavailable or not applicable in the current state."""

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

def is_binary_program_format(program_format: ProgramFormat) -> bool:
    """Returns whether a program format carries a binary payload.

    ``QIR_BASE_MODULE``, ``QIR_ADAPTIVE_MODULE``, and ``QPY`` hold bitcode or
    another serialized object. Such a payload may contain a null byte and is not
    text, so the device must receive it as exact bytes. Pass ``bytes`` to
    :meth:`Device.submit_job` for these formats and ``str`` for the others.

    Args:
        program_format: The program format to classify.

    Returns:
        True if the format requires exact-byte submission.
    """

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

    def queue_length(self) -> int | None:
        """Returns the current queue length, or None if unavailable."""

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
        """Returns the program formats reported by the device. Raises if the property is unsupported."""

    def try_supported_program_formats(self) -> list[ProgramFormat] | None:
        """Returns the reported program formats, including an empty list, or None if the property is unsupported."""

    def child_devices(self) -> list[Device]:
        """Returns the direct child devices managed by this device."""

    def query_custom_operations(self, custom_property: CustomProperty) -> list[Operation] | None:
        """Query a custom device property that contains operation handles.

        Returns normal :class:`Device.Operation` objects, or ``None`` when the custom
        slot is unsupported. A supported empty list is returned as an empty list.
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
        """Query an implementation-defined custom device property.

        The caller must provide the type documented by the device implementation.
        Use ``bytes`` to retrieve the value without interpretation. Returns ``None``
        when the custom slot is unsupported.
        """

    @overload
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
        """Submits a text job to the device."""

    @overload
    def submit_job(
        self,
        program: bytes,
        program_format: ProgramFormat,
        num_shots: int,
        *,
        custom1: str | bool | float | None = None,
        custom2: str | bool | float | None = None,
        custom3: str | bool | float | None = None,
        custom4: str | bool | float | None = None,
        custom5: str | bool | float | None = None,
    ) -> Job:
        """Submits an exact byte payload to the device."""

    def submit_calibration_job(
        self,
        program: str | bytes | None = None,
        *,
        custom1: str | bool | float | None = None,
        custom2: str | bool | float | None = None,
        custom3: str | bool | float | None = None,
        custom4: str | bool | float | None = None,
        custom5: str | bool | float | None = None,
    ) -> Job:
        """Triggers a calibration run on the device.

        QDMI does not require a program for a calibration run, so ``program`` is
        optional and may be a string or bytes. When it is given, the device defines
        what it means, which is usually a configuration for the run. A calibration run
        executes no circuit, so it takes no shot count.
        """

    def retrieve_job_by_id(self, job_id: str) -> Job:
        """Retrieves an existing job by its device-provided ID."""

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
