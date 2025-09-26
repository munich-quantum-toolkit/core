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
    "devices",
]

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
        def is_zone(self) -> bool | None:
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
        def is_zoned(self, sites: Iterable[Device.Site] = ..., params: Iterable[float] = ...) -> bool | None:
            """Returns whether the operation is zoned."""
        def sites(
            self, sites: Iterable[Device.Site] = ..., params: Iterable[float] = ...
        ) -> Iterable[Device.Site] | None:
            """Returns the list of sites the operation can be performed on."""
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
    def sites(self) -> Iterable[Site]:
        """Returns the list of sites available on the device."""
    def operations(self) -> Iterable[Operation]:
        """Returns the list of operations supported by the device."""
    def coupling_map(self) -> Iterable[tuple[Site, Site]] | None:
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
    def __eq__(self, other: object) -> bool:
        """Checks if two devices are equal."""
    def __ne__(self, other: object) -> bool:
        """Checks if two devices are not equal."""

def devices() -> Iterable[Device]:
    """Returns a list of available devices."""
