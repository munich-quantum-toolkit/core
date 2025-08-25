# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from enum import Enum

__all__ = [
    "Device",
    "DeviceStatus",
    "Operation",
    "Site",
    "devices",
]

class DeviceStatus(Enum):
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
    def t1(self) -> int:
        """Returns the T1 coherence time of the site."""
    def t2(self) -> int:
        """Returns the T2 coherence time of the site."""
    def name(self) -> str:
        """Returns the name of the site."""
    def x_coordinate(self) -> int:
        """Returns the x coordinate of the site."""
    def y_coordinate(self) -> int:
        """Returns the y coordinate of the site."""
    def z_coordinate(self) -> int:
        """Returns the z coordinate of the site."""
    def is_zone(self) -> bool:
        """Returns whether the site is a zone."""
    def x_extent(self) -> int:
        """Returns the x extent of the site."""
    def y_extent(self) -> int:
        """Returns the y extent of the site."""
    def z_extent(self) -> int:
        """Returns the z extent of the site."""
    def module_index(self) -> int:
        """Returns the index of the module the site belongs to."""
    def submodule_index(self) -> int:
        """Returns the index of the submodule the site belongs to."""
    def __eq__(self, other: object) -> bool:
        """Checks if two sites are equal."""
    def __ne__(self, other: object) -> bool:
        """Checks if two sites are not equal."""

class Operation:
    """An operation represents a quantum operation that can be performed on a quantum device."""
    def name(self) -> str:
        """Returns the name of the operation."""
    def qubits_num(self) -> int:
        """Returns the number of qubits the operation acts on."""
    def parameters_num(self) -> int:
        """Returns the number of parameters the operation has."""
    def duration(self) -> int:
        """Returns the duration of the operation."""
    def fidelity(self) -> float:
        """Returns the fidelity of the operation."""
    def interaction_radius(self) -> int:
        """Returns the interaction radius of the operation."""
    def blocking_radius(self) -> int:
        """Returns the blocking radius of the operation."""
    def idling_fidelity(self) -> float:
        """Returns the idling fidelity of the operation."""
    def is_zoned(self) -> bool:
        """Returns whether the operation is zoned."""
    def sites(self) -> list[Site]:
        """Returns the list of sites the operation can be performed on."""
    def mean_shuttling_speed(self) -> int:
        """Returns the mean shuttling speed of the operation."""
    def __eq__(self, other: object) -> bool:
        """Checks if two operations are equal."""
    def __ne__(self, other: object) -> bool:
        """Checks if two operations are not equal."""

class Device:
    """A device represents a quantum device with its properties and capabilities."""
    def name(self) -> str:
        """Returns the name of the device."""
    def version(self) -> str:
        """Returns the version of the device."""
    def status(self) -> DeviceStatus:
        """Returns the current status of the device."""
    def library_version(self) -> str:
        """Returns the version of the library used to define the device."""
    def qubits_num(self) -> int:
        """Returns the number of qubits available on the device."""
    def sites(self) -> list[Site]:
        """Returns the list of sites available on the device."""
    def operations(self) -> list[Operation]:
        """Returns the list of operations supported by the device."""
    def coupling_map(self) -> list[tuple[Site, Site]]:
        """Returns the coupling map of the device as a list of site pairs."""
    def needs_calibration(self) -> int:
        """Returns whether the device needs calibration."""
    def length_unit(self) -> str:
        """Returns the unit of length used by the device."""
    def length_scale_factor(self) -> float:
        """Returns the scale factor for length used by the device."""
    def duration_unit(self) -> str:
        """Returns the unit of duration used by the device."""
    def duration_scale_factor(self) -> float:
        """Returns the scale factor for duration used by the device."""
    def min_atom_distance(self) -> int:
        """Returns the minimum atom distance on the device."""
    def __eq__(self, other: object) -> bool:
        """Checks if two devices are equal."""
    def __ne__(self, other: object) -> bool:
        """Checks if two devices are not equal."""

def devices() -> list[Device]:
    """Returns a list of available devices."""
