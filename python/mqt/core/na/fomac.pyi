# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Reconstruction of NADevice from QDMI's Device class."""

from collections.abc import Iterable

from ..fomac import Device as GenericDevice

__all__ = ["Device"]

class Device(GenericDevice):
    """Represents a device with a lattice of traps."""

    class Vector:
        """Represents a 2D vector."""

        x: int
        """
        The x-coordinate of the vector.
        """
        y: int
        """
        The y-coordinate of the vector.
        """
        def __eq__(self, other: object) -> bool: ...
        def __ne__(self, other: object) -> bool: ...

    class Region:
        """Represents a region in the device."""

        origin: Device.Vector
        """
        The origin of the region.
        """

        class Size:
            """Represents the size of a region."""

            width: int
            """
            The width of the region.
            """
            height: int
            """
            The height of the region.
            """
            def __eq__(self, other: object) -> bool: ...
            def __ne__(self, other: object) -> bool: ...

        size: Size
        """
        The size of the region.
        """

        def __eq__(self, other: object) -> bool: ...
        def __ne__(self, other: object) -> bool: ...

    class Lattice:
        """Represents a lattice of traps in the device."""

        lattice_origin: Device.Vector
        """
        The origin of the lattice.
        """
        lattice_vector_1: Device.Vector
        """
        The first lattice vector.
        """
        lattice_vector_2: Device.Vector
        """
        The second lattice vector.
        """
        sublattice_offsets: Iterable[Device.Vector]
        """
        The offsets of the sublattices.
        """
        extent: Device.Region
        """
        The extent of the lattice.
        """
        def __eq__(self, other: object) -> bool: ...
        def __ne__(self, other: object) -> bool: ...

    traps: Iterable[Device.Lattice]
    """
    The list of trap positions in the device.
    """
    t1: int
    """
    The T1 time of the device.
    """
    t2: int
    """
    The T2 time of the device.
    """

    @staticmethod
    def try_create_from_device(device: GenericDevice) -> Device:
        """Create NA FoMaC Device from generic FoMaC Device."""
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...

def devices() -> Iterable[Device]:
    """Returns a list of available devices."""
