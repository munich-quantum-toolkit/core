# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Reconstruction of NADevice from QDMI's Device class."""

from __future__ import annotations

from functools import partial
from itertools import groupby
from typing import TYPE_CHECKING, Any

from ..qdmi.fomac import Device, Site

if TYPE_CHECKING:
    import builtins

__all__ = ["Device"]


class Vector:
    """Represents a 2D vector.

    Attributes:
        x (int): The x-coordinate.
        y (int): The y-coordinate.
    """

    def __init__(self, x: int | None = None, y: int | None = None) -> None:
        self.x = x if x is not None else 0
        self.y = y if y is not None else 0

    def __repr__(self) -> str:
        return f"Vector(x={self.x}, y={self.y})"

    def dict(self) -> builtins.dict[str, int]:
        return {"x": self.x, "y": self.y}


class Size:
    """Represents the size of a.

    Attributes:
        width (int): The width of the
        height (int): The height of the
    """

    def __init__(self, width: int = 0, height: int = 0) -> None:
        self.width = width
        self.height = height

    def __repr__(self) -> str:
        return f"Size(width={self.width}, height={self.height})"

    def dict(self) -> builtins.dict[str, int]:
        """Returns a dictionary representation of the size."""
        return {"width": self.width, "height": self.height}


class Region:
    """Represents a region in the device.

    Attributes:
        origin (Vector): The origin of the
        size (Size): The size of the
    """

    def __init__(self, origin: Vector | None = None, size: Size | None = None) -> None:
        self.origin = origin if origin is not None else Vector()
        self.size = size if size is not None else Size()

    def __repr__(self) -> str:
        return f"Region(origin={self.origin}, size={self.size})"

    def dict(self) -> builtins.dict[str, builtins.dict[str, int]]:
        """Returns a dictionary representation of the region."""
        return {"origin": self.origin.dict(), "size": self.size.dict()}


class Lattice:
    """Represents a lattice of traps in the device.

    Attributes:
        lattice_origin (Vector): The origin of the lattice.
        lattice_vector1 (Vector): The first lattice vector.
        lattice_vector2 (Vector): The second lattice vector.
        sublattice_offsets (list[Vector]): The offsets for each sublattice.
        extent (Region): The extent of the lattice.
    """

    def __init__(
        self,
        lattice_origin: Vector | None = None,
        lattice_vector1: Vector | None = None,
        lattice_vector2: Vector | None = None,
        sublattice_offsets: list[Vector] | None = None,
        extent: Region | None = None,
    ) -> None:
        self.lattice_origin = lattice_origin if lattice_origin is not None else Vector()
        self.lattice_vector1 = lattice_vector1 if lattice_vector1 is not None else Vector()
        self.lattice_vector2 = lattice_vector2 if lattice_vector2 is not None else Vector()
        self.sublattice_offsets = sublattice_offsets if sublattice_offsets is not None else []
        self.extent = extent if extent is not None else Region()

    def __repr__(self) -> str:
        return f"Lattice(extent={self.extent})"

    def dict(self) -> builtins.dict[str, Any]:
        """Returns a dictionary representation of the lattice."""
        return {
            "latticeOrigin": self.lattice_origin.dict(),
            "latticeVector1": self.lattice_vector1.dict(),
            "latticeVector2": self.lattice_vector2.dict(),
            "sublatticeOffsets": [offset.dict() for offset in self.sublattice_offsets],
            "extent": self.extent.dict(),
        }


class Unit:
    """Represents a unit of measurement for length and time.

    Attributes:
        unit (str): The unit of measurement (e.g., "um" for micrometers, "ns" for nanoseconds).
        scale_factor (float): The factor of the unit.
    """

    def __init__(self, unit: str | None = None, scale_factor: float | None = None) -> None:
        self.unit = unit if unit is not None else ""
        self.scale_factor = scale_factor if scale_factor is not None else 1.0

    def __repr__(self) -> str:
        return f"Unit(unit='{self.unit}', scale_factor={self.scale_factor})"

    def dict(self) -> builtins.dict[str, Any]:
        """Returns a dictionary representation of the unit."""
        return {"unit": self.unit, "scaleFactor": self.scale_factor}


class DecoherenceTimes:
    """Represents the decoherence times of the device.

    Attributes:
        t1 (int): The T1 time.
        t2 (int): The T2 time.
    """

    def __init__(self, t1: int | None = None, t2: int | None = None) -> None:
        self.t1 = t1 if t1 is not None else 0
        self.t2 = t2 if t2 is not None else 0

    def __repr__(self) -> str:
        return f"DecoherenceTimes(t1={self.t1}, t2={self.t2})"

    def dict(self) -> builtins.dict[str, Any]:
        """Returns a dictionary representation of the decoherence times."""
        return {"t1": self.t1, "t2": self.t2}


class NADevice:
    """Represents a device with a lattice of traps.

    Attributes:
        name (str): The name of the device.
        num_qubits (int): The number of qubits in the device.
        min_atom_distance (int): The minimum distance between atoms in the device.
        length_unit (Unit): The unit of measurement for length.
        duration_unit (Unit): The unit of measurement for time.
        decoherence_times (DecoherenceTimes): The decoherence times of the device.
        traps (list[Lattice]): The list of lattices in the device.
    """

    def __init__(self, device: Device | None = None) -> None:
        self.name = device.name() if device is not None else "Unnamed NADevice"
        self.num_qubits = device.qubits_num() if device is not None else 0
        self.min_atom_distance = device.min_atom_distance() if device is not None else 0
        self.length_unit = Unit(device.length_unit(), device.length_scale_factor()) if device is not None else Unit()
        self.duration_unit = (
            Unit(device.duration_unit(), device.duration_scale_factor()) if device is not None else Unit()
        )
        self.decoherence_times = (
            DecoherenceTimes(device.sites()[0].t1(), device.sites()[0].t2())
            if device is not None
            else DecoherenceTimes()
        )
        self.traps: list[Lattice] = []
        if device is not None:
            self._from_device(device)

    def __repr__(self) -> str:
        return f"NADevice(name='{self.name})"

    def dict(self) -> builtins.dict[str, Any]:
        """Returns a dictionary representation of the NADevice."""
        return {
            "name": self.name,
            "numQubits": self.num_qubits,
            "minAtomDistance": self.min_atom_distance,
            "lengthUnit": self.length_unit.dict(),
            "durationUnit": self.duration_unit.dict(),
            "decoherenceTimes": self.decoherence_times.dict(),
            "traps": [lattice.dict() for lattice in self.traps],
        }

    def _from_device(self, device: Device) -> None:
        """Initializes the NADevice from a given Device object.

        Args:
            device (Device): The device to initialize from.

        Raises:
            ValueError: If any site coordinate is None or required reference site is missing.
        """
        # get the non-zone sites of the device
        sites = [site for site in device.sites() if not site.is_zone()]
        # group sites by their module index (i.e., lattice) and iterate over each module
        for _, module in groupby(sites, lambda site: site.module_index()):
            # create a lattice
            lattice = Lattice()
            # convert the module from a generator to a list
            module_list = list(module)
            # get the submodule of the first site (i.e., sites[0])
            submodule0_index = module_list[0].submodule_index()
            submodule0 = [site for site in sites if site.submodule_index() == submodule0_index]
            # get the left- and bottom-most site in the sublattice
            reference_site0 = min(submodule0, key=lambda site: (site.x_coordinate(), site.y_coordinate()))
            # set the lattice origin to the coordinates of reference_site0
            x0 = reference_site0.x_coordinate()
            y0 = reference_site0.y_coordinate()
            if x0 is None or y0 is None:
                msg = "Reference site 0 coordinates must not be None."
                raise ValueError(msg)
            lattice.lattice_origin = Vector(x0, y0)
            # set the sublattice offsets to the coordinates of the sublattice sites relative to the first site
            for site in submodule0:
                x = site.x_coordinate()
                y = site.y_coordinate()
                if x is None or y is None:
                    msg = "Site coordinates must not be None."
                    raise ValueError(msg)
                lattice.sublattice_offsets.append(Vector(x - lattice.lattice_origin.x, y - lattice.lattice_origin.y))
            # get a list of the left- and bottom-most sites of each sublattice in the module
            reference_sites = [
                min(submodule, key=lambda site: (site.x_coordinate(), site.y_coordinate()))
                for index, submodule in groupby(module_list, lambda site: site.submodule_index())
                if index != submodule0_index
            ]

            # find the sites closest to the first site (reference_site0)

            def site_distance_sq(origin: Vector, site: Site) -> int:
                """Calculate the squared distance between a site and an origin vector.

                Args:
                    origin (Vector): The origin vector.
                    site (Site): The site to calculate the distance to.

                Returns:
                    int: The squared distance between the site and the origin.

                Raises:
                    ValueError: If the site coordinates are None.
                """
                x = site.x_coordinate()
                y = site.y_coordinate()
                if x is None or y is None:
                    msg = "Site coordinates must not be None."
                    raise ValueError(msg)
                return (x - origin.x) ** 2 + (y - origin.y) ** 2

            reference_site1 = min(reference_sites, key=partial(site_distance_sq, lattice.lattice_origin))
            # set the first lattice vector to the coordinates of reference_site1 relative to reference_site0
            x1 = reference_site1.x_coordinate()
            y1 = reference_site1.y_coordinate()
            if x1 is None or y1 is None:
                msg = "Reference site 1 coordinates must not be None."
                raise ValueError(msg)
            lattice.lattice_vector1 = Vector(
                x1 - lattice.lattice_origin.x,
                y1 - lattice.lattice_origin.y,
            )

            # remove all sublattices that are collinear with reference_site0 and reference_site1

            def is_not_collinear(site: Site, origin: Vector, vector1: Vector) -> bool:
                """Check if a site is not collinear with two vectors.

                Args:
                    site (Site): The site to check.
                    origin (Vector): The origin vector.
                    vector1 (Vector): The first vector.

                Returns:
                    bool: True if the site is not collinear with the two vectors, False otherwise.

                Raises:
                    ValueError: If the site coordinates are None.
                """
                x = site.x_coordinate()
                y = site.y_coordinate()
                if x is None or y is None:
                    msg = "Site coordinates must not be None."
                    raise ValueError(msg)
                return (x - origin.x) * vector1.y != (y - origin.y) * vector1.x

            reference_sites = [
                site
                for site in reference_sites
                if is_not_collinear(site, lattice.lattice_origin, lattice.lattice_vector1)
            ]
            # find the site closest to reference_site0 that is not collinear with reference_site1
            reference_site2 = min(reference_sites, key=partial(site_distance_sq, lattice.lattice_origin))
            # set the second lattice vector to the coordinates of reference_site2 relative to reference_site0
            x2 = reference_site2.x_coordinate()
            y2 = reference_site2.y_coordinate()
            if x2 is None or y2 is None:
                msg = "Reference site 2 coordinates must not be None."
                raise ValueError(msg)
            lattice.lattice_vector2 = Vector(
                x2 - lattice.lattice_origin.x,
                y2 - lattice.lattice_origin.y,
            )
            # set the extent of the lattice to the bounding box of all sites in the module
            min_x, max_x = (
                min(x for x in [site.x_coordinate() for site in module_list] if x is not None),
                max(x for x in [site.x_coordinate() for site in module_list] if x is not None),
            )
            min_y, max_y = (
                min(y for y in [site.y_coordinate() for site in module_list] if y is not None),
                max(y for y in [site.y_coordinate() for site in module_list] if y is not None),
            )
            lattice.extent = Region(Vector(min_x, min_y), Size(max_x - min_x + 1, max_y - min_y + 1))
            # add the lattice to the list of lattices
            self.traps.append(lattice)
