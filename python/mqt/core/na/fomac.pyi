# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Reconstruction of NADevice from QDMI's Device class."""

from collections.abc import Mapping
from typing import Any

__all__ = ["Device"]

class Device:
    """Represents a device with a lattice of traps."""

    class Vector:
        """Represents a 2D vector."""

        def dict(self) -> Mapping[str, int]:
            """Returns a dictionary representation of the vector."""

    class Region:
        """Represents a region in the device."""

        class Size:
            """Represents the size of a."""

            def dict(self) -> Mapping[str, int]:
                """Returns a dictionary representation of the size."""

        def dict(self) -> Mapping[str, Mapping[str, int]]:
            """Returns a dictionary representation of the region."""

    class Lattice:
        """Represents a lattice of traps in the device."""

        def dict(self) -> Mapping[str, Any]:
            """Returns a dictionary representation of the lattice."""

    class Unit:
        """Represents a unit of measurement for length and time."""

        def dict(self) -> Mapping[str, Any]:
            """Returns a dictionary representation of the unit."""

    class DecoherenceTimes:
        """Represents the decoherence times of the device."""

        def dict(self) -> Mapping[str, Any]:
            """Returns a dictionary representation of the decoherence times."""

    def dict(self) -> Mapping[str, Any]:
        """Returns a dictionary representation of the NADevice."""
