# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Typed benchmark instances and analytic references."""

from mqt.core.bench import bv as bv
from mqt.core.bench import ghz as ghz
from mqt.core.bench import grover as grover
from mqt.core.bench import multiplexer as multiplexer
from mqt.core.bench import qft as qft
from mqt.core.bench import qpe as qpe
from mqt.core.bench import teleportation as teleportation

class Output:
    """One logical classical output register."""

    @property
    def name(self) -> str:
        """The register name."""

    @property
    def width(self) -> int:
        """The number of big-endian outcome bits."""

class Evaluation:
    """The result of comparing counts with a reference."""

    @property
    def total_variation_distance(self) -> float:
        """The total variation distance from the ideal distribution."""

    @property
    def squared_hellinger_fidelity(self) -> float:
        """The squared Hellinger fidelity with the ideal distribution."""

    @property
    def success_probability(self) -> float | None:
        """The observed success probability, when defined."""
