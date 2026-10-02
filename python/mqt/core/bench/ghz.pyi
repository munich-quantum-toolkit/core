# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""GHZ benchmark instances and options."""

import enum
from collections.abc import Mapping

import mqt.core.bench
import mqt.core.mlir

class Topology(enum.Enum):
    """Entangling topology for GHZ preparation."""

    LINEAR = 0

    STAR = 1

class Basis(enum.Enum):
    """Measurement basis for GHZ verification."""

    Z = 0

    X = 1

class Options:
    """Parameters for a GHZ benchmark."""

    def __init__(self, *, qubits: int, topology: Topology = Topology.LINEAR, basis: Basis = Basis.Z) -> None: ...
    @property
    def qubits(self) -> int:
        """The number of qubits."""

    @property
    def topology(self) -> Topology:
        """The entangling topology."""

    @property
    def basis(self) -> Basis:
        """The measurement basis."""

class GHZ:
    """A validated GHZ benchmark."""

    def __init__(self, options: Options) -> None: ...
    @property
    def options(self) -> Options:
        """The resolved benchmark parameters."""

    @property
    def output(self) -> mqt.core.bench.Output:
        """The logical output register."""

    def probability(self, outcome: str) -> float:
        """Return the ideal probability of an outcome."""

    def evaluate(self, counts: Mapping[str, int]) -> mqt.core.bench.Evaluation:
        """Compare sampled counts with the ideal distribution."""

    def generate(self) -> mqt.core.mlir.QCProgram:
        """Generate the benchmark as a QC program."""

    @property
    def instance_specification_json(self) -> str:
        """The canonical instance specification JSON."""

    @property
    def manifest_json(self) -> str:
        """The canonical manifest JSON."""

    @property
    def case_id(self) -> str:
        """The stable semantic case ID."""

    @staticmethod
    def from_instance_specification_json(json: str, *, source: str = "<instance-specification>") -> GHZ:
        """Parse a strict benchmark instance specification."""

    @staticmethod
    def from_manifest_json(json: str, *, source: str = "<manifest>") -> GHZ:
        """Parse a strict benchmark manifest."""
