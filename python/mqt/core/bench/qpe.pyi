# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""QPE benchmark instances and options."""

import enum
import fractions
from collections.abc import Mapping

import mqt.core.bench
import mqt.core.mlir

class Phase:
    """An exact phase in turns modulo one turn."""

    def __init__(self, *, numerator: int, denominator: int) -> None: ...
    @property
    def numerator(self) -> int:
        """The reduced numerator."""

    @property
    def denominator(self) -> int:
        """The reduced denominator."""

class Method(enum.Enum):
    """Full-register or iterative measurement-and-feed-forward method."""

    STANDARD = 0

    ITERATIVE = 1

class Options:
    """Parameters for a QPE benchmark."""

    def __init__(self, *, precision: int, phase: fractions.Fraction | Phase, method: Method = ...) -> None: ...
    @property
    def precision(self) -> int:
        """The number of measured phase bits."""

    @property
    def phase(self) -> fractions.Fraction:
        """The reduced phase in turns."""

    @property
    def method(self) -> Method:
        """The circuit method."""

class QPE:
    """A validated QPE benchmark."""

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
    def from_instance_specification_json(json: str, *, source: str = "<instance-specification>") -> QPE:
        """Parse a strict benchmark instance specification."""

    @staticmethod
    def from_manifest_json(json: str, *, source: str = "<manifest>") -> QPE:
        """Parse a strict benchmark manifest."""
