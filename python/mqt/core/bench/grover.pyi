# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Grover benchmark instances and options."""

from collections.abc import Mapping

import mqt.core.bench
import mqt.core.mlir

class Options:
    """Parameters for a Grover benchmark."""

    def __init__(self, *, marked_bitstring: str, iterations: int | None = None) -> None: ...
    @property
    def marked_bitstring(self) -> str:
        """The big-endian marked outcome."""

    @property
    def iterations(self) -> int | None:
        """The iteration count, or ``None`` for automatic selection."""

class Grover:
    """A validated single-solution Grover benchmark."""

    def __init__(self, options: Options) -> None: ...
    @property
    def options(self) -> Options:
        """The resolved benchmark parameters."""

    @property
    def output(self) -> mqt.core.bench.Output:
        """The logical output register."""

    @property
    def qubits(self) -> int:
        """The number of search qubits."""

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
    def from_instance_specification_json(json: str, *, source: str = "<instance-specification>") -> Grover:
        """Parse a strict benchmark instance specification."""

    @staticmethod
    def from_manifest_json(json: str, *, source: str = "<manifest>") -> Grover:
        """Parse a strict benchmark manifest."""
