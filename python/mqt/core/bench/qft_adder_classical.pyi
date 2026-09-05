# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Classical-input QFT adder instances and options."""

from collections.abc import Mapping

import mqt.core.bench
import mqt.core.mlir

class Options:
    """Parameters for a classical-input QFT adder benchmark."""

    def __init__(self, *, addend: str) -> None: ...
    @property
    def addend(self) -> str:
        """The big-endian classical addend."""

class QFTAdderClassical:
    """A validated classical-input QFT adder benchmark."""

    def __init__(self, options: Options) -> None: ...
    @property
    def options(self) -> Options:
        """The resolved benchmark parameters."""

    @property
    def output(self) -> mqt.core.bench.Output:
        """The logical result register."""

    @property
    def expected_result(self) -> str:
        """The deterministic big-endian result."""

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
    def from_instance_specification_json(json: str, *, source: str = "<instance-specification>") -> QFTAdderClassical:
        """Parse a strict benchmark instance specification."""

    @staticmethod
    def from_manifest_json(json: str, *, source: str = "<manifest>") -> QFTAdderClassical:
        """Parse a strict benchmark manifest."""
