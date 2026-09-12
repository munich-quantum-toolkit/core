# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Concatenated 15-to-1 magic-state distillation."""

from collections.abc import Mapping

import mqt.core.bench
import mqt.core.mlir

class Options:
    """Parameters for concatenated 15-to-1 magic-state distillation."""

    def __init__(self, *, levels: int = 1) -> None: ...
    @property
    def levels(self) -> int:
        """Concatenated levels in [1, 4], using 15**levels qubits."""

class MagicStateDistillation:
    """Concatenated 15-to-1 Reed--Muller distillation.

    Inputs are ideal :math:`|T\\rangle = T|+\\rangle` states.
    Bit 1 flags any rejected block; bit 0 checks the retained root state in the T
    basis. Ideal output is ``00``. Each level consumes the preceding level's
    retained quantum outputs, using exactly ``15**levels`` qubits.
    """

    def __init__(self, options: Options = ...) -> None: ...
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
    def from_instance_specification_json(
        json: str, *, source: str = "<instance-specification>"
    ) -> MagicStateDistillation:
        """Parse a strict benchmark instance specification."""

    @staticmethod
    def from_manifest_json(json: str, *, source: str = "<manifest>") -> MagicStateDistillation:
        """Parse a strict benchmark manifest."""
