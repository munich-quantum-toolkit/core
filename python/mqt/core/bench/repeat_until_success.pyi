# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Repeat-until-success benchmark instance."""

from collections.abc import Mapping

import mqt.core.bench
import mqt.core.mlir

class Options:
    """Parameters for a Pauli-string repeat-until-success benchmark."""

    def __init__(self, *, data_qubits: int = 1) -> None: ...
    @property
    def data_qubits(self) -> int:
        """The number of data qubits, excluding the ancilla."""

class RepeatUntilSuccess:
    """Apply a repeat-until-success implementation of :math:`(I + i\\sqrt{2}X^{\\otimes n}) / \\sqrt{3}`.

    Each attempt measures an ancilla prepared from :math:`|0\\rangle` and retries on
    failure. After success, return the parity of a :math:`Y \\otimes X^{\\otimes(n-1)}`
    measurement on the data qubits.
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
    def from_instance_specification_json(json: str, *, source: str = "<instance-specification>") -> RepeatUntilSuccess:
        """Parse a strict benchmark instance specification."""

    @staticmethod
    def from_manifest_json(json: str, *, source: str = "<manifest>") -> RepeatUntilSuccess:
        """Parse a strict benchmark manifest."""
