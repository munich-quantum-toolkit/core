# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Quantum-input QFT adder benchmark instances and options."""

from collections.abc import Mapping

import mqt.core.bench
import mqt.core.mlir

class Options:
    """Parameters for a quantum-input QFT adder benchmark."""

    def __init__(self, *, qubits: int) -> None: ...
    @property
    def qubits(self) -> int:
        """The number of qubits in each input register."""

class QFTAdderQuantum:
    """A QFT adder with an :math:`n`-qubit addend in :math:`|+\\rangle^{\\otimes n}` and an accumulator in :math:`|1\\rangle`.

    Big-endian outcomes concatenate the addend and sum, each with :math:`n` bits.
    For addend :math:`a`, the sum is :math:`(a + 1) \\bmod 2^n`; each valid
    outcome has probability :math:`2^{-n}`.

    Reference: https://arxiv.org/abs/quant-ph/0008033
    """

    def __init__(self, options: Options) -> None: ...
    @property
    def options(self) -> Options:
        """The resolved benchmark parameters."""

    @property
    def output(self) -> mqt.core.bench.Output:
        """The logical output register, with the addend followed by the sum."""

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
    def from_instance_specification_json(json: str, *, source: str = "<instance-specification>") -> QFTAdderQuantum:
        """Parse a strict benchmark instance specification."""

    @staticmethod
    def from_manifest_json(json: str, *, source: str = "<manifest>") -> QFTAdderQuantum:
        """Parse a strict benchmark manifest."""
