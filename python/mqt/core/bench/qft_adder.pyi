# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""QFT adder benchmark instances and options."""

import enum
from collections.abc import Mapping

import mqt.core.bench
import mqt.core.mlir

class Method(enum.Enum):
    """How the addend enters the circuit."""

    REGISTER = 0

    CONSTANT = 1

class Overflow(enum.Enum):
    """Wrap the sum or retain carry."""

    WRAP = 0

    CARRY = 1

class Options:
    """Parameters for a QFT adder."""

    def __init__(
        self, *, addend: str, accumulator: str, method: Method = Method.REGISTER, overflow: Overflow = Overflow.WRAP
    ) -> None: ...
    @property
    def addend(self) -> str:
        """Big-endian addend; register inputs also accept ``+`` for a :math:`|+\\rangle` qubit."""

    @property
    def accumulator(self) -> str:
        """Binary accumulator with the same width as the addend."""

    @property
    def method(self) -> Method:
        """Register or constant addition."""

    @property
    def overflow(self) -> Overflow:
        """Wrap or carry behavior."""

class QFTAdder:
    """A validated QFT adder benchmark.

    Register addition uses controlled phases and returns the addend followed by the
    sum. Constant addition compiles the addend into phases and returns only the sum.
    Wrap mode computes :math:`(a + b) \\bmod 2^n`; carry mode retains one extra sum bit.
    All strings are big-endian, and leading zeros determine the operand width.

    Register addends may contain ``+`` for independent :math:`|+\\rangle` qubits.
    The accumulator and constant addends must be binary. The circuit follows
    Draper's register addition and its constant-input Fourier specialization.
    """

    def __init__(self, options: Options) -> None: ...
    @property
    def options(self) -> Options:
        """The resolved benchmark parameters."""

    @property
    def output(self) -> mqt.core.bench.Output:
        """The logical output register."""

    @property
    def expected_result(self) -> str | None:
        """The unique logical outcome, or ``None`` for a superposed addend."""

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
    def from_instance_specification_json(json: str, *, source: str = "<instance-specification>") -> QFTAdder:
        """Parse a strict benchmark instance specification."""

    @staticmethod
    def from_manifest_json(json: str, *, source: str = "<manifest>") -> QFTAdder:
        """Parse a strict benchmark manifest."""
