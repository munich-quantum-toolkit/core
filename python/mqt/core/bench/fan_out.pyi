# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Quantum fan-out benchmark instances and options."""

from collections.abc import Mapping

import mqt.core.bench
import mqt.core.mlir

class Options:
    """Parameters for a quantum fan-out benchmark."""

    def __init__(self, *, controls: int) -> None: ...
    @property
    def controls(self) -> int:
        """The number of control qubits, :math:`n\\in[1, 1073]`."""

class FanOut:
    """A validated quantum fan-out benchmark.

    The benchmark prepares uniform controls and a target in :math:`|+\\rangle`.
    Fan-out distributes controlled phase gates :math:`P(\\pi/2)` across the
    target and clean ancillas, then uncomputes the ancillas. A final Hadamard on
    the target makes the phase observable.

    For :math:`n` controls, the circuit uses :math:`2n` qubits. The logical
    output contains the controls followed by the target as its least-significant
    bit. For a control string of Hamming weight :math:`w`, the target is zero
    with conditional probability :math:`\\cos^2(\\pi w/4)`.
    """

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
    def from_instance_specification_json(json: str, *, source: str = "<instance-specification>") -> FanOut:
        """Parse a strict benchmark instance specification."""

    @staticmethod
    def from_manifest_json(json: str, *, source: str = "<manifest>") -> FanOut:
        """Parse a strict benchmark manifest."""
