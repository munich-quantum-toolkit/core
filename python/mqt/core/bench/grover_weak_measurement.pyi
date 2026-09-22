# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Weak-measurement Grover benchmark instances and options."""

from collections.abc import Mapping

import mqt.core.bench
import mqt.core.mlir

class Options:
    """Parameters for a weak-measurement Grover benchmark."""

    def __init__(self, *, marked_bitstring: str, measurement_strength: float | None = None) -> None: ...
    @property
    def marked_bitstring(self) -> str:
        """The big-endian marked outcome."""

    @property
    def measurement_strength(self) -> float | None:
        """The :math:`\\kappa`-measurement strength, or ``None`` for :math:`2^{-n/2}`."""

class Grover:
    """A validated weak-measurement Grover benchmark.

    The benchmark prepares a uniform state and applies one Grover iteration before
    each :math:`\\kappa`-measurement. The measurement computes the predicate with
    :math:`O_\\chi`, applies the controlled :math:`R_\\kappa` rotation, uncomputes
    :math:`O_\\chi`, and measures the probe. An outcome of :math:`0` continues the
    loop; :math:`1` exits with the marked state.

    By default, the measurement strength is :math:`\\kappa=2^{-n/2}` for :math:`n`
    search qubits. The accepted range :math:`0<\\kappa\\leq 2^{-n/2}` follows the
    paper's robustness bound.
    """

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
