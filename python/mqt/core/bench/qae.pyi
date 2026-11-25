# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Quantum amplitude estimation benchmarks and workflows."""

from collections.abc import Callable, Mapping, Sequence

import mqt.core.bench
import mqt.core.mlir

class Problem:
    """State preparation and oracle for quantum amplitude estimation."""

    def __init__(self, *, good_states: Sequence[str], amplitude: float) -> None: ...
    @property
    def good_states(self) -> list[str]:
        """The nonempty set of distinct big-endian good states. All have the same width (1-63 bits), and at least one basis state must be bad."""

    @property
    def amplitude(self) -> float:
        """The total good-state probability in [0, 1] before amplification."""

class Options:
    """Parameters for one QAE circuit.

    Supply exactly one of precision and iterations.
    """

    def __init__(self, *, problem: Problem, precision: int | None = None, iterations: int | None = None) -> None: ...
    @property
    def problem(self) -> Problem:
        """The state preparation and good-state set."""

    @property
    def precision(self) -> int | None:
        """The number of phase bits (1-31) for canonical QAE."""

    @property
    def iterations(self) -> int | None:
        """The Grover power from 0 through :math:`2^{31}-1` for an amplification experiment."""

class QAE:
    """A validated quantum amplitude estimation benchmark.

    State preparation distributes each class's probability uniformly over its basis
    states. A phase-estimation circuit returns the phase bits. An amplification
    circuit returns the data bits for classification against the good-state set.
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
    def from_instance_specification_json(json: str, *, source: str = "<instance-specification>") -> QAE:
        """Parse a strict benchmark instance specification."""

    @staticmethod
    def from_manifest_json(json: str, *, source: str = "<manifest>") -> QAE:
        """Parse a strict benchmark manifest."""

def phase_estimation(problem: Problem, run: Callable[[QAE], Mapping[str, int]], *, precision: int) -> float:
    """Estimate the good-state probability using quantum phase estimation.

    The callback takes a QAE instance with the requested precision (1-31 bits) and
    returns its counts. The estimate uses the most frequent phase bin after
    combining opposite phases. Ties select the lower probability.
    """

def maximum_likelihood(problem: Problem, run: Callable[[QAE], Mapping[str, int]], *, rounds: int) -> float:
    """Estimate the good-state probability by maximizing the joint binomial likelihood.

    The callback takes a QAE instance and returns its counts. Choose 1-31 rounds.
    The Grover powers are 1 through ``2**(rounds - 1)`` in powers of two. Each
    experiment may use a different number of shots. A single round can give
    multiple equally likely estimates.
    """

def iterative(
    problem: Problem,
    run: Callable[[QAE], Mapping[str, int]],
    *,
    epsilon: float,
    confidence: float = 0.95,
    max_batches: int = 100,
) -> tuple[float, float]:
    """Estimate a probability interval by adapting the Grover power to measured counts.

    The callback takes a QAE instance and returns an independent, fixed-size batch
    of shots. The midpoint of the returned bounds estimates the probability within
    epsilon at the requested confidence under ideal sampling. Choose epsilon in
    (0, 0.5), confidence in (0, 1), and a positive batch budget.

    Raises:
        ValueError: The inputs are invalid, the confidence intervals are disjoint,
            or the batch budget is exhausted.
    """
