# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shor order finding and classical factor recovery."""

import enum
from collections.abc import Callable, Mapping

import mqt.core.bench
import mqt.core.mlir

class Options:
    """Parameters for semiclassical Shor order finding."""

    def __init__(self, *, number: int, base: int = 2, qft_cutoff: int | None = None) -> None: ...
    @property
    def number(self) -> int:
        """The odd modulus, at most 2**31 - 1."""

    @property
    def base(self) -> int:
        """The base, coprime to the modulus."""

    @property
    def qft_cutoff(self) -> int | None:
        """The largest controlled-rotation distance, or ``None`` for exact arithmetic."""

class Evaluation:
    """Verified factors recovered from measured phases."""

    @property
    def success_probability(self) -> float:
        """The fraction of shots that independently yield a verified factor pair."""

    @property
    def factors(self) -> tuple[int, int] | None:
        """A sorted factor pair, or ``None``."""

class Shor:
    """Semiclassical order finding with one reused query qubit."""

    def __init__(self, options: Options) -> None: ...
    @property
    def options(self) -> Options:
        """The resolved benchmark parameters."""

    @property
    def output(self) -> mqt.core.bench.Output:
        """The big-endian phase register with twice the modulus bit width."""

    def evaluate(self, counts: Mapping[str, int]) -> Evaluation:
        """Recover factors using exact continued fractions and verify them by division."""

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
    def from_instance_specification_json(json: str, *, source: str = "<instance-specification>") -> Shor:
        """Parse a strict benchmark instance specification."""

    @staticmethod
    def from_manifest_json(json: str, *, source: str = "<manifest>") -> Shor:
        """Parse a strict benchmark manifest."""

class FactorStatus(enum.Enum):
    """Outcome of a factoring workflow."""

    SUCCESS = 0

    PRIME = 1

    ATTEMPTS_EXHAUSTED = 2

class FactorResult:
    """Result of a bounded factoring workflow."""

    @property
    def status(self) -> FactorStatus: ...
    @property
    def factors(self) -> tuple[int, int] | None:
        """A verified sorted pair, or ``None``."""

    @property
    def attempts(self) -> int:
        """The number of attempted bases."""

def factor(
    number: int,
    run: Callable[[Shor], Mapping[str, int]],
    *,
    max_attempts: int = 16,
    seed: int = 0,
    qft_cutoff: int | None = None,
) -> FactorResult:
    """Find one nontrivial factor pair with a bounded number of attempted bases.

    The callback accepts a :class:`Shor` instance and returns counts. It owns device
    selection, shots, and execution seeds. Even numbers, primes, and perfect powers
    are handled classically. Callback failures and invalid counts propagate.
    """
