# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Magic-state distillation benchmark instances and options."""

from collections.abc import Mapping

import mqt.core.bench
import mqt.core.mlir

class Options:
    """Parameters for concatenated 15-to-1 magic-state distillation."""

    def __init__(self, *, levels: int = 1) -> None: ...
    @property
    def levels(self) -> int:
        """Positive number of concatenated levels. The qubit count (five per level) must fit signed 64-bit circuit dimensions."""

class MagicStateDistillation:
    """A concatenated 15-to-1 magic-state distillation benchmark.

    Each block uses Litinski's five-qubit 15-to-1 circuit. The first level uses
    ideal :math:`\\pi/8` Pauli rotations; later levels implement these rotations
    with the preceding level's retained :math:`T^\\dagger|+\\rangle` states.
    The circuit uses ``5 * levels`` qubits and executes ``15**levels`` leaf
    rotations per shot. Backend and resource limits apply.

    Bit 1 flags any rejected block; bit 0 checks the retained root state against
    :math:`T^\\dagger|+\\rangle`. Ideal output is ``00``. All blocks execute once,
    regardless of rejection. The benchmark models ideal logical circuits without
    input noise, physical error correction, or retries.
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
