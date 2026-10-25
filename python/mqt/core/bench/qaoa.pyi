# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""QAOA benchmark instances and options."""

from collections.abc import Mapping

import mqt.core.bench
import mqt.core.mlir

class Options:
    """Parameters for a QAOA benchmark for MaxCut."""

    def __init__(self, *, qubits: int, depth: int = 1) -> None: ...
    @property
    def qubits(self) -> int:
        """The cycle's number of vertices, at least three and representable as a signed 64-bit integer."""

    @property
    def depth(self) -> int:
        """The number of cost-mixer layers."""

class Evaluation:
    """The sampled cut quality on a cycle graph."""

    @property
    def shots(self) -> int:
        """The number of evaluated shots."""

    @property
    def mean_cut(self) -> float:
        """The shot-weighted mean number of cut edges."""

    @property
    def approximation_ratio(self) -> float:
        """The mean cut divided by the cycle's maximum cut size."""

class QAOA:
    """A validated QAOA benchmark for MaxCut.

    The circuit starts in :math:`|+\\rangle^{\\otimes n}` on an unweighted cycle.
    Each layer applies :math:`e^{-i\\gamma_\\ell C}` followed by
    :math:`e^{-i\\beta_\\ell B}`, where
    :math:`C=\\sum_{j=0}^{n-1}(1-Z_jZ_{(j+1)\\bmod n})/2` and
    :math:`B=\\sum_{j=0}^{n-1} X_j`.

    For :math:`d` layers, the generated program takes two arrays of :math:`d`
    angles: ``gammas`` and ``betas``. Bind angles in radians with
    :meth:`mqt.core.mlir.QCProgram.bind_parameters` before sampling or export.
    Copy the program first to reuse it for other angles.

    The benchmark does not run an optimizer. Evaluation measures cut quality,
    not agreement with an ideal output distribution.
    """

    def __init__(self, options: Options) -> None: ...
    @property
    def options(self) -> Options:
        """The resolved benchmark parameters."""

    @property
    def output(self) -> mqt.core.bench.Output:
        """The logical output register."""

    def cut_size(self, outcome: str) -> int:
        """Count cycle edges whose measured endpoint bits differ."""

    def evaluate(self, counts: Mapping[str, int]) -> Evaluation:
        """Compute the mean cut and its ratio to the maximum cut size."""

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
    def from_instance_specification_json(json: str, *, source: str = "<instance-specification>") -> QAOA:
        """Parse a strict benchmark instance specification."""

    @staticmethod
    def from_manifest_json(json: str, *, source: str = "<manifest>") -> QAOA:
        """Parse a strict benchmark manifest."""
