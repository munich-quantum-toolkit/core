# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""VQE ansatz benchmark instances and options."""

import mqt.core.bench
import mqt.core.mlir

class Options:
    """Parameters for a VQE ansatz benchmark."""

    def __init__(self, *, qubits: int, depth: int = 1) -> None: ...
    @property
    def qubits(self) -> int:
        """The number of qubits, at least one."""

    @property
    def depth(self) -> int:
        """The number of rotation-entanglement layers."""

class VQEAnsatz:
    """A validated VQE ansatz benchmark.

    The RealAmplitudes circuit starts in :math:`|0\\rangle^{\\otimes n}`.
    Each layer applies :math:`R_y` rotations to all qubits, followed by
    :math:`\\mathrm{CX}(0,1), \\mathrm{CX}(1,2), \\ldots, \\mathrm{CX}(n-2,n-1)`
    in that order. A final layer of :math:`R_y` rotations follows the last
    entanglement layer.

    For :math:`d` layers, the generated program takes one array, ``thetas``,
    of :math:`n(d+1)` angles. Entry :math:`\\ell n+q` controls the rotation
    on qubit :math:`q` in rotation layer :math:`\\ell`. Bind angles in radians
    with :meth:`mqt.core.mlir.QCProgram.bind_parameters` before sampling or
    export. Copy the program first to reuse it for other angles.

    The benchmark supplies only the ansatz. It does not define a Hamiltonian,
    run an optimizer, or evaluate energies.
    """

    def __init__(self, options: Options) -> None: ...
    @property
    def options(self) -> Options:
        """The resolved benchmark parameters."""

    @property
    def output(self) -> mqt.core.bench.Output:
        """The logical output register."""

    @property
    def num_parameters(self) -> int:
        """The number of rotation angles, including the final layer."""

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
    def from_instance_specification_json(json: str, *, source: str = "<instance-specification>") -> VQEAnsatz:
        """Parse a strict benchmark instance specification."""

    @staticmethod
    def from_manifest_json(json: str, *, source: str = "<manifest>") -> VQEAnsatz:
        """Parse a strict benchmark manifest."""
