# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Device-specific Qiskit gates used by QDMI backends.

This module hosts gate classes that are not part of Qiskit's standard gate
library but are required to represent device-native operations (e.g. IQM's
``move`` gate) in a Qiskit :class:`~qiskit.transpiler.Target`.
"""

from __future__ import annotations

from qiskit.circuit import Gate

__all__ = ["MoveGate"]


def __dir__() -> list[str]:
    return __all__


class MoveGate(Gate):
    """MOVE gate for IQM devices.

    The MOVE gate transfers the state of a qubit into a computational
    resonator (or back to the qubit), as used by IQM's star-topology
    architectures. The gate is intentionally kept opaque so Qiskit does not
    attempt to decompose it.
    """

    def __init__(self, label: str | None = None) -> None:
        """Initialize the MOVE gate.

        Args:
            label: Optional gate label.
        """
        super().__init__("move", 2, [], label=label)
