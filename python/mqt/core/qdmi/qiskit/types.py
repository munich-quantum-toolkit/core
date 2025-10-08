# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Neutral program IR types for the QDMI Qiskit backend.

Lightweight, Qiskit-independent dataclasses representing a neutral program
used by the optional backend layer. Includes simple structural validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "IRValidationError",
    "ProgramIR",
    "ProgramInstruction",
]


class IRValidationError(ValueError):
    """Raised when an IR consistency or capability validation check fails."""


@dataclass(slots=True)
class ProgramInstruction:
    """A single neutral instruction.

    Attributes:
        name: Canonical operation name (device op or pseudo op like 'barrier' or
            'measure').
        qubits: Logical qubit indices the operation acts upon.
        params: Optional ordered parameter list (float values) or None.
        metadata: Optional free-form mapping for translators / serializers.
    """

    name: str
    qubits: list[int]
    params: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate structural invariants after dataclass initialization.

        Raises:
            IRValidationError: If the instruction name is empty or a qubit index is
                negative.

        Note:
            Type constraints (e.g., params being a list or None) are enforced
            statically by type checkers, i.e., mypy.
        """
        if not self.name:
            msg = "Instruction name must be non-empty"
            raise IRValidationError(msg)
        if any(q < 0 for q in self.qubits):
            msg = "Qubit indices must be non-negative"
            raise IRValidationError(msg)


@dataclass(slots=True)
class ProgramIR:
    """Neutral intermediate representation for a (single) program submission."""

    name: str
    instructions: list[ProgramInstruction]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(
        self,
        *,
        num_qubits: int,
        allowed_operations: Iterable[str],
        pseudo_ops: Iterable[str] | None = None,
    ) -> None:
        """Validate IR structure against device capabilities.

        Args:
            num_qubits: Number of qubits available on the device.
            allowed_operations: Set / iterable of device-supported operation names.
            pseudo_ops: Additional operation names permitted (e.g., 'barrier', 'measure').

        Raises:
            IRValidationError: On any structural mismatch.
        """
        allowed = set(allowed_operations)
        if pseudo_ops:
            allowed.update(pseudo_ops)
        for idx, inst in enumerate(self.instructions):
            # name
            if inst.name not in allowed:
                msg = f"Instruction {idx} uses unsupported operation '{inst.name}'"
                raise IRValidationError(msg)
            # qubit indices range
            for q in inst.qubits:
                if q >= num_qubits:
                    msg = f"Instruction {idx} qubit index {q} exceeds available qubits {num_qubits}"
                    raise IRValidationError(msg)
            # parameter list consistency optional (deferred until translator supplies expected arity)
