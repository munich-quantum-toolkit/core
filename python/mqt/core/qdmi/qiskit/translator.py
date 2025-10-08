# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Operation translation registry and neutral IR builder.

This module provides a lightweight registry mapping canonical *device operation
names* to translator callables that convert a generic :class:`InstructionContext`
into one or more :class:`~mqt.core.qdmi.qiskit.types.ProgramInstruction` objects.

No Qiskit imports are performed here to keep the optional dependency isolated;
context objects are generic and may be constructed by higher layers from Qiskit
instructions (e.g., via :class:`~mqt.core.qdmi.qiskit.QiskitBackend`) or other
front ends.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .types import IRValidationError, ProgramInstruction, ProgramIR

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "InstructionContext",
    "OperationTranslator",
    "build_program_ir",
    "clear_operation_translators",
    "get_operation_translator",
    "list_operation_translators",
    "register_operation_translator",
    "unregister_operation_translator",
]

# ---------------------------------------------------------------------------
# Data structures & types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class InstructionContext:
    """Generic context describing a single frontend instruction.

    Attributes:
        name: Canonical operation name (lookup key for translator registry).
        qubits: Logical qubit indices the instruction acts on.
        params: Optional ordered real-valued parameters.
        clbits: Classical bit indices relevant for measurements (empty if not used).
        metadata: Arbitrary auxiliary metadata (frontend-specific hints).
    """

    name: str
    qubits: list[int]
    params: list[float] | None = None
    clbits: list[int] | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:  # structural validation
        """Validate context invariants after initialization.

        Raises:
            ValueError: If name is empty or any index is negative.
        """
        if not self.name:
            msg = "InstructionContext.name must be non-empty"
            raise ValueError(msg)
        if any(q < 0 for q in self.qubits):
            msg = "Qubit indices must be non-negative"
            raise ValueError(msg)
        if self.clbits and any(c < 0 for c in self.clbits):
            msg = "Classical bit indices must be non-negative"
            raise ValueError(msg)


OperationTranslator = Callable[[InstructionContext], Sequence[ProgramInstruction]]

# ---------------------------------------------------------------------------
# Internal registry
# ---------------------------------------------------------------------------

_TRANSLATORS: dict[str, OperationTranslator] = {}

# Some pseudo operations are *allowed* in IR validation without requiring a
# translator (they have no semantic effect or are handled specially by higher layers)
_PSEUDO_OPS_DEFAULT = {"barrier"}

# Measurement is treated as a real operation with a default translator producing
# a ProgramInstruction recording measurement metadata.


def _default_measure_translator(ctx: InstructionContext) -> Sequence[ProgramInstruction]:
    if ctx.clbits is None or len(ctx.clbits) != len(ctx.qubits):
        msg = "Measurement context must supply clbits with one-to-one mapping to qubits"
        raise IRValidationError(msg)
    # Encode mapping in metadata for later serialization.
    meta = {"clbits": ctx.clbits}
    return [ProgramInstruction(name="measure", qubits=ctx.qubits, params=None, metadata=meta)]


def _passthrough_translator(ctx: InstructionContext) -> Sequence[ProgramInstruction]:
    """Pass-through translator for operations that map directly to neutral IR.

    Args:
        ctx: Instruction context to translate.

    Returns:
        Single-element sequence containing the translated instruction.
    """
    return [ProgramInstruction(name=ctx.name, qubits=ctx.qubits, params=ctx.params, metadata=ctx.metadata or {})]


# Register built-in translators at import time (idempotent if re-imported)
_TRANSLATORS.setdefault("measure", _default_measure_translator)

# Register common single-qubit gates
for gate_name in ["x", "y", "z", "h", "i", "id", "s", "sdg", "t", "tdg", "sx", "sxdg"]:
    _TRANSLATORS.setdefault(gate_name, _passthrough_translator)

# Register common parametric single-qubit gates
for gate_name in ["rx", "ry", "rz", "p", "phase", "r", "prx", "u", "u2", "u3"]:
    _TRANSLATORS.setdefault(gate_name, _passthrough_translator)

# Register common two-qubit gates
for gate_name in ["cx", "cnot", "cy", "cz", "ch", "swap", "iswap", "dcx", "ecr"]:
    _TRANSLATORS.setdefault(gate_name, _passthrough_translator)

# Register common two-qubit parametric gates
for gate_name in ["rxx", "ryy", "rzz", "rzx", "xx_plus_yy", "xx_minus_yy"]:
    _TRANSLATORS.setdefault(gate_name, _passthrough_translator)

# ---------------------------------------------------------------------------
# Public registry API
# ---------------------------------------------------------------------------


def register_operation_translator(
    name: str,
    translator: OperationTranslator,
    *,
    overwrite: bool = False,
) -> None:
    """Register a translator for a canonical operation name.

    Args:
        name: Canonical operation identifier.
        translator: Callable producing one or more ProgramInstruction objects.
        overwrite: If False (default), attempting to replace an existing
            translator raises a ValueError.

    Raises:
        ValueError: If name already registered and ``overwrite`` is False.
    """
    if not name:
        msg = "Translator name must be non-empty"
        raise ValueError(msg)
    key = name.lower()
    if key in _TRANSLATORS and not overwrite:
        msg = f"Translator already registered for '{key}'"
        raise ValueError(msg)
    _TRANSLATORS[key] = translator


def unregister_operation_translator(name: str) -> None:
    """Remove a previously registered translator.

    Silently does nothing if the name is not present.
    """
    _TRANSLATORS.pop(name.lower(), None)


def get_operation_translator(name: str) -> OperationTranslator:
    """Return the translator for ``name`` (KeyError if missing)."""
    return _TRANSLATORS[name.lower()]


def list_operation_translators() -> list[str]:
    """Return a sorted list of registered translator operation names."""
    return sorted(_TRANSLATORS)


def clear_operation_translators(*, keep_defaults: bool = True) -> None:
    """Clear all registered translators.

    Args:
        keep_defaults: If True, re-register built-in translators after clearing.
    """
    _TRANSLATORS.clear()
    if keep_defaults:
        # Re-register all built-in translators
        _TRANSLATORS["measure"] = _default_measure_translator

        # Register common single-qubit gates
        for gate_name in ["x", "y", "z", "h", "i", "id", "s", "sdg", "t", "tdg", "sx", "sxdg"]:
            _TRANSLATORS[gate_name] = _passthrough_translator

        # Register common parametric single-qubit gates
        for gate_name in ["rx", "ry", "rz", "p", "phase", "r", "prx", "u", "u2", "u3"]:
            _TRANSLATORS[gate_name] = _passthrough_translator

        # Register common two-qubit gates
        for gate_name in ["cx", "cnot", "cy", "cz", "ch", "swap", "iswap", "dcx", "ecr"]:
            _TRANSLATORS[gate_name] = _passthrough_translator

        # Register common two-qubit parametric gates
        for gate_name in ["rxx", "ryy", "rzz", "rzx", "xx_plus_yy", "xx_minus_yy"]:
            _TRANSLATORS[gate_name] = _passthrough_translator


# ---------------------------------------------------------------------------
# IR Builder
# ---------------------------------------------------------------------------


def build_program_ir(
    *,
    name: str,
    instruction_contexts: Iterable[InstructionContext],
    allowed_operations: Iterable[str],
    num_qubits: int,
    pseudo_ops: Iterable[str] | None = None,
) -> ProgramIR:
    """Translate a sequence of instruction contexts into a validated ProgramIR.

    Each context is dispatched through the translator registry by its ``name``.
    Pseudo operations (structural no-ops) that appear in ``pseudo_ops`` (and the
    default pseudo ops set) are passed through directly as ProgramInstructions
    with empty metadata and params.

    Args:
        name: Program identifier.
        instruction_contexts: Sequence of :class:`InstructionContext` objects.
        allowed_operations: Device-supported operation names (capabilities layer).
        num_qubits: Number of qubits on the target device.
        pseudo_ops: Additional pseudo op names permitted (optional).

    Returns:
        A validated :class:`ProgramIR` instance.

    Raises:
        IRValidationError: If translator output violates structural constraints.
    """
    pseudo: set[str] = set(_PSEUDO_OPS_DEFAULT)
    if pseudo_ops:
        pseudo.update(pseudo_ops)

    instructions: list[ProgramInstruction] = []
    for ctx in instruction_contexts:
        key = ctx.name.lower()
        if ctx.name in pseudo:
            # structural pseudo op: represented directly
            instructions.append(ProgramInstruction(name=key, qubits=ctx.qubits, params=None, metadata={}))
            continue
        translator = _TRANSLATORS.get(key)
        if translator is None:
            msg = f"No translator registered for operation '{ctx.name}'"
            raise IRValidationError(msg)
        produced = translator(ctx)
        instructions.extend(list(produced))

    ir = ProgramIR(name=name, instructions=instructions, metadata={})
    ir.validate(
        num_qubits=num_qubits,
        allowed_operations=allowed_operations,
        pseudo_ops=pseudo,
    )
    return ir
