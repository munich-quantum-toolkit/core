# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Qiskit target conversion for the MQT Compiler Collection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qiskit.providers import BackendV2
from qiskit.transpiler import Target

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ...mlir import CompilerTarget

__all__ = ["compiler_target_from_qiskit"]


def compiler_target_from_qiskit(
    source: Target | BackendV2,
    *,
    operation_names: Iterable[str] | None = None,
    name: str | None = None,
) -> CompilerTarget:
    """Snapshot standard operations and connectivity from a Qiskit target.

    Preserve standard gate names and ordered sites, including directional
    two-qubit gates. Target-aware Qiskit export also supports legacy u1/u3 names.
    Connectivity comes from two-qubit operation sites and is undirected.
    Global phase is circuit metadata and is always permitted.
    Gate recognition uses the same adapter and Qiskit versions as circuit import/export.
    Custom names, unsupported gates, fixed parameters, and restricted angles
    are omitted with a warning unless explicitly selected. Parameter expressions
    are unrestricted slots, following Qiskit's target-matching semantics.
    Delay, barrier, control flow, and empty applicability are ignored by default
    and rejected when explicitly selected. Timing, calibration, and scheduling
    properties are not transferred. This does not assert device support for
    classical control flow or guarantee that a program can be compiled.
    Invalid operation selections, widths, or connectivity raise :class:`ValueError`.

    Args:
        source: A target with a known positive qubit count, or a BackendV2.
        operation_names: Restrict the snapshot to these target operation names.
            By default, retain every representable operation. Explicitly selected
            operations must all be representable.
        name: Optional name of the compiler target; defaults to the backend name.

    Returns:
        An immutable compiler target independent of subsequent source changes.

    Raises:
        TypeError: The source is not a Target or BackendV2.
    """
    from ...mlir import (  # ruff: ignore[import-outside-top-level] Keep MLIR optional for QDMI-only use.
        import_target,
    )

    if isinstance(source, BackendV2):
        if name is None:
            name = source.name
        source = source.target
    if not isinstance(source, Target):
        msg = "Expected a Qiskit Target or BackendV2"
        raise TypeError(msg)
    return import_target(source, operation_names=operation_names, name=name)
