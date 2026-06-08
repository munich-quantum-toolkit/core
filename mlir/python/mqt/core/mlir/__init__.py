# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Python bindings for the MQT MLIR compiler collection.

Typical usage::

    from mqt.core.mlir import compile_program, load_qasm, compile_qasm, compile_with_record

    # Convenience entry point
    result = compile_program(qasm_string)

    # Stage-by-stage
    qc_ir = load_qasm(qasm_string)  # QASM -> QC dialect
    result = compile_qasm(qasm_string)  # full pipeline -> final IR string

    # With all intermediate snapshots
    rec = compile_with_record(qasm_string)
    print(rec.after_qco_conversion)
"""

from __future__ import annotations

from ._mlir_libs._mqtCoreMlir import _register_dialects, compile_qasm, load_qasm
from ._pipeline import CompilationResult, MQTContext, compile_with_record, qc_to_qco


def register_dialects(context: object) -> None:
    """Register all MQT MLIR dialects into the given MLIR context.

    Args:
        context: An ``mlir.ir.Context`` instance to register dialects into.
    """
    _register_dialects(context)


def compile_program(
    qasm: str,
    *,
    convert_to_qir: bool = False,
    disable_merge_single_qubit_rotation_gates: bool = False,
    enable_hadamard_lifting: bool = False,
) -> str:
    """Push an OpenQASM program through the full MQT compiler pipeline.

    This is the main user-facing entry point. It accepts an OpenQASM string
    and returns the final compiled MLIR module as a text string.

    Args:
        qasm: OpenQASM 2/3 source string.
        convert_to_qir: Also lower to QIR at the end.
        disable_merge_single_qubit_rotation_gates: Skip quaternion-based
            single-qubit rotation gate merging.
        enable_hadamard_lifting: Apply Hadamard lifting during optimisation.

    Returns:
        The final compiled MLIR module as a string.
    """
    return compile_qasm(
        qasm,
        convert_to_qir=convert_to_qir,
        disable_merge_single_qubit_rotation_gates=disable_merge_single_qubit_rotation_gates,
        enable_hadamard_lifting=enable_hadamard_lifting,
    )


__all__ = [
    "CompilationResult",
    "MQTContext",
    "compile_program",
    "compile_qasm",
    "compile_with_record",
    "load_qasm",
    "qc_to_qco",
    "register_dialects",
]
