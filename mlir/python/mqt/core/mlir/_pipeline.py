# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""High-level Python API for the MQT MLIR compiler collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mlir.ir import Context, Module
from mlir.passmanager import PassManager

from ._mlir_libs._mqtCoreMlir import _register_dialects, compile, load_qasm

if TYPE_CHECKING:
    pass


@dataclass
class CompilationResult:
    """Snapshots of the MLIR module at every compilation stage.

    Returned by :func:`compile_with_record`. Each field holds the MLIR text
    printed immediately after that stage completes. Fields for optional stages
    (QIR conversion and cleanup) are empty strings when *convert_to_qir* is
    ``False``.
    """

    result: str
    after_qc_import: str = field(default="")
    after_initial_canon: str = field(default="")
    after_qco_conversion: str = field(default="")
    after_qco_canon: str = field(default="")
    after_optimization: str = field(default="")
    after_optimization_canon: str = field(default="")
    after_qc_conversion: str = field(default="")
    after_qc_canon: str = field(default="")
    after_qir_conversion: str = field(default="")
    after_qir_canon: str = field(default="")


class MQTContext(Context):
    """An MLIR context with all MQT dialects pre-registered.

    Use as a context manager::

        with MQTContext() as ctx:
            module = Module.parse(mlir_text, ctx)
    """

    def __init__(self) -> None:
        super().__init__()
        _register_dialects(self)


def qc_to_qco(mlir_text: str) -> str:
    """Apply the QC-to-QCO conversion pass to an already-loaded QC module.

    Args:
        mlir_text: MLIR module in the QC dialect (as returned by
            :func:`load_qasm`).

    Returns:
        The QCO-dialect MLIR module as a string.
    """
    with MQTContext() as ctx:
        module = Module.parse(mlir_text, ctx)
        PassManager.parse("builtin.module(qc-to-qco)", ctx).run(
            module.operation
        )
        return str(module)


def compile_with_record(
    qasm: str,
    *,
    convert_to_qir: bool = False,
    disable_merge_single_qubit_rotation_gates: bool = False,
    enable_hadamard_lifting: bool = False,
) -> CompilationResult:
    """Run the full compiler pipeline and capture every intermediate stage.

    Args:
        qasm: OpenQASM 2/3 source string.
        convert_to_qir: Also lower to QIR at the end.
        disable_merge_single_qubit_rotation_gates: Skip quaternion-based
            single-qubit rotation gate merging.
        enable_hadamard_lifting: Apply Hadamard lifting during optimisation.

    Returns:
        A :class:`CompilationResult` with the final IR and all stage snapshots.
    """
    stages: dict[str, str] = compile(  # type: ignore[assignment]
        qasm,
        convert_to_qir=convert_to_qir,
        disable_merge_single_qubit_rotation_gates=disable_merge_single_qubit_rotation_gates,
        enable_hadamard_lifting=enable_hadamard_lifting,
        capture_intermediates=True,
    )
    return CompilationResult(
        result=stages["result"],
        after_qc_import=stages["after_qc_import"],
        after_initial_canon=stages["after_initial_canon"],
        after_qco_conversion=stages["after_qco_conversion"],
        after_qco_canon=stages["after_qco_canon"],
        after_optimization=stages["after_optimization"],
        after_optimization_canon=stages["after_optimization_canon"],
        after_qc_conversion=stages["after_qc_conversion"],
        after_qc_canon=stages["after_qc_canon"],
        after_qir_conversion=stages["after_qir_conversion"],
        after_qir_canon=stages["after_qir_canon"],
    )
