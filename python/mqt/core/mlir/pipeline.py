# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""The MQT Compiler Collection pipeline exposed through MLIR's Python bindings.

OpenQASM 3 programs are imported into the QC dialect, and any registered MQT
pass or pass pipeline can be run on the resulting :class:`mlir.ir.Module` via
:func:`run_pipeline`. The heavy lifting (parsing, conversions, optimizations)
runs in C++.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mlir.ir import Context  # ty: ignore[unresolved-import]
from mlir.passmanager import PassManager  # ty: ignore[unresolved-import]

from ._mlir import (  # ty: ignore[unresolved-import]
    import_qasm3_to_qc,
    register_dialects,
    register_passes,
)

if TYPE_CHECKING:
    from mlir.ir import Module  # ty: ignore[unresolved-import]

__all__ = [
    "QASMProgram",
    "create_context",
    "read_qasm",
    "run_pipeline",
    "transform_to_qco",
    "translate_to_qc",
]

# Passes are registered with MLIR's global registry exactly once per process.
register_passes()


@dataclass(frozen=True)
class QASMProgram:
    """An in-memory OpenQASM 3 program (the ``py:qasm`` stage of the pipeline).

    Attributes:
        source: The OpenQASM 3 source code.
    """

    source: str


def create_context() -> Context:
    """Create an MLIR context with all MQT Compiler Collection dialects registered.

    Returns:
        A fresh :class:`mlir.ir.Context` with the QC, QCO, QTensor, Jeff, and
        supporting dialects registered and loaded.
    """
    context = Context()
    register_dialects(context)
    return context


def read_qasm(source: str | Path) -> QASMProgram:
    """Read an OpenQASM 3 program (the ``py:qasm`` stage of the pipeline).

    Args:
        source: Either an OpenQASM 3 source string or a path to a ``.qasm``
            file.

    Returns:
        The loaded program as a :class:`QASMProgram`.
    """
    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8")
    elif "\n" not in source and source.endswith(".qasm") and Path(source).is_file():
        text = Path(source).read_text(encoding="utf-8")
    else:
        text = source
    return QASMProgram(source=text)


def translate_to_qc(program: QASMProgram | str, context: Context | None = None) -> Module:
    """Translate an OpenQASM 3 program to the QC dialect (``mlir:qc`` stage).

    Args:
        program: The program to translate, as returned by :func:`read_qasm`, or
            a raw OpenQASM 3 source string.
        context: The MLIR context to own the resulting module. If ``None``, a
            new context with all dialects registered is created.

    Returns:
        The quantum program as an :class:`mlir.ir.Module` in the QC dialect.
    """
    qasm = program.source if isinstance(program, QASMProgram) else program
    if context is None:
        context = create_context()
    return import_qasm3_to_qc(context, qasm)


def run_pipeline(module: Module, pipeline: str) -> Module:
    """Run a textual MLIR pass pipeline on a module in place.

    Any pass of the MQT Compiler Collection (e.g. ``qc-to-qco``,
    ``hadamard-lifting``, ``qc-to-qir``) can be used, as well as their
    combinations.

    Args:
        module: The :class:`mlir.ir.Module` to transform.
        pipeline: A textual pass pipeline, e.g. ``"builtin.module(qc-to-qco)"``.

    Returns:
        The same module, transformed by the pipeline.
    """
    pass_manager = PassManager.parse(pipeline, context=module.context)
    pass_manager.run(module.operation)
    return module


def transform_to_qco(module: Module) -> Module:
    """Transform a QC-dialect module to the QCO dialect (``mlir:qco`` stage).

    Convenience wrapper around :func:`run_pipeline` with the ``qc-to-qco`` pass.

    Args:
        module: A QC-dialect :class:`mlir.ir.Module`, as returned by
            :func:`translate_to_qc`.

    Returns:
        The same module, transformed to the QCO dialect.
    """
    return run_pipeline(module, "builtin.module(qc-to-qco)")
