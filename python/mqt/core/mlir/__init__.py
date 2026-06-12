# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Python bindings for the MQT Core MLIR compilation pipeline.

Exposes the (py:qasm) -> (mlir:qc) -> (mlir:qco) pipeline of the MQT
Compiler Collection to Python. The QC and QCO stages are returned as native
class: mlir.ir.Moduleobjects; the transformations themselves run in C++.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from mlir.ir import Context

from ._mqtCore import import_qasm3_to_qc, qc_to_qco, register_dialects

if TYPE_CHECKING:
    from mlir.ir import Module

__all__ = [
    "QASMProgram",
    "create_context",
    "read_qasm",
    "transform_to_qco",
    "translate_to_qc",
]


@dataclass(frozen=True)
class QASMProgram:
    """An in-memory OpenQASM 3 program (the py:qasm stage of the pipeline).

    Attributes:
        source: The OpenQASM 3 source code.
    """

    source: str


def create_context() -> Context:
    """Create an MLIR context with all MQT pipeline dialects registered.

    Returns:
        A fresh :class:`mlir.ir.Context` with the QC, QCO, and supporting
        dialects registered and loaded.
    """
    context = Context()
    register_dialects(context)
    return context


def read_qasm(source: str | Path) -> QASMProgram:
    """Read an OpenQASM 3 program (py:qasm stage of the pipeline).

    Args:
        source: Either an OpenQASM 3 source string or a path to a .qasm file.

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
    """Translate an OpenQASM 3 program to the QC dialect (mlir:qc stage).

    Args:
        program: The program to translate, as returned by :func:`read_qasm`, or
            a raw OpenQASM 3 source string.
        context: The MLIR context to own the resulting module. If None, a
            new context with all pipeline dialects registered is created.

    Returns:
        The quantum program as an :class: mlir.ir.Module in the QC dialect.
    """
    qasm = program.source if isinstance(program, QASMProgram) else program
    if context is None:
        context = create_context()
    return import_qasm3_to_qc(context, qasm)


def transform_to_qco(module: Module) -> Module:
    """Transform a QC-dialect module to the QCO dialect (mlir:qco stage).

    Runs the qc-to-qco conversion pass on the module in place.

    Args:
        module: A QC-dialect :class: mlir.ir.Module , as returned by
            :func: translate_to_qc.

    Returns:
        The same module, transformed to the QCO dialect.
    """
    return qc_to_qco(module)
