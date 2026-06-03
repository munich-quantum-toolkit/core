# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""High-level pipeline helpers built on top of the MLIR Python bindings."""

from __future__ import annotations

from mlir.ir import Context, Module
from mlir.passmanager import PassManager

from ._mlir_libs._mqtCoreMlir import register_dialects


def make_context() -> Context:
    """Return an MLIRContext with all MQT dialects registered and loaded."""
    ctx = Context()
    register_dialects(ctx)
    return ctx


def compile_qc_to_qco(mlir_text: str) -> str:
    """Run the qc-to-qco conversion pass on an MLIR module given as text.

    Args:
        mlir_text: MLIR module in the QC dialect, as a string.

    Returns:
        The resulting QCO dialect MLIR module as a string.
    """
    with make_context() as ctx:
        module = Module.parse(mlir_text, ctx)
        pm = PassManager.parse("builtin.module(qc-to-qco)", ctx)
        pm.run(module.operation)
        return str(module)
