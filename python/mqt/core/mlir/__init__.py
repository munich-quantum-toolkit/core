# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Python bindings for the MQT Compiler Collection.

Exposes the dialects and passes of the MQT Compiler Collection to Python via
MLIR's Python bindings. OpenQASM 3 programs are imported into the QC dialect and
returned as native :class:`mlir.ir.Module` objects, which can then be run
through any MQT pass pipeline.
"""

from __future__ import annotations

from .pipeline import (
    QASMProgram,
    create_context,
    read_qasm,
    run_pipeline,
    transform_to_qco,
    translate_to_qc,
)

__all__ = [
    "QASMProgram",
    "create_context",
    "read_qasm",
    "run_pipeline",
    "transform_to_qco",
    "translate_to_qc",
]
