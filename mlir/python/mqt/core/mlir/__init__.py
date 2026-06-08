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

from ._mlir_libs._mqtCoreMlir import compile_qasm, load_qasm
from ._pipeline import CompilationResult, MQTContext, compile_program, compile_with_record, qc_to_qco, register_dialects

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
