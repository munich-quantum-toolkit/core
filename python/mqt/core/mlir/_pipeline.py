# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Pipeline helpers that wrap the MQT MLIR C++ compiler via nanobind."""

from __future__ import annotations

from mqt.core import mlir as _ext  # the nanobind extension


def load_qasm(qasm: str) -> str:
    """Parse an OpenQASM string and return the QC dialect MLIR module as text.

    Args:
        qasm: OpenQASM 2.0 or 3.0 program string.

    Returns:
        The QC dialect MLIR module as a string.
    """
    return _ext.load_qasm(qasm)  # type: ignore[attr-defined]


def convert_qc_to_qco(mlir_text: str) -> str:
    """Convert a QC dialect MLIR module to the QCO dialect.

    Args:
        mlir_text: A QC dialect MLIR module as a string (from :func:`load_qasm`).

    Returns:
        The QCO dialect MLIR module as a string.
    """
    return _ext.convert_qc_to_qco(mlir_text)  # type: ignore[attr-defined]


def compile_program(qasm: str, *, convert_to_qir: bool = False) -> str:
    """Run the full MQT compiler pipeline on an OpenQASM program.

    Mirrors the stages in ``QuantumCompilerPipeline``:
    QC → cleanup → QCO → cleanup → optimise → QCO cleanup → QC → cleanup
    and optionally QC → QIR → cleanup.

    Args:
        qasm: OpenQASM 2.0 or 3.0 program string.
        convert_to_qir: If True, lower the final result to QIR.

    Returns:
        The compiled MLIR module as a string.
    """
    return _ext.compile_program(qasm, convert_to_qir)  # type: ignore[attr-defined]
