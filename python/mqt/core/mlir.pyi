# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""MQT Core MLIR - Python bindings for the MQT MLIR compiler collection."""

def load_qasm(qasm: str) -> str:
    """Parse an OpenQASM string and return the QC dialect MLIR module as text.

    Args:
        qasm: OpenQASM 2.0 or 3.0 program string.

    Returns:
        The QC dialect MLIR module as a string.
    """

def convert_qc_to_qco(mlir_text: str) -> str:
    """Convert a QC dialect module (as text) to QCO dialect.

    Args:
        mlir_text: A QC dialect MLIR module as a string (from :func:`load_qasm`).

    Returns:
        The QCO dialect MLIR module as a string.
    """

def compile_program(qasm: str, convert_to_qir: bool = False) -> str:
    """Run the full compiler pipeline on an OpenQASM program.

    Args:
        qasm: OpenQASM 2.0 or 3.0 program string.
        convert_to_qir: If True, lower the final result to QIR.

    Returns:
        The compiled MLIR module as a string.
    """
