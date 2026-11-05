# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Export native compiler output through Qiskit's QPY serializer."""

from __future__ import annotations

__lazy_modules__ = {"argparse", "pathlib", "subprocess", "tempfile"}

import argparse
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def compile_qpy(arguments: list[str]) -> int | None:
    """Handle QPY output, or leave other formats to the native compiler.

    Returns:
        The export exit status, or None for native output formats.

    Raises:
        ValueError: If QPY serialization fails.
    """
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--emit", "-emit")
    parser.add_argument("-o", "--o", default="-")
    options, remaining = parser.parse_known_args(arguments[2:])
    for argument in remaining:
        name, _, value = argument.lstrip("-").partition("=")
        if (
            argument.startswith("-")
            and name in {"h", "help", "help-hidden", "help-list", "help-list-hidden", "version", "qdmi-list-devices"}
            and value.lower() not in {"false", "0"}
        ):
            return None
    if options.emit != "qpy" and not (options.emit is None and Path(options.o).suffix == ".qpy"):
        return None

    try:
        from qiskit import QiskitError  # ruff: ignore[import-outside-top-level] - Optional serializer.
    except ImportError:
        sys.stderr.write("QPY output requires Qiskit: install mqt-core[qiskit].\n")
        return 1
    from mqt.core.mlir import QCOProgram, _export_qpy  # ruff: ignore[import-outside-top-level] - Load bindings only for QPY.

    with TemporaryDirectory() as directory:
        intermediate = Path(directory) / "program.qco.mlir"
        result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] - Native executable, no shell.
            [*arguments[:2], "--emit=qco-optimized", "-o", str(intermediate), *remaining], check=False
        )
        if result.returncode:
            return result.returncode
        try:
            payload = _export_qpy(QCOProgram.from_mlir_file(intermediate).to_qc())
        except QiskitError as error:
            raise ValueError(str(error)) from error
        if options.o == "-":
            sys.stdout.buffer.write(payload)
        else:
            Path(options.o).write_bytes(payload)
    return 0
