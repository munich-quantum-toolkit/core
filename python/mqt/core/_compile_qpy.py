# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compile QPY through the Python compiler API."""

from __future__ import annotations

__lazy_modules__ = {"argparse", "pathlib"}

import argparse
import os
import sys
from pathlib import Path


def _unsigned(value: str) -> int:
    """Parse a compiler option with an unsigned 64-bit range.

    Returns:
        The parsed integer.

    Raises:
        argparse.ArgumentTypeError: If the spelling or range is invalid.
    """
    if not value.isascii() or value[:1] in {"+", "-"} or "_" in value or value != value.strip():
        msg = "expected an unsigned integer"
        raise argparse.ArgumentTypeError(msg)
    number = int(value, 8 if len(value) > 1 and value[0] == "0" and value[1].isdigit() else 0)
    if not 0 <= number < 2**64:
        msg = "expected an unsigned 64-bit integer"
        raise argparse.ArgumentTypeError(msg)
    return number


def compile_qpy(arguments: list[str]) -> int | None:
    """Handle QPY output, or leave other formats to the native compiler.

    Returns:
        The export exit status, or None for native output formats.

    Raises:
        ValueError: If compilation or QPY serialization fails.
    """
    parser = argparse.ArgumentParser(prog="mqt-core compile", add_help=False, allow_abbrev=False)
    parser.add_argument("--emit", "-emit")
    parser.add_argument("-o", "--o", default="-")
    options, remaining = parser.parse_known_args(arguments[2:])
    if any(
        argument.lstrip("-")
        in {"h", "help", "help-hidden", "help-list", "help-list-hidden", "version", "qdmi-list-devices"}
        for argument in remaining
        if argument.startswith("-")
    ):
        return None
    if options.emit != "qpy" and not (options.emit is None and Path(options.o).suffix == ".qpy"):
        return None

    parser.add_argument("input", nargs="?", default="-")
    parser.add_argument("--input-format", "-input-format", choices=["auto", "openqasm", "mlir", "jeff"], default="auto")
    targets = parser.add_mutually_exclusive_group()
    targets.add_argument("--target", "-target")
    targets.add_argument("--qdmi-device", "-qdmi-device")
    parser.add_argument("--payload-spec", "-payload-spec")
    parser.add_argument("--qdmi-config", "-qdmi-config")
    parser.add_argument("--qdmi-manifest", "-qdmi-manifest", action="append", default=[])
    parser.add_argument("--pass-pipeline", "-pass-pipeline", "--passes", "-passes")
    parser.add_argument(
        "--decompose-multi-controlled-min-qubits", "-decompose-multi-controlled-min-qubits", type=_unsigned, default=3
    )
    parser.add_argument("--seed", "-seed", type=_unsigned)
    for name in ("trials", "iterations", "lookahead", "search-memory-limit"):
        parser.add_argument(f"--mapping-{name}", f"-mapping-{name}", type=_unsigned)
    boolean_options = ("decompose-multi-controlled", "time-passes", "mlir-pass-statistics")
    for name in boolean_options:
        parser.add_argument(f"--{name}", f"-{name}", action=argparse.BooleanOptionalAction, default=False)
    for index, argument in enumerate(remaining):
        name, separator, value = argument.partition("=")
        if separator and name.lstrip("-") in boolean_options:
            value = value.lower()
            if value not in {"", "true", "false", "0", "1"}:
                parser.error(f"{name} expects true, false, 0, or 1")
            prefix = "no-" if value in {"false", "0"} else ""
            remaining[index] = f"--{prefix}{name.lstrip('-')}"
    options = parser.parse_args(remaining, namespace=options)

    has_target = options.target is not None or bool(options.qdmi_device)
    mapping = {
        name: value
        for name in ("trials", "iterations", "lookahead", "search_memory_limit")
        if (value := getattr(options, f"mapping_{name}")) is not None
    }
    if mapping and not has_target:
        parser.error("mapping controls require --qdmi-device or --target")
    if mapping.get("trials") == 0:
        parser.error("--mapping-trials must be greater than zero")
    if (options.target is not None and not options.payload_spec) or (options.payload_spec and not has_target):
        parser.error("--qdmi-device or --target and --payload-spec must be provided together")
    if has_target and (options.pass_pipeline is not None or options.decompose_multi_controlled):
        parser.error("target compilation cannot be combined with --pass-pipeline or --decompose-multi-controlled")
    if options.decompose_multi_controlled and options.decompose_multi_controlled_min_qubits < 3:
        parser.error("--decompose-multi-controlled-min-qubits must be at least 3")
    if options.qdmi_config:
        if not options.qdmi_device:
            parser.error("--qdmi-config requires --qdmi-device")
        os.environ["MQT_CORE_QDMI_CONFIG_FILE"] = options.qdmi_config

    pipeline = "mqt-qco-default"
    if options.pass_pipeline is not None:
        anchor, _, body = options.pass_pipeline.strip().partition("(")
        if anchor.rstrip() != "builtin.module" or not body.endswith(")"):
            parser.error("--pass-pipeline must be anchored on builtin.module")
        pipeline = body[:-1]
    elif options.decompose_multi_controlled:
        pipeline = (
            f"decompose-multi-controlled{{min-qubits={options.decompose_multi_controlled_min_qubits}}},mqt-qco-default"
        )

    try:
        from qiskit import QiskitError  # ruff: ignore[import-outside-top-level] - Optional QPY dependency.
    except ImportError:
        sys.stderr.write("QPY output requires Qiskit: install mqt-core[qiskit].\n")
        return 1
    from mqt.core import mlir  # ruff: ignore[import-outside-top-level] - Load bindings only for QPY.
    from mqt.core.qdmi import builtin_driver  # ruff: ignore[import-outside-top-level] - Configure before device access.

    for manifest in options.qdmi_manifest:
        builtin_driver.add_manifest(manifest)
    target = options.qdmi_device or None
    if options.payload_spec:
        compiler_target = (
            mlir.CompilerTarget.from_mlir_str(options.target)
            if options.target is not None
            else mlir.CompilerTarget.from_device(builtin_driver.open_device(options.qdmi_device))
        )
        target = mlir.TargetEnvironment(compiler_target, mlir.PayloadSpecification.from_mlir_str(options.payload_spec))

    path = Path(options.input)
    input_format = options.input_format
    if input_format == "auto":
        input_format = (
            "mlir" if options.input == "-" else {".qasm": "openqasm", ".mlir": "mlir", ".jeff": "jeff"}.get(path.suffix)
        )
    if input_format == "openqasm":
        program = mlir.QCProgram.from_openqasm_file(path)
    elif input_format == "mlir":
        program = mlir.Program.from_mlir_file(path)
    elif input_format == "jeff":
        if options.input == "-":
            parser.error("reading jeff from standard input is not supported")
        program = mlir.JeffProgram.from_file(path)
    else:
        parser.error("could not determine the input format; use --input-format")

    try:
        payload = mlir.compile_program(
            program,
            target=target,
            output=mlir.OutputFormat.QPY,
            qco_pipeline=pipeline,
            options=mlir.CompilationOptions(
                seed=options.seed,
                mapping=mlir.MappingOptions(**mapping),
                enable_timing=options.time_passes,
                enable_statistics=options.mlir_pass_statistics,
            ),
        )
    except QiskitError as error:
        raise ValueError(str(error)) from error
    if options.o == "-":
        sys.stdout.buffer.write(payload)
    else:
        Path(options.o).write_bytes(payload)
    return 0
