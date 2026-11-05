# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Sequence input, ownership, and return types of the compiler convenience API."""

from __future__ import annotations

from collections import UserList
from typing import TYPE_CHECKING, assert_type, cast

import pytest

from mqt.core.mlir import (
    CompiledProgram,
    CompilerTarget,
    JeffProgram,
    OpenQASMProgram,
    OutputFormat,
    QCOProgram,
    QCProgram,
    compile_program,
    sample,
    submit_program,
)
from mqt.core.qdmi import open_device

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

ZERO = "OPENQASM 3.0; qubit q; bit c = measure q;"
ONE = 'OPENQASM 3.0; include "stdgates.inc"; qubit q; x q; bit c = measure q;'


@pytest.mark.parametrize("sources", [[], (ZERO,), UserList([ZERO, ONE])])
def test_compile_sequence(sources: Sequence[str]) -> None:
    """Compile generic sequences into ordered lists, including empty and singleton inputs."""
    result = compile_program(sources, output=OutputFormat.QCO_OPTIMIZED)
    assert_type(result, list[QCOProgram])
    assert isinstance(result, list)
    assert [sample(program, shots=1) for program in result] == [{"0": 1}, {"1": 1}][: len(sources)]


@pytest.mark.parametrize("as_paths", [False, True])
def test_sequence_infers_each_source_format(tmp_path: Path, *, as_paths: bool) -> None:
    """One Python input kind can contain different source formats."""
    target = CompilerTarget(
        1,
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations.unrestricted(),
    )
    sources = [ZERO, QCProgram.from_openqasm_str(ONE).ir]
    if as_paths:
        paths = [tmp_path / "zero.qasm", tmp_path / "one.mlir"]
        for path, source in zip(paths, sources, strict=True):
            path.write_text(source)
        outputs = compile_program(paths, target=target)
    else:
        outputs = compile_program(sources, target=target)
    assert_type(outputs, list[QCProgram | JeffProgram | OpenQASMProgram])
    assert isinstance(outputs[0], OpenQASMProgram)
    assert isinstance(outputs[1], QCProgram)
    assert [sample(program, shots=1) for program in outputs] == [{"0": 1}, {"1": 1}]


def test_sequence_for_device() -> None:
    """Compile a sequence to device payloads and execute each result."""
    device = open_device("mqt.ddsim.default")
    compiled = compile_program((ZERO, ONE), target=device)
    assert_type(compiled, list[CompiledProgram])
    results = []
    for program in compiled:
        job = submit_program(program, target=device, num_shots=1)
        job.wait()
        results.append(job.get_counts())
    assert results == [{"0": 1}, {"1": 1}]


@pytest.mark.parametrize("copy", [False, True])
def test_sequence_consumption(*, copy: bool) -> None:
    """Keep scalar ownership semantics for every typed input."""
    inputs = [QCProgram.from_openqasm_str(source) for source in (ZERO, ONE)]
    result = compile_program(inputs, copy=copy)
    assert_type(result, list[QCProgram])
    assert [program.is_valid for program in inputs] == [copy, copy]
    assert [sample(program, shots=1) for program in result] == [{"0": 1}, {"1": 1}]


@pytest.mark.parametrize("copy", [False, True])
def test_sequence_failure_consumes_only_processed_inputs(*, copy: bool) -> None:
    """Stop at a failing program without consuming subsequent elements."""
    inputs = [
        QCProgram.from_openqasm_str(source) for source in (ZERO, "OPENQASM 3.0; qubit[3] q; bit[3] c = measure q;", ONE)
    ]
    target = CompilerTarget(
        2,
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations.unrestricted(),
    )
    with pytest.raises(RuntimeError):
        compile_program(inputs, target=target, copy=copy)
    assert [program.is_valid for program in inputs] == [copy, copy, True]


def test_sequence_preflights_input_kinds() -> None:
    """Reject mixed representations before consuming any program."""
    qc = QCProgram.from_openqasm_str(ZERO)
    qco = compile_program(ONE, output=OutputFormat.QCO)
    inputs = cast("Sequence[QCProgram]", [qc, qco])
    with pytest.raises(TypeError, match=r"input kind.*index 1"):
        compile_program(inputs)
    assert qc.is_valid
    assert qco.is_valid
