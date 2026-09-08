# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compile to the fixed square-root iSWAP capability without external SDKs."""

from mqt.core.mlir import CompilerTarget, OutputFormat, QCOProgram, compile_program


def test_compile_cx_to_sqrt_iswap() -> None:
    """Expose the target enum and compile CX to two native entanglers."""
    target = CompilerTarget(
        2,
        connectivity=CompilerTarget.Connectivity.all_to_all(),
        native_operations=CompilerTarget.NativeOperations([
            CompilerTarget.Operation("u", 1, 3),
            CompilerTarget.Operation("gphase", 0, 1),
            CompilerTarget.Operation("sqrt_iswap", 2, 0),
        ]),
    )
    assert target.synthesis_basis is not None
    assert target.synthesis_basis.entangler == CompilerTarget.GateKind.SQRT_ISWAP
    program = compile_program(
        'OPENQASM 3.0; include "stdgates.inc"; qubit[2] q; cx q[0], q[1];',
        output=OutputFormat.QCO,
    )
    assert isinstance(program, QCOProgram)
    program.compile_for_target(target)
    assert program.ir.count("qco.xx_plus_yy") == 2
    assert "qco.ctrl" not in program.ir
