# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for QDMI Qiskit translator registry & neutral IR."""

from __future__ import annotations

import pytest

from mqt.core.qdmi.qiskit import (
    InstructionContext,
    IRValidationError,
    ProgramInstruction,
    ProgramIR,
    build_program_ir,
    clear_operation_translators,
    get_operation_translator,
    list_operation_translators,
    register_operation_translator,
    unregister_operation_translator,
)


def setup_module() -> None:  # noqa: D103
    # Ensure a clean registry state (keeping default measure translator)
    clear_operation_translators(keep_defaults=True)


def test_register_and_list_translators() -> None:
    """Register a new translator and ensure it appears in the listing."""

    def _custom_gate(ctx: InstructionContext) -> list[ProgramInstruction]:  # simple passthrough
        return [ProgramInstruction(name="custom_gate", qubits=ctx.qubits, params=ctx.params)]

    register_operation_translator("custom_gate", _custom_gate)
    names = list_operation_translators()
    assert "measure" in names  # default
    assert "custom_gate" in names
    # retrieve translator
    retrieved = get_operation_translator("custom_gate")
    assert retrieved is _custom_gate


def test_unregister_translator() -> None:
    """Unregister an existing translator and verify it's removed."""

    def _custom_gate(ctx: InstructionContext) -> list[ProgramInstruction]:  # noqa: ARG001
        return [ProgramInstruction(name="custom_gate", qubits=[0])]

    register_operation_translator("custom_gate", _custom_gate, overwrite=True)
    unregister_operation_translator("custom_gate")
    assert "custom_gate" not in list_operation_translators()


def test_overwrite_flag() -> None:
    """Re-registering without overwrite should fail; with overwrite should pass."""

    def _tmp(ctx: InstructionContext) -> list[ProgramInstruction]:  # noqa: ARG001
        return [ProgramInstruction(name="tmp", qubits=[0])]

    register_operation_translator("tmp", _tmp)

    with pytest.raises(ValueError, match="already registered"):
        register_operation_translator("tmp", _tmp, overwrite=False)

    register_operation_translator("tmp", _tmp, overwrite=True)  # succeeds
    unregister_operation_translator("tmp")


def test_build_program_ir_basic_measure_and_barrier() -> None:
    """Build IR with measure and barrier pseudo op; verify validation passes."""

    # Provide explicit translators for a sample operation
    def _cz(ctx: InstructionContext) -> list[ProgramInstruction]:
        return [ProgramInstruction(name="cz", qubits=ctx.qubits)]

    register_operation_translator("cz", _cz, overwrite=True)

    contexts = [
        InstructionContext(name="cz", qubits=[0, 1]),
        InstructionContext(name="barrier", qubits=[0, 1]),
        InstructionContext(name="measure", qubits=[0, 1], clbits=[0, 1]),
    ]
    allowed_ops = {"cz", "measure"}
    ir = build_program_ir(
        name="test_prog",
        instruction_contexts=contexts,
        allowed_operations=allowed_ops,
        num_qubits=5,
        pseudo_ops=["barrier"],
    )
    assert isinstance(ir, ProgramIR)
    assert [inst.name for inst in ir.instructions] == ["cz", "barrier", "measure"]
    unregister_operation_translator("cz")


def test_measure_translator_requires_clbits() -> None:
    """The default measure translator should enforce one-to-one clbit mapping."""
    ctx = InstructionContext(name="measure", qubits=[0])  # missing clbits
    with pytest.raises(IRValidationError):
        build_program_ir(
            name="bad_measure",
            instruction_contexts=[ctx],
            allowed_operations={"measure"},
            num_qubits=2,
        )


def test_build_program_ir_missing_translator() -> None:
    """Building IR with unsupported op should raise IRValidationError."""
    ctx = InstructionContext(name="unknown_op", qubits=[0])
    with pytest.raises(IRValidationError):
        build_program_ir(
            name="bad_prog",
            instruction_contexts=[ctx],
            allowed_operations={"measure"},
            num_qubits=2,
        )


def test_build_program_ir_qubit_range_violation() -> None:
    """A qubit index exceeding num_qubits triggers validation error."""

    def _id(ctx: InstructionContext) -> list[ProgramInstruction]:
        return [ProgramInstruction(name="id", qubits=ctx.qubits)]

    register_operation_translator("id", _id, overwrite=True)

    ctx = InstructionContext(name="id", qubits=[0, 10])  # 10 out of range for num_qubits=5
    with pytest.raises(IRValidationError):
        build_program_ir(
            name="range_fail",
            instruction_contexts=[ctx],
            allowed_operations={"id"},
            num_qubits=5,
        )

    unregister_operation_translator("id")
