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
    unregister_operation_translator("custom_gate")


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


def test_instruction_context_creation() -> None:
    """InstructionContext can be created with valid data."""
    ctx = InstructionContext(name="cz", qubits=[0, 1])
    assert ctx.name == "cz"
    assert ctx.qubits == [0, 1]
    assert ctx.params is None
    assert ctx.clbits is None
    assert ctx.metadata is None


def test_instruction_context_with_params() -> None:
    """InstructionContext can include parameters."""
    ctx = InstructionContext(name="ry", qubits=[0], params=[1.5708])
    assert ctx.params == [1.5708]


def test_instruction_context_with_clbits() -> None:
    """InstructionContext can include classical bits."""
    ctx = InstructionContext(name="measure", qubits=[0], clbits=[0])
    assert ctx.clbits == [0]


def test_instruction_context_with_metadata() -> None:
    """InstructionContext can include metadata."""
    meta = {"custom": "data"}
    ctx = InstructionContext(name="cz", qubits=[0, 1], metadata=meta)
    assert ctx.metadata == meta


def test_instruction_context_empty_name() -> None:
    """InstructionContext with empty name should raise ValueError."""
    with pytest.raises(ValueError, match="name must be non-empty"):
        InstructionContext(name="", qubits=[0])


def test_instruction_context_negative_qubit() -> None:
    """InstructionContext with negative qubit should raise ValueError."""
    with pytest.raises(ValueError, match="Qubit indices must be non-negative"):
        InstructionContext(name="cz", qubits=[0, -1])


def test_instruction_context_negative_clbit() -> None:
    """InstructionContext with negative clbit should raise ValueError."""
    with pytest.raises(ValueError, match="Classical bit indices must be non-negative"):
        InstructionContext(name="measure", qubits=[0], clbits=[-1])


def test_get_operation_translator_not_found() -> None:
    """Getting a non-existent translator should raise KeyError."""
    with pytest.raises(KeyError):
        get_operation_translator("nonexistent_operation")


def test_unregister_nonexistent_translator() -> None:
    """Unregistering a non-existent translator should not raise."""
    # Should not raise an error
    unregister_operation_translator("nonexistent_operation")


def test_clear_operation_translators_without_defaults() -> None:
    """clear_operation_translators with keep_defaults=False should remove all."""

    def _tmp(ctx: InstructionContext) -> list[ProgramInstruction]:  # noqa: ARG001
        return [ProgramInstruction(name="tmp", qubits=[0])]

    register_operation_translator("tmp", _tmp)
    clear_operation_translators(keep_defaults=False)

    # Both default and custom should be gone
    assert "tmp" not in list_operation_translators()

    # Restore defaults to avoid leaking state into other tests
    clear_operation_translators(keep_defaults=True)


def test_clear_operation_translators_with_defaults() -> None:
    """clear_operation_translators with keep_defaults=True should keep defaults."""

    def _tmp(ctx: InstructionContext) -> list[ProgramInstruction]:  # noqa: ARG001
        return [ProgramInstruction(name="tmp", qubits=[0])]

    register_operation_translator("tmp", _tmp)
    clear_operation_translators(keep_defaults=True)

    # Custom should be gone, defaults should remain
    assert "tmp" not in list_operation_translators()
    assert "measure" in list_operation_translators()  # default


def test_build_program_ir_with_params() -> None:
    """build_program_ir should handle instructions with parameters."""

    def _ry(ctx: InstructionContext) -> list[ProgramInstruction]:
        return [ProgramInstruction(name="ry", qubits=ctx.qubits, params=ctx.params)]

    register_operation_translator("ry", _ry, overwrite=True)

    ctx = InstructionContext(name="ry", qubits=[0], params=[1.5708])
    ir = build_program_ir(
        name="param_prog",
        instruction_contexts=[ctx],
        allowed_operations={"ry"},
        num_qubits=2,
    )

    assert len(ir.instructions) == 1
    assert ir.instructions[0].params == [1.5708]
    unregister_operation_translator("ry")


def test_build_program_ir_empty_contexts() -> None:
    """build_program_ir should handle empty instruction list."""
    ir = build_program_ir(
        name="empty_prog",
        instruction_contexts=[],
        allowed_operations={"cz"},
        num_qubits=2,
    )

    assert len(ir.instructions) == 0


def test_build_program_ir_multiple_pseudo_ops() -> None:
    """build_program_ir should handle multiple pseudo operations."""

    def _cz(ctx: InstructionContext) -> list[ProgramInstruction]:
        return [ProgramInstruction(name="cz", qubits=ctx.qubits)]

    register_operation_translator("cz", _cz, overwrite=True)

    contexts = [
        InstructionContext(name="barrier", qubits=[0]),
        InstructionContext(name="cz", qubits=[0, 1]),
        InstructionContext(name="delay", qubits=[0]),
    ]

    ir = build_program_ir(
        name="multi_pseudo",
        instruction_contexts=contexts,
        allowed_operations={"cz"},
        num_qubits=3,
        pseudo_ops=["barrier", "delay"],
    )

    assert len(ir.instructions) == 3
    unregister_operation_translator("cz")


def test_measure_translator_mismatched_clbits() -> None:
    """Measure translator should fail if clbits count doesn't match qubits."""
    ctx = InstructionContext(name="measure", qubits=[0, 1], clbits=[0])  # mismatch

    with pytest.raises(IRValidationError, match="one-to-one mapping"):
        build_program_ir(
            name="bad_measure",
            instruction_contexts=[ctx],
            allowed_operations={"measure"},
            num_qubits=3,
        )


def test_program_ir_name_preserved() -> None:
    """build_program_ir should preserve the program name."""
    ir = build_program_ir(
        name="my_program",
        instruction_contexts=[],
        allowed_operations=set(),
        num_qubits=2,
    )

    assert ir.name == "my_program"


def test_default_translators_are_registered() -> None:
    """Verify that default translators are registered at module load."""
    clear_operation_translators(keep_defaults=True)
    translators = list_operation_translators()

    # Check some common gates
    assert "measure" in translators
    assert "h" in translators
    assert "x" in translators
    assert "cx" in translators
    assert "ry" in translators
