# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for QDMI Qiskit types (ProgramIR, ProgramInstruction)."""

from __future__ import annotations

import pytest

from mqt.core.qdmi.qiskit import IRValidationError, ProgramInstruction, ProgramIR


def test_program_instruction_creation() -> None:
    """ProgramInstruction can be created with valid data."""
    inst = ProgramInstruction(name="cz", qubits=[0, 1])
    assert inst.name == "cz"
    assert inst.qubits == [0, 1]
    assert inst.params is None
    assert inst.metadata == {}


def test_program_instruction_with_params() -> None:
    """ProgramInstruction can include parameters."""
    inst = ProgramInstruction(name="ry", qubits=[0], params=[1.5708])
    assert inst.name == "ry"
    assert inst.params == [1.5708]


def test_program_instruction_with_metadata() -> None:
    """ProgramInstruction can include metadata."""
    meta = {"custom": "data"}
    inst = ProgramInstruction(name="cz", qubits=[0, 1], metadata=meta)
    assert inst.metadata == meta


def test_program_instruction_empty_name() -> None:
    """ProgramInstruction with empty name should raise IRValidationError."""
    with pytest.raises(IRValidationError, match="name must be non-empty"):
        ProgramInstruction(name="", qubits=[0])


def test_program_instruction_negative_qubit() -> None:
    """ProgramInstruction with negative qubit index should raise IRValidationError."""
    with pytest.raises(IRValidationError, match="Qubit indices must be non-negative"):
        ProgramInstruction(name="cz", qubits=[0, -1])


def test_program_instruction_all_negative_qubits() -> None:
    """ProgramInstruction with all negative qubits should raise IRValidationError."""
    with pytest.raises(IRValidationError, match="Qubit indices must be non-negative"):
        ProgramInstruction(name="x", qubits=[-5])


def test_program_ir_creation() -> None:
    """ProgramIR can be created with instructions."""
    inst1 = ProgramInstruction(name="cz", qubits=[0, 1])
    inst2 = ProgramInstruction(name="measure", qubits=[0, 1])

    program = ProgramIR(name="test_program", instructions=[inst1, inst2])
    assert program.name == "test_program"
    assert len(program.instructions) == 2
    assert program.metadata == {}


def test_program_ir_with_metadata() -> None:
    """ProgramIR can include metadata."""
    inst = ProgramInstruction(name="cz", qubits=[0, 1])
    meta = {"author": "test"}

    program = ProgramIR(name="test", instructions=[inst], metadata=meta)
    assert program.metadata == meta


def test_program_ir_validate_success() -> None:
    """ProgramIR validation should pass with valid operations."""
    inst1 = ProgramInstruction(name="cz", qubits=[0, 1])
    inst2 = ProgramInstruction(name="measure", qubits=[0])

    program = ProgramIR(name="test", instructions=[inst1, inst2])

    # Should not raise
    program.validate(
        num_qubits=5,
        allowed_operations={"cz", "measure"},
    )


def test_program_ir_validate_with_pseudo_ops() -> None:
    """ProgramIR validation should allow pseudo operations."""
    inst1 = ProgramInstruction(name="barrier", qubits=[0, 1])
    inst2 = ProgramInstruction(name="cz", qubits=[0, 1])

    program = ProgramIR(name="test", instructions=[inst1, inst2])

    # Should not raise
    program.validate(
        num_qubits=5,
        allowed_operations={"cz"},
        pseudo_ops={"barrier"},
    )


def test_program_ir_validate_unsupported_operation() -> None:
    """ProgramIR validation should fail with unsupported operation."""
    inst = ProgramInstruction(name="unsupported", qubits=[0])
    program = ProgramIR(name="test", instructions=[inst])

    with pytest.raises(IRValidationError, match="Instruction 0 uses unsupported operation 'unsupported'"):
        program.validate(
            num_qubits=5,
            allowed_operations={"cz", "measure"},
        )


def test_program_ir_validate_qubit_out_of_range() -> None:
    """ProgramIR validation should fail when qubit index exceeds num_qubits."""
    inst = ProgramInstruction(name="cz", qubits=[0, 10])
    program = ProgramIR(name="test", instructions=[inst])

    with pytest.raises(IRValidationError, match="qubit index 10 exceeds available qubits 5"):
        program.validate(
            num_qubits=5,
            allowed_operations={"cz"},
        )


def test_program_ir_validate_first_instruction_fails() -> None:
    """ProgramIR validation should report correct instruction index."""
    inst1 = ProgramInstruction(name="unsupported", qubits=[0])
    inst2 = ProgramInstruction(name="cz", qubits=[0, 1])

    program = ProgramIR(name="test", instructions=[inst1, inst2])

    with pytest.raises(IRValidationError, match="Instruction 0"):
        program.validate(
            num_qubits=5,
            allowed_operations={"cz"},
        )


def test_program_ir_validate_second_instruction_fails() -> None:
    """ProgramIR validation should report correct instruction index for later instructions."""
    inst1 = ProgramInstruction(name="cz", qubits=[0, 1])
    inst2 = ProgramInstruction(name="unsupported", qubits=[0])

    program = ProgramIR(name="test", instructions=[inst1, inst2])

    with pytest.raises(IRValidationError, match="Instruction 1"):
        program.validate(
            num_qubits=5,
            allowed_operations={"cz"},
        )


def test_program_ir_validate_empty_instructions() -> None:
    """ProgramIR with empty instruction list should validate successfully."""
    program = ProgramIR(name="empty", instructions=[])

    # Should not raise
    program.validate(
        num_qubits=5,
        allowed_operations={"cz"},
    )


def test_program_ir_validate_multiple_qubit_violations() -> None:
    """ProgramIR validation should catch the first out-of-range qubit."""
    inst = ProgramInstruction(name="cz", qubits=[3, 8])
    program = ProgramIR(name="test", instructions=[inst])

    with pytest.raises(IRValidationError, match=r"qubit index .* exceeds available qubits"):
        program.validate(
            num_qubits=5,
            allowed_operations={"cz"},
        )


def test_ir_validation_error_is_value_error() -> None:
    """IRValidationError should be a subclass of ValueError."""
    assert issubclass(IRValidationError, ValueError)


def test_program_instruction_params_none_default() -> None:
    """ProgramInstruction params should default to None."""
    inst = ProgramInstruction(name="cz", qubits=[0, 1])
    assert inst.params is None


def test_program_instruction_metadata_default_factory() -> None:
    """ProgramInstruction metadata should be independent across instances."""
    inst1 = ProgramInstruction(name="cz", qubits=[0, 1])
    inst2 = ProgramInstruction(name="ry", qubits=[0])

    inst1.metadata["key"] = "value1"

    # inst2 should have its own empty dict
    assert "key" not in inst2.metadata


def test_program_ir_metadata_default_factory() -> None:
    """ProgramIR metadata should be independent across instances."""
    inst = ProgramInstruction(name="cz", qubits=[0, 1])

    prog1 = ProgramIR(name="prog1", instructions=[inst])
    prog2 = ProgramIR(name="prog2", instructions=[inst])

    prog1.metadata["key"] = "value1"

    # prog2 should have its own empty dict
    assert "key" not in prog2.metadata
