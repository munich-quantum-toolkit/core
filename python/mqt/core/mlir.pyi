# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""MQT Core MLIR compiler bindings."""

def compile_program(
    program: object,
    *,
    convert_to_qir_base: bool = False,
    convert_to_qir_adaptive: bool = False,
    disable_merge_single_qubit_rotation_gates: bool = False,
    enable_hadamard_lifting: bool = False,
    enable_timing: bool = False,
    enable_statistics: bool = False,
) -> str:
    """Compile an input quantum program with the MQT MLIR compiler pipeline.

    Args:
        program: Input program in one of the supported forms:
            - :class:`mqt.core.ir.QuantumComputation`
            - OpenQASM source text
            - Path to `.qasm`, `.mlir`, or `.jeff` files
            - Qiskit :class:`~qiskit.circuit.QuantumCircuit`
            - MLIR source text
        convert_to_qir_base: Whether to lower the result to a QIR program compliant with the Base Profile.
        convert_to_qir_adaptive: Whether to lower the result to QIR program compliant with the Adaptive Profile.
        disable_merge_single_qubit_rotation_gates: Disable quaternion-based
            rotation merging.
        enable_hadamard_lifting: Enable Hadamard lifting optimization.
        enable_timing: Enable MLIR pass timing.
        enable_statistics: Enable MLIR pass statistics.

    Returns:
        The final MLIR module as text.
    """
