# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for MQT plugin execution with PennyLane and Catalyst.

These tests check that the MQT plugin conversion passes execute successfully
for various gate categories, mirroring the MLIR conversion tests. They verify
that the full lossless roundtrip (CatalystQuantum → MQTOpt → CatalystQuantum)
works correctly.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import catalyst
import pennylane as qml
from catalyst.passes import apply_pass

from mqt.core.plugins.catalyst import get_device


def test_clifford_gates_roundtrip() -> None:
    """Test roundtrip conversion of Clifford+T gates.

    Mirrors: quantum_clifford.mlir
    Gates: H, SX, SX†, S, S†, T, T†, and their controlled variants
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")
    @apply_pass("mqt.catalystquantum-to-mqtopt")
    @qml.qnode(get_device("lightning.qubit", wires=2))
    def circuit() -> None:
        # Non-controlled Clifford+T gates
        qml.Hadamard(wires=0)
        qml.SX(wires=0)
        qml.adjoint(qml.SX(wires=0))
        qml.S(wires=0)
        qml.adjoint(qml.S(wires=0))
        qml.T(wires=0)
        qml.adjoint(qml.T(wires=0))

        # Controlled Clifford+T gates
        qml.CH(wires=[1, 0])
        qml.ctrl(qml.SX(wires=0), control=1)
        # Why is `qml.ctrl(qml.adjoint(qml.SX(wires=0)), control=1)` not supported by Catalyst?
        qml.ctrl(qml.S(wires=0), control=1)
        qml.ctrl(qml.adjoint(qml.S(wires=0)), control=1)
        qml.ctrl(qml.T(wires=0), control=1)
        qml.ctrl(qml.adjoint(qml.T(wires=0)), control=1)

        catalyst.measure(0)
        catalyst.measure(1)

    @qml.qjit(target="mlir", autograph=True)
    def module() -> None:
        return circuit()

    # Verify the roundtrip completes successfully
    mlir_opt = module.mlir_opt
    assert mlir_opt


def test_pauli_gates_roundtrip() -> None:
    """Test roundtrip conversion of Pauli gates.

    Mirrors: quantum_pauli.mlir
    Gates: X, Y, Z, I, and their controlled variants (CNOT, CY, CZ, Toffoli)
    Structure:
    1. Uncontrolled Pauli gates (X, Y, Z, I)
    2. Controlled Pauli gates (using qml.ctrl on Pauli gates)
    3. Two-qubit controlled gates (CNOT, CY, CZ)
    4. Toffoli (CCX)
    5. Controlled two-qubit gates (controlled CNOT, CY, CZ, Toffoli)
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")
    @apply_pass("mqt.catalystquantum-to-mqtopt")
    @qml.qnode(get_device("lightning.qubit", wires=4))
    def circuit() -> None:
        # Uncontrolled Pauli gates
        qml.PauliX(wires=0)
        qml.PauliY(wires=0)
        qml.PauliZ(wires=0)
        qml.Identity(wires=0)

        # Controlled Pauli gates (single control) - use qml.ctrl on Pauli gates
        qml.ctrl(qml.PauliX(wires=0), control=1)
        qml.ctrl(qml.PauliY(wires=0), control=1)
        qml.ctrl(qml.PauliZ(wires=0), control=1)
        # Why is `qml.ctrl(qml.Identity(wires=0), control=1)` not supported by Catalyst?

        # Two-qubit controlled gates (explicit CNOT, CY, CZ gate names)
        qml.CNOT(wires=[0, 1])
        qml.CY(wires=[0, 1])
        qml.CZ(wires=[0, 1])

        # Toffoli (also CCX)
        qml.Toffoli(wires=[0, 1, 2])

        # Controlled two-qubit gates (adding extra controls)
        qml.ctrl(qml.CNOT(wires=[0, 1]), control=2)
        qml.ctrl(qml.CY(wires=[0, 1]), control=2)
        qml.ctrl(qml.CZ(wires=[0, 1]), control=2)
        qml.ctrl(qml.Toffoli(wires=[0, 1, 2]), control=3)

        catalyst.measure(0)
        catalyst.measure(1)
        catalyst.measure(2)
        catalyst.measure(3)

    @qml.qjit(target="mlir", autograph=True)
    def module() -> None:
        return circuit()

    # Verify the roundtrip completes successfully
    mlir_opt = module.mlir_opt
    assert mlir_opt


def test_parameterized_gates_roundtrip() -> None:
    """Test roundtrip conversion of parameterized rotation gates.

    Mirrors: quantum_param.mlir
    Gates: RX, RY, RZ, PhaseShift, and their controlled variants (CRX, CRY)
    Note: MLIR test does NOT include CRZ, only CRX and CRY
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")
    @apply_pass("mqt.catalystquantum-to-mqtopt")
    @qml.qnode(get_device("lightning.qubit", wires=2))
    def circuit() -> None:
        angle = 0.3

        # Non-controlled parameterized gates
        qml.RX(angle, wires=0)
        qml.RY(angle, wires=0)
        qml.RZ(angle, wires=0)
        qml.PhaseShift(angle, wires=0)

        # Controlled parameterized gates
        qml.CRX(angle, wires=[1, 0])
        qml.CRY(angle, wires=[1, 0])

        catalyst.measure(0)
        catalyst.measure(1)

    @qml.qjit(target="mlir", autograph=True)
    def module() -> None:
        return circuit()

    # Verify the roundtrip completes successfully
    mlir_opt = module.mlir_opt
    assert mlir_opt


def test_entangling_gates_roundtrip() -> None:
    """Test roundtrip conversion of entangling/permutation gates.

    Mirrors: quantum_entangling.mlir
    Gates: SWAP, ISWAP, ISWAP†, ECR, and their controlled variants
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")
    @apply_pass("mqt.catalystquantum-to-mqtopt")
    @qml.qnode(get_device("lightning.qubit", wires=3))
    def circuit() -> None:
        # Uncontrolled permutation gates
        qml.SWAP(wires=[0, 1])
        qml.ISWAP(wires=[0, 1])
        qml.adjoint(qml.ISWAP(wires=[0, 1]))
        qml.ECR(wires=[0, 1])

        # Controlled permutation gates
        qml.CSWAP(wires=[2, 0, 1])
        qml.ctrl(qml.ISWAP(wires=[0, 1]), control=2)
        qml.ctrl(qml.adjoint(qml.ISWAP(wires=[0, 1])), control=2)
        qml.ctrl(qml.ECR(wires=[0, 1]), control=2)

        catalyst.measure(0)
        catalyst.measure(1)

    @qml.qjit(target="mlir", autograph=True)
    def module() -> None:
        return circuit()

    # Verify the roundtrip completes successfully
    mlir_opt = module.mlir_opt
    assert mlir_opt


def test_ising_gates_roundtrip() -> None:
    """Test roundtrip conversion of Ising-type gates.

    Mirrors: quantum_ising.mlir
    Gates: IsingXY, IsingXX, IsingYY, IsingZZ, and their controlled variants
    Note: IsingXY takes 2 parameters in MLIR (phi and beta)
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")
    @apply_pass("mqt.catalystquantum-to-mqtopt")
    @qml.qnode(get_device("lightning.qubit", wires=3))
    def circuit() -> None:
        angle = 0.3

        # Uncontrolled Ising gates
        qml.IsingXY(angle, wires=[0, 1])
        qml.IsingXX(angle, wires=[0, 1])
        qml.IsingYY(angle, wires=[0, 1])
        qml.IsingZZ(angle, wires=[0, 1])

        # Controlled Ising gates
        qml.ctrl(qml.IsingXY(angle, wires=[0, 1]), control=2)
        qml.ctrl(qml.IsingXX(angle, wires=[0, 1]), control=2)
        qml.ctrl(qml.IsingYY(angle, wires=[0, 1]), control=2)
        qml.ctrl(qml.IsingZZ(angle, wires=[0, 1]), control=2)

        catalyst.measure(0)
        catalyst.measure(1)

    custom_pipeline = [
        ("to-mqtopt", ["builtin.module(catalystquantum-to-mqtopt)"]),
        ("to-catalystquantum", ["builtin.module(mqtopt-to-catalystquantum)"]),
    ]

    @qml.qjit(target="mlir", pipelines=custom_pipeline, autograph=True, keep_intermediate=2)
    def module() -> None:
        return circuit()

    # Verify the roundtrip completes successfully
    mlir_opt = module.mlir_opt
    assert mlir_opt

    # Find the python directory where MLIR files are generated
    # This works regardless of where pytest is run from (locally or CI)
    test_file_dir = Path(__file__).parent
    python_dir = test_file_dir.parent.parent / "python"
    
    # Read the intermediate MLIR files
    mlir_to_mqtopt = python_dir / "3_to-mqtopt.mlir"
    mlir_to_catalyst = python_dir / "4_MQTOptToCatalystQuantum.mlir"
    
    if not mlir_to_mqtopt.exists() or not mlir_to_catalyst.exists():
        # Fallback: list what files actually exist for debugging
        available_files = list(python_dir.glob("*.mlir"))
        raise FileNotFoundError(
            f"Expected MLIR files not found in {python_dir}.\n"
            f"Available files: {[f.name for f in available_files]}"
        )
    
    with open(mlir_to_mqtopt, "r") as f:
        mlir_after_mqtopt = f.read()
    with open(mlir_to_catalyst, "r") as f:
        mlir_after_roundtrip = f.read()

    # TODO: As with the lit CHECK tests, we can do some basic string checks here:
    # assert ... in mlir_after_mqtopt 
    # assert ... in mlir_after_roundtrip

    # Remove all intermediate files created during the test
    for mlir_file in python_dir.glob("*.mlir"):
        mlir_file.unlink()

    

def test_mqtopt_roundtrip() -> None:
    """Execute the full roundtrip including MQT Core IR.

    Executes the conversion passes to and from MQTOpt dialect AND
    the roundtrip through MQT Core IR.
    """

    @apply_pass("mqt.mqtopt-to-catalystquantum")
    @apply_pass("mqt.mqt-core-round-trip")
    @apply_pass("mqt.catalystquantum-to-mqtopt")
    @qml.qnode(get_device("lightning.qubit", wires=2))
    def circuit() -> None:
        qml.Hadamard(wires=[0])
        qml.CNOT(wires=[0, 1])
        catalyst.measure(0)
        catalyst.measure(1)

    @qml.qjit(target="mlir", autograph=True)
    def module() -> None:
        return circuit()

    # This will execute the pass and return the final MLIR
    mlir_opt = module.mlir_opt
    assert mlir_opt
