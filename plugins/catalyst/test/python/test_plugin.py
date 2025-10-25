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
works correctly. The tests use FileCheck (from LLVM) to verify the generated MLIR output.

Environment Variables:
    FILECHECK_PATH: Optional path to FileCheck binary if not in PATH
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pennylane as qml
from catalyst.passes import apply_pass

from mqt.core.plugins.catalyst import get_device


def _run_filecheck(mlir_content: str, check_patterns: str, test_name: str = "test") -> None:
    """Run FileCheck on MLIR content using CHECK patterns from a string.

    Args:
        mlir_content: The MLIR output to verify
        check_patterns: String containing FileCheck directives (lines starting with // CHECK)
        test_name: Name of the test (for error messages)

    Raises:
        RuntimeError: If FileCheck is not found
        AssertionError: If FileCheck validation fails
    """
    # Find FileCheck (usually in LLVM bin directory)
    filecheck = None
    possible_paths = [
        "FileCheck",  # If in PATH
        os.environ.get("FILECHECK_PATH"),  # Custom env variable
        "/opt/homebrew/opt/llvm/bin/FileCheck",  # Common macOS location
    ]

    for path in possible_paths:
        if path:
            try:
                result = subprocess.run([path, "--version"], check=False, capture_output=True, timeout=5)  # noqa: S603
                if result.returncode == 0:
                    filecheck = path
                    break
            except (subprocess.SubprocessError, FileNotFoundError):
                continue

    if not filecheck:
        msg = (
            "FileCheck not found. Please ensure LLVM's FileCheck is in your PATH, "
            "or set FILECHECK_PATH environment variable."
        )
        raise RuntimeError(msg)

    # Write CHECK patterns to a temporary file
    import tempfile

    with tempfile.NamedTemporaryFile(encoding="utf-8", mode="w", suffix=".mlir", delete=False) as check_file:
        check_file.write(check_patterns)
        check_file_path = check_file.name

    try:
        # Run FileCheck: pipe MLIR content as stdin, use check_file for CHECK directives
        result = subprocess.run(  # noqa: S603
            [filecheck, check_file_path, "--allow-unused-prefixes"],
            check=False,
            input=mlir_content.encode(),
            capture_output=True,
            timeout=30,
        )

        if result.returncode != 0:
            error_msg = result.stderr.decode() if result.stderr else "Unknown error"
            msg = (
                f"FileCheck failed for {test_name}:\n{error_msg}\n\n"
                f"MLIR Output (first 2000 chars):\n{mlir_content[:2000]}..."
            )
            raise AssertionError(msg)
    finally:
        # Clean up temporary file
        Path(check_file_path).unlink()


def test_clifford_gates_roundtrip() -> None:
    """Test roundtrip conversion of Clifford+T gates.

    Mirrors: quantum_clifford.mlir
    Gates: H, SX, SX†, S, S†, T, T†, and their controlled variants

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found
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

    # Find where MLIR files are generated (relative to cwd where pytest is run)
    # Catalyst generates MLIR files in the current working directory
    mlir_dir = Path.cwd()

    # Read the intermediate MLIR files
    mlir_to_mqtopt = mlir_dir / "3_to-mqtopt.mlir"
    mlir_to_catalyst = mlir_dir / "4_MQTOptToCatalystQuantum.mlir"

    if not mlir_to_mqtopt.exists() or not mlir_to_catalyst.exists():
        available_files = list(mlir_dir.glob("*.mlir"))
        msg = f"Expected MLIR files not found in {mlir_dir}.\nAvailable files: {[f.name for f in available_files]}"
        raise FileNotFoundError(msg)

    with Path(mlir_to_mqtopt).open("r", encoding="utf-8") as f:
        mlir_after_mqtopt = f.read()
    with Path(mlir_to_catalyst).open("r", encoding="utf-8") as f:
        mlir_after_roundtrip = f.read()

    # Verify CatalystQuantum → MQTOpt conversion
    check_to_mqtopt = """
    // CHECK: func.func {{.*}}@circuit
    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x!mqtopt.Qubit>
    // CHECK: %[[Q0:.*]] = memref.load %[[ALLOC]][{{.*}}] : memref<2x!mqtopt.Qubit>

    // Uncontrolled Clifford+T gates
    // Hadamard
    // CHECK: %[[H:.*]] = mqtopt.h(static [] mask []) %[[Q0]] : !mqtopt.Qubit

    // SX gate decomposes to RZ(π/2) -> RY(π/2) -> RZ(-π/2) + gphase
    // CHECK: %[[RZ1:.*]] = mqtopt.rz({{.*}} static [] mask [false]) %[[H]] : !mqtopt.Qubit
    // CHECK: %[[RY1:.*]] = mqtopt.ry({{.*}} static [] mask [false]) %[[RZ1]] : !mqtopt.Qubit
    // CHECK: %[[RZ2:.*]] = mqtopt.rz({{.*}} static [] mask [false]) %[[RY1]] : !mqtopt.Qubit
    // CHECK: mqtopt.gphase({{.*}} static [] mask [false])
    // CHECK: mqtopt.gphase({{.*}} static [] mask [false])

    // SX† gate also decomposes to RZ -> RY -> RZ
    // CHECK: %[[RZ3:.*]] = mqtopt.rz({{.*}} static [] mask [false]) %[[RZ2]] : !mqtopt.Qubit
    // CHECK: %[[RY2:.*]] = mqtopt.ry({{.*}} static [] mask [false]) %[[RZ3]] : !mqtopt.Qubit
    // CHECK: %[[RZ4:.*]] = mqtopt.rz({{.*}} static [] mask [false]) %[[RY2]] : !mqtopt.Qubit

    // S, S†, T, T†
    // CHECK: %[[S:.*]] = mqtopt.s(static [] mask []) %[[RZ4]] : !mqtopt.Qubit
    // CHECK: %[[SDG:.*]] = mqtopt.s(static [] mask []) %[[S]] : !mqtopt.Qubit
    // CHECK: %[[T:.*]] = mqtopt.t(static [] mask []) %[[SDG]] : !mqtopt.Qubit
    // CHECK: %[[TDG:.*]] = mqtopt.t(static [] mask []) %[[T]] : !mqtopt.Qubit

    // Load control qubit
    // CHECK: %[[Q1:.*]] = memref.load %[[ALLOC]][{{.*}}] : memref<2x!mqtopt.Qubit>

    // Controlled Clifford+T gates
    // Controlled Hadamard
    // CHECK: %[[CH_T:.*]], %[[CH_C:.*]] = mqtopt.h(static [] mask []) %[[TDG]] ctrl %[[Q1]]
    // CHECK-SAME: : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // Controlled SX decomposes to controlled RZ -> RY -> RZ + gphase
    // CHECK: %[[CRZ1_T:.*]], %[[CRZ1_C:.*]] = mqtopt.rz({{.*}} static [] mask [false]) %[[CH_T]] ctrl %[[CH_C]]
    // CHECK-SAME: : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CRY1_T:.*]], %[[CRY1_C:.*]] = mqtopt.ry({{.*}} static [] mask [false]) %[[CRZ1_C]] ctrl %[[CRZ1_T]]
    // CHECK-SAME: : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CRZ2_T:.*]], %[[CRZ2_C:.*]] = mqtopt.rz({{.*}} static [] mask [false]) %[[CRY1_C]] ctrl %[[CRY1_T]]
    // CHECK-SAME: : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: mqtopt.gphase({{.*}} static [] mask [false]) ctrl

    // Controlled S
    // CHECK: %[[CS_T:.*]], %[[CS_C:.*]] = mqtopt.s(static [] mask []) %[[CRZ2_C]] ctrl %[[CRZ2_T]]
    // CHECK-SAME: : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // Controlled S† (represented as controlled phase gate)
    // CHECK: %[[CSDG_T:.*]], %[[CSDG_C:.*]] = mqtopt.p({{.*}} static [] mask [false]) %[[CS_T]] ctrl %[[CS_C]]
    // CHECK-SAME: : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // Controlled T
    // CHECK: %[[CT_T:.*]], %[[CT_C:.*]] = mqtopt.t(static [] mask []) %[[CSDG_C]] ctrl %[[CSDG_T]]
    // CHECK-SAME: : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // Controlled T† (represented as controlled phase gate)
    // CHECK: %[[CTDG_T:.*]], %[[CTDG_C:.*]] = mqtopt.p({{.*}} static [] mask [false]) %[[CT_T]] ctrl %[[CT_C]]
    // CHECK-SAME: : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // Reinsertion
    // CHECK: memref.store {{.*}}, %[[ALLOC]][{{.*}}] : memref<2x!mqtopt.Qubit>
    // CHECK: memref.store {{.*}}, %[[ALLOC]][{{.*}}] : memref<2x!mqtopt.Qubit>
    // CHECK: memref.dealloc %[[ALLOC]] : memref<2x!mqtopt.Qubit>
    """
    _run_filecheck(mlir_after_mqtopt, check_to_mqtopt, "Clifford: CatalystQuantum to MQTOpt")

    # Verify MQTOpt → CatalystQuantum conversion
    check_to_catalyst = """
    // CHECK: func.func {{.*}}@circuit
    // CHECK: %[[QREG:.*]] = quantum.alloc({{.*}}) : !quantum.reg
    // CHECK: %[[Q0:.*]] = quantum.extract %[[QREG]][{{.*}}] : !quantum.reg -> !quantum.bit

    // Uncontrolled Clifford+T gates
    // Hadamard
    // CHECK: %[[H:.*]] = quantum.custom "Hadamard"() %[[Q0]] : !quantum.bit

    // SX decomposed to RZ -> RY -> RZ
    // CHECK: %[[RZ1:.*]] = quantum.custom "RZ"({{.*}}) %[[H]] : !quantum.bit
    // CHECK: %[[RY1:.*]] = quantum.custom "RY"({{.*}}) %[[RZ1]] : !quantum.bit
    // CHECK: %[[RZ2:.*]] = quantum.custom "RZ"({{.*}}) %[[RY1]] : !quantum.bit

    // SX† decomposed to RZ -> RY -> RZ
    // CHECK: %[[RZ3:.*]] = quantum.custom "RZ"({{.*}}) %[[RZ2]] : !quantum.bit
    // CHECK: %[[RY2:.*]] = quantum.custom "RY"({{.*}}) %[[RZ3]] : !quantum.bit
    // CHECK: %[[RZ4:.*]] = quantum.custom "RZ"({{.*}}) %[[RY2]] : !quantum.bit

    // S, S†, T, T†
    // CHECK: %[[S:.*]] = quantum.custom "S"() %[[RZ4]] : !quantum.bit
    // CHECK: %[[SDG:.*]] = quantum.custom "S"() %[[S]] : !quantum.bit
    // CHECK: %[[T:.*]] = quantum.custom "T"() %[[SDG]] : !quantum.bit
    // CHECK: %[[TDG:.*]] = quantum.custom "T"() %[[T]] : !quantum.bit

    // Extract control qubit
    // CHECK: %[[Q1:.*]] = quantum.extract %[[QREG]][{{.*}}] : !quantum.reg -> !quantum.bit

    // Controlled Clifford+T gates
    // Controlled Hadamard
    // CHECK: %[[CH_T:.*]], %[[CH_C:.*]] = quantum.custom "Hadamard"() %[[TDG]] ctrls(%[[Q1]])
    // CHECK-SAME: ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit

    // Controlled SX decomposed to CRZ -> CRY -> CRZ
    // CHECK: %[[CRZ1_T:.*]], %[[CRZ1_C:.*]] = quantum.custom "CRZ"({{.*}}) %[[CH_T]] ctrls(%[[CH_C]])
    // CHECK-SAME: ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CRY_T:.*]], %[[CRY_C:.*]] = quantum.custom "CRY"({{.*}}) %[[CRZ1_C]] ctrls(%[[CRZ1_T]])
    // CHECK-SAME: ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CRZ2_T:.*]], %[[CRZ2_C:.*]] = quantum.custom "CRZ"({{.*}}) %[[CRY_C]] ctrls(%[[CRY_T]])
    // CHECK-SAME: ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit

    // Controlled S
    // CHECK: %[[CS_T:.*]], %[[CS_C:.*]] = quantum.custom "S"() %[[CRZ2_C]] ctrls(%[[CRZ2_T]])
    // CHECK-SAME: ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit

    // Controlled S† (as ControlledPhaseShift)
    // CHECK: %[[CSDG_T:.*]], %[[CSDG_C:.*]] = quantum.custom "ControlledPhaseShift"({{.*}}) %[[CS_T]]
    // CHECK-SAME: ctrls(%[[CS_C]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit

    // Controlled T
    // CHECK: %[[CT_T:.*]], %[[CT_C:.*]] = quantum.custom "T"() %[[CSDG_C]] ctrls(%[[CSDG_T]])
    // CHECK-SAME: ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit

    // Controlled T† (as ControlledPhaseShift)
    // CHECK: %[[CTDG_T:.*]], %[[CTDG_C:.*]] = quantum.custom "ControlledPhaseShift"({{.*}}) %[[CT_T]]
    // CHECK-SAME: ctrls(%[[CT_C]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit

    // Reinsertion
    // CHECK: quantum.insert %[[QREG]][{{.*}}], {{.*}} : !quantum.reg, !quantum.bit
    // CHECK: quantum.insert %[[QREG]][{{.*}}], {{.*}} : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc %[[QREG]] : !quantum.reg
    """
    _run_filecheck(mlir_after_roundtrip, check_to_catalyst, "Clifford: MQTOpt to CatalystQuantum")


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

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found
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

        # Controlled multi-qubit gates (adding extra controls)
        qml.ctrl(qml.CNOT(wires=[0, 1]), control=2)
        qml.ctrl(qml.CY(wires=[0, 1]), control=2)
        qml.ctrl(qml.CZ(wires=[0, 1]), control=2)
        qml.ctrl(qml.Toffoli(wires=[0, 1, 2]), control=3)

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

    # Find where MLIR files are generated (relative to cwd where pytest is run)
    # Catalyst generates MLIR files in the current working directory
    mlir_dir = Path.cwd()

    # Read the intermediate MLIR files
    mlir_to_mqtopt = mlir_dir / "3_to-mqtopt.mlir"
    mlir_to_catalyst = mlir_dir / "4_MQTOptToCatalystQuantum.mlir"

    if not mlir_to_mqtopt.exists() or not mlir_to_catalyst.exists():
        available_files = list(mlir_dir.glob("*.mlir"))
        msg = f"Expected MLIR files not found in {mlir_dir}.\nAvailable files: {[f.name for f in available_files]}"
        raise FileNotFoundError(msg)

    with Path(mlir_to_mqtopt).open("r", encoding="utf-8") as f:
        mlir_after_mqtopt = f.read()
    with Path(mlir_to_catalyst).open("r", encoding="utf-8") as f:
        mlir_after_roundtrip = f.read()

    # Verify CatalystQuantum → MQTOpt conversion
    check_to_mqtopt = """
    // CHECK: func.func {{.*}}@circuit
    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<4x!mqtopt.Qubit>
    // CHECK: %[[Q0:.*]] = memref.load %[[ALLOC]][{{.*}}] : memref<4x!mqtopt.Qubit>

    // Uncontrolled Pauli gates
    // CHECK: %[[X1:.*]] = mqtopt.x({{.*}}) %[[Q0]] : !mqtopt.Qubit
    // CHECK: %[[Y1:.*]] = mqtopt.y({{.*}}) %[[X1]] : !mqtopt.Qubit
    // CHECK: %[[Z1:.*]] = mqtopt.z({{.*}}) %[[Y1]] : !mqtopt.Qubit
    // CHECK: %[[I1:.*]] = mqtopt.i({{.*}}) %[[Z1]] : !mqtopt.Qubit

    // Load Q1 for controlled Pauli gates
    // CHECK: %[[Q1:.*]] = memref.load %[[ALLOC]][{{.*}}] : memref<4x!mqtopt.Qubit>

    // Controlled Pauli gates (single control)
    // CHECK: %[[T1:.*]], %[[C1:.*]] = mqtopt.x({{.*}}) %[[I1]] ctrl %[[Q1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[T2:.*]], %[[C2:.*]] = mqtopt.y({{.*}}) %[[C1]] ctrl %[[T1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[T3:.*]], %[[C3:.*]] = mqtopt.z({{.*}}) %[[C2]] ctrl %[[T2]] : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // Two-qubit controlled gates (CNOT, CY, CZ as X, Y, Z with ctrl)
    // CHECK: %[[T4:.*]], %[[C4:.*]] = mqtopt.x({{.*}}) %[[T3]] ctrl %[[C3]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[T5:.*]], %[[C5:.*]] = mqtopt.y({{.*}}) %[[C4]] ctrl %[[T4]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[T6:.*]], %[[C6:.*]] = mqtopt.z({{.*}}) %[[C5]] ctrl %[[T5]] : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // Load Q2 for Toffoli
    // CHECK: %[[Q2:.*]] = memref.load %[[ALLOC]][{{.*}}] : memref<4x!mqtopt.Qubit>

    // Toffoli (X with 2 controls)
    // CHECK: %[[T7:.*]], %[[C7:.*]]:2 = mqtopt.x({{.*}}) %[[Q2]] ctrl %[[T6]], %[[C6]]
    // CHECK-SAME: : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit

    // Controlled two-qubit gates (add extra control) - qubit ordering from actual MLIR
    // CHECK: %[[T8:.*]], %[[C8:.*]]:2 = mqtopt.x({{.*}}) %[[C7]]#0 ctrl %[[C7]]#1, %[[T7]]
    // CHECK-SAME: : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[T9:.*]], %[[C9:.*]]:2 = mqtopt.y({{.*}}) %[[C8]]#1 ctrl %[[T8]], %[[C8]]#0
    // CHECK-SAME: : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[T10:.*]], %[[C10:.*]]:2 = mqtopt.z({{.*}}) %[[T9]] ctrl %[[C9]]#0, %[[C9]]#1
    // CHECK-SAME: : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit

    // Load Q3 for controlled Toffoli
    // CHECK: %[[Q3:.*]] = memref.load %[[ALLOC]][{{.*}}] : memref<4x!mqtopt.Qubit>

    // Controlled Toffoli (X with 3 controls)
    // CHECK: %[[T11:.*]], %[[C11:.*]]:3 = mqtopt.x({{.*}}) %[[C10]]#0 ctrl %[[Q3]], %[[C10]]#1, %[[T10]]
    // CHECK-SAME: : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit, !mqtopt.Qubit

    // Reinsertion
    // CHECK: memref.store {{.*}}, %[[ALLOC]][{{.*}}] : memref<4x!mqtopt.Qubit>
    // CHECK: memref.store {{.*}}, %[[ALLOC]][{{.*}}] : memref<4x!mqtopt.Qubit>
    // CHECK: memref.dealloc %[[ALLOC]] : memref<4x!mqtopt.Qubit>
    """
    _run_filecheck(mlir_after_mqtopt, check_to_mqtopt, "Pauli: CatalystQuantum to MQTOpt")

    # Verify MQTOpt → CatalystQuantum conversion
    check_to_catalyst = """
    // CHECK: func.func {{.*}}@circuit
    // CHECK: %[[QREG:.*]] = quantum.alloc({{.*}}) : !quantum.reg

    // Qubits extracted as needed
    // CHECK: %[[Q0:.*]] = quantum.extract %[[QREG]][{{.*}}] : !quantum.reg -> !quantum.bit

    // Uncontrolled Pauli gates
    // CHECK: %[[X:.*]] = quantum.custom "PauliX"() %[[Q0]] : !quantum.bit
    // CHECK: %[[Y:.*]] = quantum.custom "PauliY"() %[[X]] : !quantum.bit
    // CHECK: %[[Z:.*]] = quantum.custom "PauliZ"() %[[Y]] : !quantum.bit
    // CHECK: %[[I:.*]] = quantum.custom "Identity"() %[[Z]] : !quantum.bit

    // Extract Q1 for controlled Pauli gates
    // CHECK: %[[Q1:.*]] = quantum.extract %[[QREG]][{{.*}}] : !quantum.reg -> !quantum.bit

    // Controlled Pauli gates using CNOT/CY/CZ
    // CHECK: quantum.custom "CNOT"() %[[I]] ctrls(%[[Q1]]) ctrlvals(
    // CHECK: quantum.custom "CY"() {{.*}} ctrls({{.*}}) ctrlvals(
    // CHECK: quantum.custom "CZ"() {{.*}} ctrls({{.*}}) ctrlvals(

    // Two-qubit controlled gates (CNOT, CY, CZ)
    // CHECK: quantum.custom "CNOT"() {{.*}} ctrls({{.*}}) ctrlvals(
    // CHECK: quantum.custom "CY"() {{.*}} ctrls({{.*}}) ctrlvals(
    // CHECK: quantum.custom "CZ"() {{.*}} ctrls({{.*}}) ctrlvals(

    // Extract Q2 for Toffoli
    // CHECK: %[[Q2:.*]] = quantum.extract %[[QREG]][{{.*}}] : !quantum.reg -> !quantum.bit

    // Toffoli
    // CHECK: quantum.custom "Toffoli"() {{.*}} ctrls({{.*}}, {{.*}}) ctrlvals(

    // Reinsertion
    // CHECK: quantum.insert %[[QREG]][{{.*}}], {{.*}} : !quantum.reg, !quantum.bit
    // CHECK: quantum.insert %[[QREG]][{{.*}}], {{.*}} : !quantum.reg, !quantum.bit
    // CHECK: quantum.insert %[[QREG]][{{.*}}], {{.*}} : !quantum.reg, !quantum.bit
    // CHECK: quantum.insert %[[QREG]][{{.*}}], {{.*}} : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc %[[QREG]] : !quantum.reg
    """
    _run_filecheck(mlir_after_roundtrip, check_to_catalyst, "Pauli: MQTOpt to CatalystQuantum")


def test_parameterized_gates_roundtrip() -> None:
    """Test roundtrip conversion of parameterized rotation gates.

    Mirrors: quantum_param.mlir
    Gates: RX, RY, RZ, PhaseShift, and their controlled variants (CRX, CRY)
    Note: MLIR test does NOT include CRZ, only CRX and CRY

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found
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

    # Find where MLIR files are generated (relative to cwd where pytest is run)
    # Catalyst generates MLIR files in the current working directory
    mlir_dir = Path.cwd()

    # Read the intermediate MLIR files
    mlir_to_mqtopt = mlir_dir / "3_to-mqtopt.mlir"
    mlir_to_catalyst = mlir_dir / "4_MQTOptToCatalystQuantum.mlir"

    if not mlir_to_mqtopt.exists() or not mlir_to_catalyst.exists():
        available_files = list(mlir_dir.glob("*.mlir"))
        msg = f"Expected MLIR files not found in {mlir_dir}.\nAvailable files: {[f.name for f in available_files]}"
        raise FileNotFoundError(msg)

    with Path(mlir_to_mqtopt).open("r", encoding="utf-8") as f:
        mlir_after_mqtopt = f.read()
    with Path(mlir_to_catalyst).open("r", encoding="utf-8") as f:
        mlir_after_roundtrip = f.read()

    # Verify CatalystQuantum → MQTOpt conversion
    check_to_mqtopt = """
    // CHECK: func.func {{.*}}@circuit
    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x!mqtopt.Qubit>
    // CHECK: %[[Q0:.*]] = memref.load %[[ALLOC]][{{.*}}] : memref<2x!mqtopt.Qubit>

    // Uncontrolled parameterized gates
    // CHECK: %[[RX:.*]] = mqtopt.rx({{.*}}) %[[Q0]] : !mqtopt.Qubit
    // CHECK: %[[RY:.*]] = mqtopt.ry({{.*}}) %[[RX]] : !mqtopt.Qubit
    // CHECK: %[[RZ:.*]] = mqtopt.rz({{.*}}) %[[RY]] : !mqtopt.Qubit
    // CHECK: %[[PS:.*]] = mqtopt.p({{.*}}) %[[RZ]] : !mqtopt.Qubit

    // Load Q1 for controlled parameterized gates
    // CHECK: %[[Q1:.*]] = memref.load %[[ALLOC]][{{.*}}] : memref<2x!mqtopt.Qubit>

    // Controlled parameterized gates (qubits swap roles after each operation like Pauli gates)
    // CHECK: %[[CRX_T:.*]], %[[CRX_C:.*]] = mqtopt.rx({{.*}}) %[[PS]] ctrl %[[Q1]]
    // CHECK-SAME: : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CRY_T:.*]], %[[CRY_C:.*]] = mqtopt.ry({{.*}}) %[[CRX_C]] ctrl %[[CRX_T]]
    // CHECK-SAME: : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // Reinsertion
    // CHECK: memref.store {{.*}}, %[[ALLOC]][{{.*}}] : memref<2x!mqtopt.Qubit>
    // CHECK: memref.store {{.*}}, %[[ALLOC]][{{.*}}] : memref<2x!mqtopt.Qubit>
    // CHECK: memref.dealloc %[[ALLOC]] : memref<2x!mqtopt.Qubit>
    """
    _run_filecheck(mlir_after_mqtopt, check_to_mqtopt, "Param: CatalystQuantum to MQTOpt")

    # Verify MQTOpt → CatalystQuantum conversion
    check_to_catalyst = """
    // CHECK: func.func {{.*}}@circuit
    // CHECK: %[[QREG:.*]] = quantum.alloc({{.*}}) : !quantum.reg

    // Q0 is extracted first for uncontrolled gates
    // CHECK: %[[Q0:.*]] = quantum.extract %[[QREG]][{{.*}}] : !quantum.reg -> !quantum.bit

    // Uncontrolled parameterized gates
    // CHECK: %[[RX:.*]] = quantum.custom "RX"({{.*}}) %[[Q0]] : !quantum.bit
    // CHECK: %[[RY:.*]] = quantum.custom "RY"({{.*}}) %[[RX]] : !quantum.bit
    // CHECK: %[[RZ:.*]] = quantum.custom "RZ"({{.*}}) %[[RY]] : !quantum.bit
    // CHECK: %[[PS:.*]] = quantum.custom "PhaseShift"({{.*}}) %[[RZ]] : !quantum.bit

    // Q1 is extracted lazily right before controlled gates
    // CHECK: %[[Q1:.*]] = quantum.extract %[[QREG]][{{.*}}] : !quantum.reg -> !quantum.bit

    // Controlled parameterized gates (qubits swap after each operation)
    // CRX: target=%[[PS]], control=%[[Q1]]
    // CHECK: %[[CRX_T:.*]], %[[CRX_C:.*]] = quantum.custom "CRX"({{.*}}) %[[PS]] ctrls(%[[Q1]])
    // CHECK-SAME: ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CRY: target=%[[CRX_C]] (previous control), control=%[[CRX_T]] (previous target)
    // CHECK: quantum.custom "CRY"({{.*}}) %[[CRX_C]] ctrls(%[[CRX_T]]) ctrlvals(%true{{.*}})
    // CHECK-SAME: : !quantum.bit ctrls !quantum.bit

    // Reinsertion
    // CHECK: quantum.insert %[[QREG]][{{.*}}], {{.*}} : !quantum.reg, !quantum.bit
    // CHECK: quantum.insert %[[QREG]][{{.*}}], {{.*}} : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc %[[QREG]] : !quantum.reg
    """
    _run_filecheck(mlir_after_roundtrip, check_to_catalyst, "Param: MQTOpt to CatalystQuantum")


def test_entangling_gates_roundtrip() -> None:
    """Test roundtrip conversion of entangling/permutation gates.

    Mirrors: quantum_entangling.mlir
    Gates: SWAP, ISWAP, ISWAP†, ECR, and their controlled variants

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found
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

    # Find where MLIR files are generated (relative to cwd where pytest is run)
    # Catalyst generates MLIR files in the current working directory
    mlir_dir = Path.cwd()

    # Read the intermediate MLIR files
    mlir_to_mqtopt = mlir_dir / "3_to-mqtopt.mlir"
    mlir_to_catalyst = mlir_dir / "4_MQTOptToCatalystQuantum.mlir"

    if not mlir_to_mqtopt.exists() or not mlir_to_catalyst.exists():
        available_files = list(mlir_dir.glob("*.mlir"))
        msg = f"Expected MLIR files not found in {mlir_dir}.\nAvailable files: {[f.name for f in available_files]}"
        raise FileNotFoundError(msg)

    with Path(mlir_to_mqtopt).open("r", encoding="utf-8") as f:
        mlir_after_mqtopt = f.read()
    with Path(mlir_to_catalyst).open("r", encoding="utf-8") as f:
        mlir_after_roundtrip = f.read()

    # Verify CatalystQuantum → MQTOpt conversion
    check_to_mqtopt = """
    // CHECK: func.func {{.*}}@circuit
    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<3x!mqtopt.Qubit>

    // Qubits loaded as needed
    // CHECK: %[[Q0:.*]] = memref.load %[[ALLOC]][{{.*}}] : memref<3x!mqtopt.Qubit>
    // CHECK: %[[Q1:.*]] = memref.load %[[ALLOC]][{{.*}}] : memref<3x!mqtopt.Qubit>

    // Uncontrolled entangling gates (ISWAP/ECR get heavily decomposed, but SWAP should be visible)
    // CHECK: mqtopt.swap({{.*}}) %[[Q0]], %[[Q1]] : !mqtopt.Qubit, !mqtopt.Qubit

    // After decompositions, Q2 is loaded for controlled gates
    // CHECK: %[[Q2:.*]] = memref.load %[[ALLOC]][{{.*}}] : memref<3x!mqtopt.Qubit>

    // Controlled swap gate
    // CHECK: mqtopt.swap({{.*}}) {{.*}} ctrl %[[Q2]] : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit

    // Reinsertion
    // CHECK: memref.store {{.*}}, %[[ALLOC]][{{.*}}] : memref<3x!mqtopt.Qubit>
    // CHECK: memref.store {{.*}}, %[[ALLOC]][{{.*}}] : memref<3x!mqtopt.Qubit>
    // CHECK: memref.store {{.*}}, %[[ALLOC]][{{.*}}] : memref<3x!mqtopt.Qubit>
    // CHECK: memref.dealloc %[[ALLOC]] : memref<3x!mqtopt.Qubit>
    """
    _run_filecheck(mlir_after_mqtopt, check_to_mqtopt, "Entangling: CatalystQuantum to MQTOpt")

    # Verify MQTOpt → CatalystQuantum conversion (simplified - ISWAP/ECR are heavily decomposed)
    check_to_catalyst = """
    // CHECK: func.func {{.*}}@circuit
    // CHECK: %[[QREG:.*]] = quantum.alloc({{.*}}) : !quantum.reg

    // Qubits extracted as needed
    // CHECK: %[[Q0:.*]] = quantum.extract %[[QREG]][{{.*}}] : !quantum.reg -> !quantum.bit
    // CHECK: %[[Q1:.*]] = quantum.extract %[[QREG]][{{.*}}] : !quantum.reg -> !quantum.bit

    // SWAP is visible, but ISWAP/ECR/adjISWAP are heavily decomposed into primitives (H, S, CNOT, RZ, RY chains)
    // CHECK: quantum.custom "SWAP"() %[[Q0]], %[[Q1]] : !quantum.bit, !quantum.bit

    // After all decompositions, Q2 extracted for controlled gates
    // CHECK: %[[Q2:.*]] = quantum.extract %[[QREG]][{{.*}}] : !quantum.reg -> !quantum.bit

    // Controlled swap
    // CHECK: quantum.custom "CSWAP"() {{.*}} ctrls(%[[Q2]]) ctrlvals(

    // Reinsertion
    // CHECK: quantum.insert %[[QREG]][{{.*}}], {{.*}} : !quantum.reg, !quantum.bit
    // CHECK: quantum.insert %[[QREG]][{{.*}}], {{.*}} : !quantum.reg, !quantum.bit
    // CHECK: quantum.insert %[[QREG]][{{.*}}], {{.*}} : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc %[[QREG]] : !quantum.reg
    """
    _run_filecheck(mlir_after_roundtrip, check_to_catalyst, "Entangling: MQTOpt to CatalystQuantum")


def test_ising_gates_roundtrip() -> None:
    """Test roundtrip conversion of Ising-type gates.

    Mirrors: quantum_ising.mlir
    Gates: IsingXY, IsingXX, IsingYY, IsingZZ, and their controlled variants
    Note: IsingXY takes 2 parameters in MLIR (phi and beta)

    Raises:
        FileNotFoundError: If intermediate MLIR files are not found
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

    # Find where MLIR files are generated (relative to cwd where pytest is run)
    # Catalyst generates MLIR files in the current working directory
    # This works regardless of where pytest is run from (locally or CI)
    test_file_dir = Path(__file__).parent
    mlir_dir = Path.cwd()
    test_file_dir.parent / "Conversion"

    # Read the intermediate MLIR files
    mlir_to_mqtopt = mlir_dir / "3_to-mqtopt.mlir"
    mlir_to_catalyst = mlir_dir / "4_MQTOptToCatalystQuantum.mlir"

    if not mlir_to_mqtopt.exists() or not mlir_to_catalyst.exists():
        # Fallback: list what files actually exist for debugging
        available_files = list(mlir_dir.glob("*.mlir"))
        msg = f"Expected MLIR files not found in {mlir_dir}.\nAvailable files: {[f.name for f in available_files]}"
        raise FileNotFoundError(msg)

    with Path(mlir_to_mqtopt).open("r", encoding="utf-8") as f:
        mlir_after_mqtopt = f.read()
    with Path(mlir_to_catalyst).open("r", encoding="utf-8") as f:
        mlir_after_roundtrip = f.read()

    # Verify CatalystQuantum → MQTOpt conversion with FileCheck
    check_to_mqtopt = """
    // CHECK: func.func {{.*}}@circuit
    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<3x!mqtopt.Qubit>
    // CHECK: %[[Q0:.*]] = memref.load %[[ALLOC]][{{.*}}] : memref<3x!mqtopt.Qubit>
    // CHECK: %[[Q1:.*]] = memref.load %[[ALLOC]][{{.*}}] : memref<3x!mqtopt.Qubit>

    // Uncontrolled Ising gates
    // CHECK: %[[XY_OUT:.*]]:2 = mqtopt.xx_plus_yy({{.*}}) %[[Q0]], %[[Q1]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[XX_OUT:.*]]:2 = mqtopt.rxx({{.*}}) %[[XY_OUT]]#0, %[[XY_OUT]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[YY_OUT:.*]]:2 = mqtopt.ryy({{.*}}) %[[XX_OUT]]#0, %[[XX_OUT]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[ZZ_OUT:.*]]:2 = mqtopt.rzz({{.*}}) %[[YY_OUT]]#0, %[[YY_OUT]]#1 : !mqtopt.Qubit, !mqtopt.Qubit

    // Controlled Ising gates (control qubit loaded here)
    // CHECK: %[[Q2:.*]] = memref.load %[[ALLOC]][{{.*}}] : memref<3x!mqtopt.Qubit>
    // CHECK: %[[CXY_OUT:.*]]:2, %[[CTRL1:.*]] = mqtopt.xx_plus_yy({{.*}}) %[[ZZ_OUT]]#0, %[[ZZ_OUT]]#1 ctrl %[[Q2]]
    // CHECK-SAME: : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CXX_OUT:.*]]:2, %[[CTRL2:.*]] = mqtopt.rxx({{.*}}) %[[CXY_OUT]]#0, %[[CXY_OUT]]#1 ctrl %[[CTRL1]]
    // CHECK-SAME: : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CYY_OUT:.*]]:2, %[[CTRL3:.*]] = mqtopt.ryy({{.*}}) %[[CXX_OUT]]#0, %[[CXX_OUT]]#1 ctrl %[[CTRL2]]
    // CHECK-SAME: : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CZZ_OUT:.*]]:2, %[[CTRL4:.*]] = mqtopt.rzz({{.*}}) %[[CYY_OUT]]#0, %[[CYY_OUT]]#1 ctrl %[[CTRL3]]
    // CHECK-SAME: : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit

    // Reinsertion
    // CHECK: memref.store %[[CZZ_OUT]]#0, %[[ALLOC]][{{.*}}] : memref<3x!mqtopt.Qubit>
    // CHECK: memref.store %[[CZZ_OUT]]#1, %[[ALLOC]][{{.*}}] : memref<3x!mqtopt.Qubit>
    // CHECK: memref.store %[[CTRL4]], %[[ALLOC]][{{.*}}] : memref<3x!mqtopt.Qubit>
    // CHECK: memref.dealloc %[[ALLOC]] : memref<3x!mqtopt.Qubit>
    """
    _run_filecheck(mlir_after_mqtopt, check_to_mqtopt, "Ising: CatalystQuantum to MQTOpt")

    # Verify MQTOpt → CatalystQuantum conversion with FileCheck
    # Based on mqtopt_ising.mlir reference test
    check_to_catalyst = """
    // CHECK: func.func {{.*}}@circuit
    // CHECK: %[[QREG:.*]] = quantum.alloc({{.*}}) : !quantum.reg
    // CHECK: %[[Q0:.*]] = quantum.extract %[[QREG]][{{.*}}] : !quantum.reg -> !quantum.bit
    // CHECK: %[[Q1:.*]] = quantum.extract %[[QREG]][{{.*}}] : !quantum.reg -> !quantum.bit

    // Uncontrolled Ising gates
    // IsingXY is decomposed: RZ -> IsingXY -> RZ
    // CHECK: %[[RZ0:.*]] = quantum.custom "RZ"({{.*}}) %[[Q1]] : !quantum.bit
    // CHECK: %[[XY:.*]]:2 = quantum.custom "IsingXY"({{.*}}) %[[Q0]], %[[RZ0]] : !quantum.bit, !quantum.bit
    // CHECK: %[[RZ1:.*]] = quantum.custom "RZ"({{.*}}) %[[XY]]#1 : !quantum.bit

    // IsingXX, IsingYY, IsingZZ gates
    // CHECK: %[[XX:.*]]:2 = quantum.custom "IsingXX"({{.*}}) %[[XY]]#0, %[[RZ1]] : !quantum.bit, !quantum.bit
    // CHECK: %[[YY:.*]]:2 = quantum.custom "IsingYY"({{.*}}) %[[XX]]#0, %[[XX]]#1 : !quantum.bit, !quantum.bit
    // CHECK: %[[ZZ:.*]]:2 = quantum.custom "IsingZZ"({{.*}}) %[[YY]]#0, %[[YY]]#1 : !quantum.bit, !quantum.bit

    // Controlled Ising gates (with ctrls)
    // Extract control qubit
    // CHECK: %[[Q2:.*]] = quantum.extract %[[QREG]][{{.*}}] : !quantum.reg -> !quantum.bit

    // Controlled IsingXY: RZ(ctrl) -> IsingXY(ctrl) -> RZ(ctrl)
    // CHECK: %[[CRZ0:.*]], %[[CTRL1:.*]] = quantum.custom "RZ"({{.*}}) %[[ZZ]]#1 ctrls(%[[Q2]])
    // CHECK-SAME: ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CXY:.*]]:2, %[[CTRL2:.*]] = quantum.custom "IsingXY"({{.*}}) %[[ZZ]]#0, %[[CRZ0]]
    // CHECK-SAME: ctrls(%[[CTRL1]]) ctrlvals(%true{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CRZ1:.*]], %[[CTRL3:.*]] = quantum.custom "RZ"({{.*}}) %[[CXY]]#1 ctrls(%[[CTRL2]])
    // CHECK-SAME: ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit

    // Controlled IsingXX, IsingYY, IsingZZ
    // CHECK: %[[CXX:.*]]:2, %[[CTRL4:.*]] = quantum.custom "IsingXX"({{.*}}) %[[CXY]]#0, %[[CRZ1]]
    // CHECK-SAME: ctrls(%[[CTRL3]]) ctrlvals(%true{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CYY:.*]]:2, %[[CTRL5:.*]] = quantum.custom "IsingYY"({{.*}}) %[[CXX]]#0, %[[CXX]]#1
    // CHECK-SAME: ctrls(%[[CTRL4]]) ctrlvals(%true{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CZZ:.*]]:2, %[[CTRL6:.*]] = quantum.custom "IsingZZ"({{.*}}) %[[CYY]]#0, %[[CYY]]#1
    // CHECK-SAME: ctrls(%[[CTRL5]]) ctrlvals(%true{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit

    // Reinsertion
    // CHECK: quantum.insert %[[QREG]][{{.*}}], %[[CZZ]]#0 : !quantum.reg, !quantum.bit
    // CHECK: quantum.insert %[[QREG]][{{.*}}], %[[CZZ]]#1 : !quantum.reg, !quantum.bit
    // CHECK: quantum.insert %[[QREG]][{{.*}}], %[[CTRL6]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc %[[QREG]] : !quantum.reg
    """
    _run_filecheck(mlir_after_roundtrip, check_to_catalyst, "Ising: MQTOpt to CatalystQuantum")

    # Remove all intermediate files created during the test
    for mlir_file in mlir_dir.glob("*.mlir"):
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

    @qml.qjit(target="mlir", autograph=True)
    def module() -> None:
        return circuit()

    # This will execute the pass and return the final MLIR
    mlir_opt = module.mlir_opt
    assert mlir_opt
