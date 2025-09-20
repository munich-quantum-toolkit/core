// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: catalyst --tool=opt \
// RUN:   --load-pass-plugin=%mqt_plugin_path% \
// RUN:   --load-dialect-plugin=%mqt_plugin_path% \
// RUN:   --debug \
// RUN:   --catalyst-pipeline="builtin.module(catalystquantum-to-mqtopt)" \
// RUN:   %s | FileCheck %s


// ============================================================================
// Pauli family (X, Y, Z) and controlled variants
// Groups: Allocation & extraction / Uncontrolled / Controlled / Reinsertion
// ============================================================================
module {
  // CHECK-LABEL: func.func @testCatalystQuantumToMQTOptPauliGates
  func.func @testCatalystQuantumToMQTOptPauliGates() {
    // --- Allocation & extraction ---------------------------------------------------------------
    // CHECK: %[[QREG:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister
    // CHECK: %[[QR1:.*]], %[[Q0:.*]] = "mqtopt.extractQubit"(%[[QREG]]) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[QR2:.*]], %[[Q1:.*]] = "mqtopt.extractQubit"(%[[QR1]]) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[QR3:.*]], %[[Q2:.*]] = "mqtopt.extractQubit"(%[[QR2]]) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // --- Uncontrolled -------------------------------------------------------------------------
    // CHECK: %[[X:.*]] = mqtopt.x( static [] mask []) %[[Q0]] : !mqtopt.Qubit
    // CHECK: %[[Y:.*]] = mqtopt.y( static [] mask []) %[[X]] : !mqtopt.Qubit
    // CHECK: %[[Z:.*]] = mqtopt.z( static [] mask []) %[[Y]] : !mqtopt.Qubit
    // CHECK: %[[I:.*]] = mqtopt.i( static [] mask []) %[[Z]] : !mqtopt.Qubit

    // --- Controlled ----------------------------------------------------------------------------
    // CHECK: %[[CX_T:.*]], %[[CX_C:.*]] = mqtopt.x( static [] mask []) %[[I]] ctrl %[[Q1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CY_T:.*]], %[[CY_C:.*]] = mqtopt.y( static [] mask []) %[[CX_T]] ctrl %[[CX_C]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CZ_T:.*]], %[[CZ_C:.*]] = mqtopt.z( static [] mask []) %[[CY_T]] ctrl %[[CY_C]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CI_T:.*]], %[[CI_C:.*]] = mqtopt.i( static [] mask []) %[[CZ_T]] ctrl %[[CZ_C]] : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // --- Multi-controlled ----------------------------------------------------------------------
    // CHECK: %[[MCX_T:.*]], %[[MCX_C:.*]]:2 = mqtopt.x( static [] mask []) %[[CI_T]] ctrl %[[CI_C]], %[[Q2]] : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit

    // --- Reinsertion ---------------------------------------------------------------------------
    // CHECK: %[[R1:.*]] = "mqtopt.insertQubit"(%[[QR3]], %[[MCX_C]]#1) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %[[R2:.*]] = "mqtopt.insertQubit"(%[[R1]], %[[MCX_C]]#0) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %[[R3:.*]] = "mqtopt.insertQubit"(%[[R2]], %[[MCX_T]]) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: "mqtopt.deallocQubitRegister"(%[[R3]]) : (!mqtopt.QubitRegister) -> ()

    // Prepare qubits
    %qreg = quantum.alloc( 3) : !quantum.reg
    %q0 = quantum.extract %qreg[ 0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %qreg[ 1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %qreg[ 2] : !quantum.reg -> !quantum.bit

    // Non-controlled Pauli gates
    %q0_x = quantum.custom "PauliX"() %q0 : !quantum.bit
    %q0_y = quantum.custom "PauliY"() %q0_x : !quantum.bit
    %q0_z = quantum.custom "PauliZ"() %q0_y : !quantum.bit
    %q0_i = quantum.custom "Identity"() %q0_z : !quantum.bit

    // Controlled Pauli gates
    %true = arith.constant true
    %q0_cx, %q1_cx = quantum.custom "CNOT"() %q0_i ctrls(%q1) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %q0_cy, %q1_cy = quantum.custom "CY"() %q0_cx ctrls(%q1_cx) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %q0_cz, %q1_cz = quantum.custom "CZ"() %q0_cy ctrls(%q1_cy) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %q0_ci, %q1_ci = quantum.custom "Identity"() %q0_cz ctrls(%q1_cz) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %q0_ccx, %q1_ccx, %q2_ccx = quantum.custom "PauliX"() %q0_ci ctrls(%q1_ci, %q2) ctrlvals(%true, %true) : !quantum.bit ctrls !quantum.bit, !quantum.bit

    // Release qubits
    %qreg1 = quantum.insert %qreg[ 2], %q2_ccx : !quantum.reg, !quantum.bit
    %qreg2 = quantum.insert %qreg1[ 1], %q1_ccx : !quantum.reg, !quantum.bit
    %qreg3 = quantum.insert %qreg2[ 0], %q0_ccx : !quantum.reg, !quantum.bit
    quantum.dealloc %qreg3 : !quantum.reg
    return
  }
}
