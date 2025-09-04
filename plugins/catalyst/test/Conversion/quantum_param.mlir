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
// Parameterized gates RX/RY/RZ, PhaseShift and controlled variants
// Groups: Constants / Allocation & extraction / Uncontrolled / Controlled / Reinsertion
// ============================================================================
module {
  // CHECK-LABEL: func.func @testCatalystQuantumToMQTOptParameterized
  func.func @testCatalystQuantumToMQTOptParameterized() {
    // --- Constants & Allocation & extraction ---------------------------------------------------
    // CHECK: %cst = arith.constant 3.000000e-01 : f64
    // CHECK: %[[QREG:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
    // CHECK: %[[QR1:.*]], %[[Q0:.*]] = "mqtopt.extractQubit"(%[[QREG]]) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[QR2:.*]], %[[Q1:.*]] = "mqtopt.extractQubit"(%[[QR1]]) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // --- Uncontrolled -------------------------------------------------------------------------
    // CHECK: %[[RX:.*]] = mqtopt.rx(%cst static [] mask [false]) %[[Q0]] : !mqtopt.Qubit
    // CHECK: %[[RY:.*]] = mqtopt.ry(%cst static [] mask [false]) %[[RX]] : !mqtopt.Qubit
    // CHECK: %[[RZ:.*]] = mqtopt.rz(%cst static [] mask [false]) %[[RY]] : !mqtopt.Qubit
    // CHECK: %[[PS:.*]] = mqtopt.p(%cst static [] mask [false]) %[[RZ]] : !mqtopt.Qubit

    // --- Controlled ----------------------------------------------------------------------------
    // CHECK: %[[CRX_T:.*]], %[[CRX_C:.*]] = mqtopt.rx(%cst static [] mask [false]) %[[PS]] ctrl %[[Q1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CRY_T:.*]], %[[CRY_C:.*]] = mqtopt.ry(%cst static [] mask [false]) %[[CRX_T]] ctrl %[[CRX_C]] : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // --- Reinsertion ---------------------------------------------------------------------------
    // CHECK: %[[R1:.*]] = "mqtopt.insertQubit"(%[[QR2]], %[[CRY_T]]) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %[[R2:.*]] = "mqtopt.insertQubit"(%[[R1]], %[[CRY_C]]) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: "mqtopt.deallocQubitRegister"(%[[R2]]) : (!mqtopt.QubitRegister) -> ()

    // Prepare qubits
    %angle = arith.constant 3.000000e-01 : f64
    %qreg = quantum.alloc( 2) : !quantum.reg
    %q0 = quantum.extract %qreg[ 0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %qreg[ 1] : !quantum.reg -> !quantum.bit

    // Non-controlled parameterized gates
    %q0_rx = quantum.custom "RX"(%angle) %q0 : !quantum.bit
    %q0_ry = quantum.custom "RY"(%angle) %q0_rx : !quantum.bit
    %q0_rz = quantum.custom "RZ"(%angle) %q0_ry : !quantum.bit
    %q0_p = quantum.custom "PhaseShift"(%angle) %q0_rz : !quantum.bit

    // Controlled parameterized gates
    %true = arith.constant true
    %q1_out, %q0_out = quantum.custom "RX"(%angle) %q0_p ctrls(%q1) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %q1_out2, %q0_out2 = quantum.custom "RY"(%angle) %q1_out ctrls(%q0_out) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit

    // Release qubits
    %qreg1 = quantum.insert %qreg[ 0], %q1_out2 : !quantum.reg, !quantum.bit
    %qreg2 = quantum.insert %qreg1[ 1], %q0_out2 : !quantum.reg, !quantum.bit
    quantum.dealloc %qreg2 : !quantum.reg
    return
  }
}
