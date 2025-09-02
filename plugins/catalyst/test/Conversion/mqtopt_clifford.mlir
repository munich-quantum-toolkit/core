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
// RUN:   --catalyst-pipeline="builtin.module(mqtopt-to-catalystquantum)" \
// RUN:   %s | FileCheck %s


// ============================================================================
// I, H, V, V†, S, S†, T, T† and controlled variants
// Groups: Allocation & extraction / Uncontrolled sequence / Controlled sequence / Reinsertion
// ============================================================================
module {
  // CHECK-LABEL: func.func @testMQTOptToCatalystQuantumCliffordTFamily
  func.func @testMQTOptToCatalystQuantumCliffordTFamily() {
    // --- Allocation & extraction ---------------------------------------------------------------
    // CHECK: %[[QREG:.*]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: %[[Q0_0:.*]] = quantum.extract %[[QREG]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: %[[Q1_0:.*]] = quantum.extract %[[QREG]][ 1] : !quantum.reg -> !quantum.bit

    // --- Uncontrolled sequence -----------------------------------------------------------------
    // CHECK: %[[I:.*]]   = quantum.custom "Identity"() %[[Q0_0]] : !quantum.bit
    // CHECK: %[[H:.*]]   = quantum.custom "Hadamard"() %[[I]] : !quantum.bit

    // V gate gets decomposed into a sequence of single-qubit rotations
    // CHECK: %[[CST:.*]] = arith.constant {{.*}} : f64
    // CHECK: %[[RZ1:.*]] = quantum.custom "RZ"(%[[CST]]) %[[H]] : !quantum.bit
    // CHECK: %[[RY1:.*]] = quantum.custom "RY"(%[[CST]]) %[[RZ1]] : !quantum.bit
    // CHECK: %[[RZ2:.*]] = quantum.custom "RZ"(%[[CST]]) %[[RY1]] adj : !quantum.bit


    // CHECK: %[[NEG_CST:.*]] = arith.constant -{{.*}} : f64
    // CHECK: %[[RZ3:.*]] = quantum.custom "RZ"(%[[NEG_CST]]) %[[RZ2]] adj : !quantum.bit
    // CHECK: %[[RY2:.*]] = quantum.custom "RY"(%[[NEG_CST]]) %[[RZ3]] : !quantum.bit
    // CHECK: %[[RZ4:.*]] = quantum.custom "RZ"(%[[NEG_CST]]) %[[RY2]] : !quantum.bit

    // CHECK: %[[S:.*]]   = quantum.custom "S"() %[[RZ4]] : !quantum.bit
    // CHECK: %[[SDG:.*]] = quantum.custom "S"() %[[S]] adj : !quantum.bit
    // CHECK: %[[T:.*]]   = quantum.custom "T"() %[[SDG]] : !quantum.bit
    // CHECK: %[[TDG:.*]] = quantum.custom "T"() %[[T]] adj : !quantum.bit

    // --- Controlled sequence -------------------------------------------------------------------
    // CHECK: %[[CH_T:.*]], %[[CH_C:.*]] = quantum.custom "Hadamard"() %[[TDG]] ctrls(%[[Q1_0]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit

    // CHECK: %[[CST:.*]] = arith.constant {{.*}} : f64
    // CHECK: %[[CRZ1_T:.*]], %[[CRZ1_C:.*]] = quantum.custom "RZ"(%[[CST]]) %[[CH_T]] ctrls(%[[CH_C]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CRY1_T:.*]], %[[CRY1_C:.*]] = quantum.custom "RY"(%[[CST]]) %[[CRZ1_T]] ctrls(%[[CRZ1_C]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CRZ2_T:.*]], %[[CRZ2_C:.*]] = quantum.custom "RZ"(%[[CST]]) %[[CRY1_T]] adj ctrls(%[[CRY1_C]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit

    // CHECK: %[[NEG_CST:.*]] = arith.constant -{{.*}} : f64
    // CHECK: %[[CRZ3_T:.*]], %[[CRZ3_C:.*]] = quantum.custom "RZ"(%[[NEG_CST]]) %[[CRZ2_T]] adj ctrls(%[[CRZ2_C]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CRY2_T:.*]], %[[CRY2_C:.*]] = quantum.custom "RY"(%[[NEG_CST]]) %[[CRZ3_T]] ctrls(%[[CRZ3_C]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CRZ4_T:.*]], %[[CRZ4_C:.*]] = quantum.custom "RZ"(%[[NEG_CST]]) %[[CRY2_T]] ctrls(%[[CRY2_C]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit

    // CHECK: %[[CS_T:.*]], %[[CS_C:.*]]   = quantum.custom "S"()   %[[CRZ4_T]] ctrls(%[[CRZ4_C]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CSD_T:.*]], %[[CSD_C:.*]] = quantum.custom "S"() %[[CS_T]] adj ctrls(%[[CS_C]])  ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CT_T:.*]], %[[CT_C:.*]]   = quantum.custom "T"()   %[[CSD_T]]  ctrls(%[[CSD_C]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CTD_T:.*]], %[[CTD_C:.*]] = quantum.custom "T"() %[[CT_T]]  adj ctrls(%[[CT_C]])  ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit

    // --- Reinsertion ---------------------------------------------------------------------------
    // CHECK: %[[R1:.*]] = quantum.insert %[[QREG]][ 0], %[[CTD_T]] : !quantum.reg, !quantum.bit
    // CHECK: %[[R2:.*]] = quantum.insert %[[R1]][ 1], %[[CTD_C]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc %[[R2]] : !quantum.reg

    // Prepare qubits
    %r0_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
    %r0_1, %q0_0 = "mqtopt.extractQubit"(%r0_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    %r0_2, %q1_0 = "mqtopt.extractQubit"(%r0_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // I/H/V/Vdg/S/Sdg/T/Tdg (non-controlled)
    %q0_1 = mqtopt.i()   %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.h()   %q0_1 : !mqtopt.Qubit
    %q0_3 = mqtopt.v()   %q0_2 : !mqtopt.Qubit
    %q0_4 = mqtopt.vdg() %q0_3 : !mqtopt.Qubit
    %q0_5 = mqtopt.s()   %q0_4 : !mqtopt.Qubit
    %q0_6 = mqtopt.sdg() %q0_5 : !mqtopt.Qubit
    %q0_7 = mqtopt.t()   %q0_6 : !mqtopt.Qubit
    %q0_8 = mqtopt.tdg() %q0_7 : !mqtopt.Qubit

    // Controlled H/V/Vdg/S/Sdg/T/Tdg
    %q0_9,  %q1_1 = mqtopt.h()   %q0_8 ctrl %q1_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_10, %q1_2 = mqtopt.v()   %q0_9 ctrl %q1_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_11, %q1_3 = mqtopt.vdg() %q0_10 ctrl %q1_2 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_12, %q1_4 = mqtopt.s()   %q0_11 ctrl %q1_3 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_13, %q1_5 = mqtopt.sdg() %q0_12 ctrl %q1_4 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_14, %q1_6 = mqtopt.t()   %q0_13 ctrl %q1_5 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_15, %q1_7 = mqtopt.tdg() %q0_14 ctrl %q1_6 : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // Release qubits
    %r0_3 = "mqtopt.insertQubit"(%r0_2, %q0_15)  <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    %r0_4 = "mqtopt.insertQubit"(%r0_3, %q1_7) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    "mqtopt.deallocQubitRegister"(%r0_4) : (!mqtopt.QubitRegister) -> ()
    return
  }
}
