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
// SWAP / ISWAP / ISWAP† and controlled variants
// Groups: Allocation & extraction / Uncontrolled / Controlled / Reinsertion
// ============================================================================
module {
  // CHECK-LABEL: func.func @testMQTOptToCatalystQuantumSwapAndISwap
  func.func @testMQTOptToCatalystQuantumSwapAndISwap() {
    // --- Allocation & extraction ---------------------------------------------------------------
    // CHECK: %[[QREG:.*]] = quantum.alloc( 3) : !quantum.reg
    // CHECK: %[[Q0_0:.*]] = quantum.extract %[[QREG]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: %[[Q1_0:.*]] = quantum.extract %[[QREG]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: %[[Q2_0:.*]] = quantum.extract %[[QREG]][ 2] : !quantum.reg -> !quantum.bit

    // --- Uncontrolled -------------------------------------------------------------------------
    // CHECK: %[[SW0:.*]]:2 = quantum.custom "SWAP"() %[[Q0_0]], %[[Q1_0]] : !quantum.bit, !quantum.bit
    // CHECK: %[[IS0:.*]]:2 = quantum.custom "ISWAP"() %[[SW0]]#0, %[[SW0]]#1 : !quantum.bit, !quantum.bit
    // CHECK: %[[ISD0:.*]]:2 = quantum.custom "ISWAP"() %[[IS0]]#0, %[[IS0]]#1 adj : !quantum.bit, !quantum.bit

    // --- Controlled ----------------------------------------------------------------------------
    // CHECK: %[[CSW_T:.*]]:2, %[[CSW_C:.*]] = quantum.custom "CSWAP"() %[[ISD0]]#0, %[[ISD0]]#1 ctrls(%[[Q2_0]]) ctrlvals(%true{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CISW_T:.*]]:2, %[[CISW_C:.*]] = quantum.custom "ISWAP"() %[[CSW_T]]#0, %[[CSW_T]]#1 ctrls(%[[CSW_C]]) ctrlvals(%true{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CISWD_T:.*]]:2, %[[CISWD_C:.*]] = quantum.custom "ISWAP"() %[[CISW_T]]#0, %[[CISW_T]]#1 adj ctrls(%[[CISW_C]]) ctrlvals(%true{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit


    // --- Reinsertion ---------------------------------------------------------------------------
    // CHECK: %[[R1:.*]] = quantum.insert %[[QREG]][ 0], %[[CISWD_T]]#0 : !quantum.reg, !quantum.bit
    // CHECK: %[[R2:.*]] = quantum.insert %[[R1]][ 1], %[[CISWD_T]]#1 : !quantum.reg, !quantum.bit
    // CHECK: %[[R3:.*]] = quantum.insert %[[R2]][ 2], %[[CISWD_C]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc %[[R3]] : !quantum.reg

    // Prepare qubits
    %r0_0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister
    %r0_1, %q0_0 = "mqtopt.extractQubit"(%r0_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    %r0_2, %q1_0 = "mqtopt.extractQubit"(%r0_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    %r0_3, %q2_0 = "mqtopt.extractQubit"(%r0_2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // SWAP / ISWAP / ISWAPdg
    %q0_1, %q1_1 = mqtopt.swap() %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
    %q0_2, %q1_2 = mqtopt.iswap() %q0_1, %q1_1 : !mqtopt.Qubit, !mqtopt.Qubit
    %q0_3, %q1_3 = mqtopt.iswapdg() %q0_2, %q1_2 : !mqtopt.Qubit, !mqtopt.Qubit

    // Controlled SWAP / ISWAP / ISWAPdg
    %q0_4, %q1_4, %q2_1 = mqtopt.swap() %q0_3, %q1_3 ctrl %q2_0 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_5, %q1_5, %q2_2 = mqtopt.iswap() %q0_4, %q1_4 ctrl %q2_1 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_6, %q1_6, %q2_3 = mqtopt.iswapdg() %q0_5, %q1_5 ctrl %q2_2 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit

    // Release qubits
    %r0_4 = "mqtopt.insertQubit"(%r0_3, %q0_6) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    %r0_5 = "mqtopt.insertQubit"(%r0_4, %q1_6) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    %r0_6 = "mqtopt.insertQubit"(%r0_5, %q2_3) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    "mqtopt.deallocQubitRegister"(%r0_6) : (!mqtopt.QubitRegister) -> ()
    return
  }
}