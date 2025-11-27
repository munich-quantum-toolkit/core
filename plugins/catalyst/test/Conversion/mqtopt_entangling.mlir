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
// RUN:   --catalyst-pipeline="builtin.module(mqtopt-to-catalystquantum)" \
// RUN:   %s | FileCheck %s


// ============================================================================
// Entangling gates (SWAP, ISWAP, ECR) and controlled variants
// Groups: Allocation & extraction / Uncontrolled / Controlled / Reinsertion
// ============================================================================
module {
  // CHECK-LABEL: func.func @testMQTOptToCatalystQuantumEntanglingGates
  func.func @testMQTOptToCatalystQuantumEntanglingGates() {
    // --- Allocation & extraction ---------------------------------------------------------------
    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[C1:.*]] = arith.constant 1 : index
    // CHECK: %[[C2:.*]] = arith.constant 2 : index
    // CHECK: %[[QREG:.*]] = quantum.alloc( 3) : !quantum.reg
    // CHECK: %[[IDX0:.*]] = arith.index_cast %[[C0]] : index to i64
    // CHECK: %[[Q0:.*]] = quantum.extract %[[QREG]][%[[IDX0]]] : !quantum.reg -> !quantum.bit
    // CHECK: %[[IDX1:.*]] = arith.index_cast %[[C1]] : index to i64
    // CHECK: %[[Q1:.*]] = quantum.extract %[[QREG]][%[[IDX1]]] : !quantum.reg -> !quantum.bit
    // CHECK: %[[IDX2:.*]] = arith.index_cast %[[C2]] : index to i64
    // CHECK: %[[Q2:.*]] = quantum.extract %[[QREG]][%[[IDX2]]] : !quantum.reg -> !quantum.bit

    // --- Uncontrolled -------------------------------------------------------------------------
    // CHECK: %[[SW0:.*]]:2 = quantum.custom "SWAP"() %[[Q0]], %[[Q1]] : !quantum.bit, !quantum.bit
    // CHECK: %[[IS0:.*]]:2 = quantum.custom "ISWAP"() %[[SW0]]#0, %[[SW0]]#1 : !quantum.bit, !quantum.bit
    // CHECK: %[[ISD0:.*]]:2 = quantum.custom "ISWAP"() %[[IS0]]#0, %[[IS0]]#1 adj : !quantum.bit, !quantum.bit
    // CHECK: %[[ECR0:.*]]:2 = quantum.custom "ECR"() %[[ISD0]]#0, %[[ISD0]]#1 : !quantum.bit, !quantum.bit
    // CHECK: %[[DCX0:.*]]:2 = quantum.custom "CNOT"() %[[ECR0]]#0, %[[ECR0]]#1 : !quantum.bit, !quantum.bit
    // CHECK: %[[DCX1:.*]]:2 = quantum.custom "CNOT"() %[[DCX0]]#1, %[[DCX0]]#0 : !quantum.bit, !quantum.bit

    // --- Controlled ----------------------------------------------------------------------------
    // CHECK: %[[TRUE:.*]] = arith.constant true
    // CHECK: %[[CSW_T:.*]]:2, %[[CSW_C:.*]] = quantum.custom "CSWAP"() %[[DCX1]]#0, %[[DCX1]]#1 ctrls(%[[Q2]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CISW_T:.*]]:2, %[[CISW_C:.*]] = quantum.custom "ISWAP"() %[[CSW_T]]#0, %[[CSW_T]]#1 ctrls(%[[CSW_C]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CISWD_T:.*]]:2, %[[CISWD_C:.*]] = quantum.custom "ISWAP"() %[[CISW_T]]#0, %[[CISW_T]]#1 adj ctrls(%[[CISW_C]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CECR_T:.*]]:2, %[[CECR_C:.*]] = quantum.custom "ECR"() %[[CISWD_T]]#0, %[[CISWD_T]]#1 ctrls(%[[CISWD_C]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CDCX1_T:.*]]:2, %[[CDCX1_C:.*]] = quantum.custom "CNOT"() %[[CECR_T]]#0, %[[CECR_T]]#1 ctrls(%[[CECR_C]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CDCX2_T:.*]]:2, %[[CDCX2_C:.*]] = quantum.custom "CNOT"() %[[CDCX1_T]]#1, %[[CDCX1_T]]#0 ctrls(%[[CDCX1_C]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit

    // --- Reinsertion ---------------------------------------------------------------------------
    // CHECK: %[[C0_FINAL:.*]] = arith.index_cast %c0 : index to i64
    // CHECK: quantum.insert %[[QREG]][%[[C0_FINAL]]], %[[CDCX2_T]]#0 : !quantum.reg, !quantum.bit
    // CHECK: %[[C1_FINAL:.*]] = arith.index_cast %c1 : index to i64
    // CHECK: quantum.insert %[[QREG]][%[[C1_FINAL]]], %[[CDCX2_T]]#1 : !quantum.reg, !quantum.bit
    // CHECK: %[[C2_FINAL:.*]] = arith.index_cast %c2 : index to i64
    // CHECK: quantum.insert %[[QREG]][%[[C2_FINAL]]], %[[CDCX2_C]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc %[[QREG]] : !quantum.reg

    // Prepare qubits
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %r0_0 = memref.alloc() : memref<3x!mqtopt.Qubit>
    %q0_0 = memref.load %r0_0[%i0] : memref<3x!mqtopt.Qubit>
    %q1_0 = memref.load %r0_0[%i1] : memref<3x!mqtopt.Qubit>
    %q2_0 = memref.load %r0_0[%i2] : memref<3x!mqtopt.Qubit>

    // Uncontrolled
    %q0_1, %q1_1 = mqtopt.swap() %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
    %q0_2, %q1_2 = mqtopt.iswap() %q0_1, %q1_1 : !mqtopt.Qubit, !mqtopt.Qubit
    %q0_3, %q1_3 = mqtopt.iswapdg() %q0_2, %q1_2 : !mqtopt.Qubit, !mqtopt.Qubit
    %q0_4, %q1_4 = mqtopt.ecr() %q0_3, %q1_3 : !mqtopt.Qubit, !mqtopt.Qubit
    %q0_5, %q1_5 = mqtopt.dcx() %q0_4, %q1_4 : !mqtopt.Qubit, !mqtopt.Qubit

    // Controlled
    %q0_6, %q1_6, %q2_1 = mqtopt.swap() %q0_5, %q1_5 ctrl %q2_0 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_7, %q1_7, %q2_2 = mqtopt.iswap() %q0_6, %q1_6 ctrl %q2_1 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_8, %q1_8, %q2_3 = mqtopt.iswapdg() %q0_7, %q1_7 ctrl %q2_2 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_9, %q1_9, %q2_4 = mqtopt.ecr() %q0_8, %q1_8 ctrl %q2_3 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_10, %q1_10, %q2_5 = mqtopt.dcx() %q0_9, %q1_9 ctrl %q2_4 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit

    // Release qubits
    memref.store %q0_10, %r0_0[%i0] : memref<3x!mqtopt.Qubit>
    memref.store %q1_10, %r0_0[%i1] : memref<3x!mqtopt.Qubit>
    memref.store %q2_5, %r0_0[%i2] : memref<3x!mqtopt.Qubit>
    memref.dealloc %r0_0 : memref<3x!mqtopt.Qubit>
    return
  }
}
