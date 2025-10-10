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
// RUN:   --catalyst-pipeline="builtin.module(catalystquantum-to-mqtopt)" \
// RUN:   %s | FileCheck %s


// ============================================================================
// Entangling gates (SWAP, ISWAP, ECR) and controlled variants
// Groups: Allocation & extraction / Uncontrolled / Controlled / Reinsertion
// ============================================================================
module {
  // CHECK-LABEL: func.func @testCatalystQuantumToMQTOptEntanglingGates
  func.func @testCatalystQuantumToMQTOptEntanglingGates() {
    // --- Allocation & extraction ---------------------------------------------------------------
    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<3x!mqtopt.Qubit>
    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[Q0:.*]] = memref.load %[[ALLOC]][%[[C0]]] : memref<3x!mqtopt.Qubit>
    // CHECK: %[[C1:.*]] = arith.constant 1 : index
    // CHECK: %[[Q1:.*]] = memref.load %[[ALLOC]][%[[C1]]] : memref<3x!mqtopt.Qubit>
    // CHECK: %[[C2:.*]] = arith.constant 2 : index
    // CHECK: %[[Q2:.*]] = memref.load %[[ALLOC]][%[[C2]]] : memref<3x!mqtopt.Qubit>

    // --- Uncontrolled entangling gates ---------------------------------------------------------
    // CHECK: %[[SW:.*]]:2 = mqtopt.swap(static [] mask []) %[[Q0]], %[[Q1]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[IS:.*]]:2 = mqtopt.iswap(static [] mask []) %[[SW]]#0, %[[SW]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[ISD:.*]]:2 = mqtopt.iswapdg(static [] mask []) %[[IS]]#0, %[[IS]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[ECR:.*]]:2 = mqtopt.ecr(static [] mask []) %[[Q0]], %[[Q1]] : !mqtopt.Qubit, !mqtopt.Qubit

    // --- Controlled entangling gates -----------------------------------------------------------
    // CHECK: %[[CSW_T:.*]]:2, %[[CSW_C:.*]] = mqtopt.swap(static [] mask []) %[[ECR]]#0, %[[ECR]]#1 ctrl %[[Q2]] : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CISW_T:.*]]:2, %[[CISW_C:.*]] = mqtopt.iswap(static [] mask []) %[[CSW_T]]#0, %[[CSW_T]]#1 ctrl %[[CSW_C]] : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CISWD_T:.*]]:2, %[[CISWD_C:.*]] = mqtopt.iswapdg(static [] mask []) %[[CISW_T]]#0, %[[CISW_T]]#1 ctrl %[[CISW_C]] : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CECR_T:.*]]:2, %[[CECR_C:.*]] = mqtopt.ecr(static [] mask []) %[[CISWD_T]]#0, %[[CISWD_T]]#1 ctrl %[[CISWD_C]] : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit

    // --- Reinsertion ---------------------------------------------------------------------------
    // CHECK: %[[C0_FINAL:.*]] = arith.constant 0 : index
    // CHECK: memref.store %[[CECR_T]]#0, %[[ALLOC]][%[[C0_FINAL]]] : memref<3x!mqtopt.Qubit>
    // CHECK: %[[C1_FINAL:.*]] = arith.constant 1 : index
    // CHECK: memref.store %[[CECR_T]]#1, %[[ALLOC]][%[[C1_FINAL]]] : memref<3x!mqtopt.Qubit>
    // CHECK: %[[C2_FINAL:.*]] = arith.constant 2 : index
    // CHECK: memref.store %[[CECR_C]], %[[ALLOC]][%[[C2_FINAL]]] : memref<3x!mqtopt.Qubit>


    // Prepare qubits
    %qreg = quantum.alloc( 3) : !quantum.reg
    %q0 = quantum.extract %qreg[ 0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %qreg[ 1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %qreg[ 2] : !quantum.reg -> !quantum.bit

    // Uncontrolled permutation gates
    %q0_sw, %q1_sw = quantum.custom "SWAP"() %q0, %q1 : !quantum.bit, !quantum.bit
    %q0_is, %q1_is = quantum.custom "ISWAP"() %q0_sw, %q1_sw : !quantum.bit, !quantum.bit
    %q0_isd, %q1_isd = quantum.custom "ISWAP"() %q0_is, %q1_is adj : !quantum.bit, !quantum.bit
    %q0_ecr, %q1_ecr = quantum.custom "ECR"() %q0, %q1 : !quantum.bit, !quantum.bit

    // Controlled permutation gates
    %true = arith.constant true
    %q0_csw, %q1_csw, %q2_csw = quantum.custom "SWAP"() %q0_ecr, %q1_ecr ctrls(%q2) ctrlvals(%true) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    %q0_cis, %q1_cis, %q2_cis = quantum.custom "ISWAP"() %q0_csw, %q1_csw ctrls(%q2_csw) ctrlvals(%true) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    %q0_cisd, %q1_cisd, %q2_cisd = quantum.custom "ISWAP"() %q0_cis, %q1_cis adj ctrls(%q2_cis) ctrlvals(%true) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    %q0_cecr, %q1_cecr, %q2_cecr = quantum.custom "ECR"() %q0_cisd, %q1_cisd ctrls(%q2_cisd) ctrlvals(%true) : !quantum.bit, !quantum.bit ctrls !quantum.bit

    // Release qubits
    %qreg1 = quantum.insert %qreg[ 0], %q0_cecr : !quantum.reg, !quantum.bit
    %qreg2 = quantum.insert %qreg1[ 1], %q1_cecr : !quantum.reg, !quantum.bit
    %qreg3 = quantum.insert %qreg2[ 2], %q2_cecr : !quantum.reg, !quantum.bit
    quantum.dealloc %qreg3 : !quantum.reg
    return
  }
}
