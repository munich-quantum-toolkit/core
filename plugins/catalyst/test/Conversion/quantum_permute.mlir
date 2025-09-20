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
// Permutation gates and controlled variants
// Groups: Allocation & extraction / Uncontrolled / Controlled / Reinsertion
// ============================================================================
module {
  // CHECK-LABEL: func.func @testCatalystQuantumToMQTOptSwapAndISwap
  func.func @testCatalystQuantumToMQTOptSwapAndISwap() {
    // --- Allocation & extraction ---------------------------------------------------------------
    // CHECK: %[[QREG:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister
    // CHECK: %[[QR1:.*]], %[[Q0:.*]] = "mqtopt.extractQubit"(%[[QREG]]) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[QR2:.*]], %[[Q1:.*]] = "mqtopt.extractQubit"(%[[QR1]]) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[QR3:.*]], %[[Q2:.*]] = "mqtopt.extractQubit"(%[[QR2]]) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // --- Uncontrolled -------------------------------------------------------------------------
    // CHECK: %[[SW:.*]]:2 = mqtopt.swap( static [] mask []) %[[Q0]], %[[Q1]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[IS:.*]]:2 = mqtopt.iswap( static [] mask []) %[[SW]]#0, %[[SW]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[ISD:.*]]:2 = mqtopt.iswap( static [] mask []) %[[IS]]#0, %[[IS]]#1 : !mqtopt.Qubit, !mqtopt.Qubit

    // --- Controlled ----------------------------------------------------------------------------
    // CHECK: %[[CSW_T:.*]]:2, %[[CSW_C:.*]] = mqtopt.swap( static [] mask []) %[[ISD]]#0, %[[ISD]]#1 ctrl %[[Q2]] : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CISW_T:.*]]:2, %[[CISW_C:.*]] = mqtopt.iswap( static [] mask []) %[[CSW_T]]#0, %[[CSW_T]]#1 ctrl %[[CSW_C]] : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CISWD_T:.*]]:2, %[[CISWD_C:.*]] = mqtopt.iswap( static [] mask []) %[[CISW_T]]#0, %[[CISW_T]]#1 ctrl %[[CISW_C]] : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit

    // --- Reinsertion ---------------------------------------------------------------------------
    // CHECK: %[[R1:.*]] = "mqtopt.insertQubit"(%[[QR3]], %[[CISWD_T]]#0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %[[R2:.*]] = "mqtopt.insertQubit"(%[[R1]], %[[CISWD_T]]#1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %[[R3:.*]] = "mqtopt.insertQubit"(%[[R2]], %[[CISWD_C]]) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: "mqtopt.deallocQubitRegister"(%[[R3]]) : (!mqtopt.QubitRegister) -> ()

    // Prepare qubits
    %qreg = quantum.alloc( 3) : !quantum.reg
    %q0 = quantum.extract %qreg[ 0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %qreg[ 1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %qreg[ 2] : !quantum.reg -> !quantum.bit

    // Uncontrolled permutation gates
    %q0_sw, %q1_sw = quantum.custom "SWAP"() %q0, %q1 : !quantum.bit, !quantum.bit
    %q0_is, %q1_is = quantum.custom "ISWAP"() %q0_sw, %q1_sw : !quantum.bit, !quantum.bit
    %q0_isd, %q1_isd = quantum.custom "ISWAP"() %q0_is, %q1_is adj : !quantum.bit, !quantum.bit

    // Controlled permutation gates
    %true = arith.constant true
    %q0_csw, %q1_csw, %q2_csw = quantum.custom "SWAP"() %q0_isd, %q1_isd ctrls(%q2) ctrlvals(%true) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    %q0_cis, %q1_cis, %q2_cis = quantum.custom "ISWAP"() %q0_csw, %q1_csw ctrls(%q2_csw) ctrlvals(%true) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    %q0_cisd, %q1_cisd, %q2_cisd = quantum.custom "ISWAP"() %q0_cis, %q1_cis adj ctrls(%q2_cis) ctrlvals(%true) : !quantum.bit, !quantum.bit ctrls !quantum.bit

    // Release qubits
    %qreg1 = quantum.insert %qreg[ 0], %q0_cisd : !quantum.reg, !quantum.bit
    %qreg2 = quantum.insert %qreg1[ 1], %q1_cisd : !quantum.reg, !quantum.bit
    %qreg3 = quantum.insert %qreg2[ 2], %q2_cisd : !quantum.reg, !quantum.bit
    quantum.dealloc %qreg3 : !quantum.reg
    return
  }
}
