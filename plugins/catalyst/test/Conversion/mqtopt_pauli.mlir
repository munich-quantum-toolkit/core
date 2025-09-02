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
// Pauli family (X, Y, Z) and controlled variants
// Groups: Allocation & extraction / Uncontrolled / Controlled / Reinsertion
// ============================================================================
module {
  // CHECK-LABEL: func.func @testMQTOptToCatalystQuantumPauliGates
  func.func @testMQTOptToCatalystQuantumPauliGates() {
    // --- Allocation & extraction ---------------------------------------------------------------
    // CHECK: %[[QREG:.*]] = quantum.alloc( 3) : !quantum.reg
    // CHECK: %[[Q0_0:.*]] = quantum.extract %[[QREG]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: %[[Q1_0:.*]] = quantum.extract %[[QREG]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: %[[Q2_0:.*]] = quantum.extract %[[QREG]][ 2] : !quantum.reg -> !quantum.bit

    // --- Uncontrolled -------------------------------------------------------------------------
    // CHECK: %[[X:.*]] = quantum.custom "PauliX"() %[[Q0_0]] : !quantum.bit
    // CHECK: %[[Y:.*]] = quantum.custom "PauliY"() %[[X]] : !quantum.bit
    // CHECK: %[[Z:.*]] = quantum.custom "PauliZ"() %[[Y]] : !quantum.bit
    // CHECK: %[[I:.*]] = quantum.custom "Identity"() %[[Z]] : !quantum.bit

    // --- Controlled ----------------------------------------------------------------------------
    // CHECK: %[[CNOT_T:.*]], %[[CNOT_C:.*]] = quantum.custom "CNOT"() %[[I]] ctrls(%[[Q1_0]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CY_T:.*]], %[[CY_C:.*]] = quantum.custom "CY"() %[[CNOT_T]] ctrls(%[[CNOT_C]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CZ_T:.*]], %[[CZ_C:.*]] = quantum.custom "CZ"() %[[CY_T]] ctrls(%[[CY_C]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[I_T:.*]], %[[I_C:.*]] = quantum.custom "Identity"() %[[CZ_T]] ctrls(%[[CZ_C]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[TOF_T:.*]], %[[TOF_C:.*]]:2 = quantum.custom "Toffoli"() %[[I_T]] ctrls(%[[I_C]], %[[Q2_0]]) ctrlvals(%true{{.*}}, %true{{.*}}) : !quantum.bit ctrls !quantum.bit, !quantum.bit

    // --- Reinsertion ---------------------------------------------------------------------------
    // CHECK: %[[R1:.*]] = quantum.insert %[[QREG]][ 2], %[[TOF_T]] : !quantum.reg, !quantum.bit
    // CHECK: %[[R2:.*]] = quantum.insert %[[R1]][ 1], %[[TOF_C]]#0 : !quantum.reg, !quantum.bit
    // CHECK: %[[R3:.*]] = quantum.insert %[[R2]][ 0], %[[TOF_C]]#1 : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc %[[R3]] : !quantum.reg

    // Prepare qubits
    %r0_0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister
    %r0_1, %q0_0 = "mqtopt.extractQubit"(%r0_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    %r0_2, %q1_0 = "mqtopt.extractQubit"(%r0_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    %r0_3, %q2_0 = "mqtopt.extractQubit"(%r0_2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // Non-controlled Pauli gates
    %q0_1 = mqtopt.x() %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.y() %q0_1 : !mqtopt.Qubit
    %q0_3 = mqtopt.z() %q0_2 : !mqtopt.Qubit
    %q0_4 = mqtopt.i() %q0_3 : !mqtopt.Qubit

    // Controlled Pauli gates
    %q0_5, %q1_1 = mqtopt.x() %q0_4 ctrl %q1_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_6, %q1_2 = mqtopt.y() %q0_5 ctrl %q1_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_7, %q1_3 = mqtopt.z() %q0_6 ctrl %q1_2 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_8, %q1_4 = mqtopt.i() %q0_7 ctrl %q1_3 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_9, %q1_5, %q2_1 = mqtopt.x() %q0_8 ctrl %q1_4, %q2_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit

    // Release qubits
    %r0_4 = "mqtopt.insertQubit"(%r0_3, %q0_9) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    %r0_5 = "mqtopt.insertQubit"(%r0_4, %q1_5) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    %r0_6 = "mqtopt.insertQubit"(%r0_5, %q2_1) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    "mqtopt.deallocQubitRegister"(%r0_6) : (!mqtopt.QubitRegister) -> ()
    return
  }
}
