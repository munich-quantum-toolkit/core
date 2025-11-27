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
// Pauli family (X, Y, Z) and controlled variants
// Groups: Allocation & extraction / Uncontrolled / Controlled / Reinsertion
// ============================================================================
module {
  // CHECK-LABEL: func.func @testMQTOptToCatalystQuantumPauliGates
  func.func @testMQTOptToCatalystQuantumPauliGates() {
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

    // --- Uncontrolled Pauli gates --------------------------------------------------------------
    // CHECK: %[[X:.*]] = quantum.custom "PauliX"() %[[Q0]] : !quantum.bit
    // CHECK: %[[Y:.*]] = quantum.custom "PauliY"() %[[X]] : !quantum.bit
    // CHECK: %[[Z:.*]] = quantum.custom "PauliZ"() %[[Y]] : !quantum.bit
    // CHECK: %[[I:.*]] = quantum.custom "Identity"() %[[Z]] : !quantum.bit

    // CHECK: %[[TRUE:.*]] = arith.constant true
    // CHECK: %[[CNOT_T:.*]], %[[CNOT_C:.*]] = quantum.custom "CNOT"() %[[I]] ctrls(%[[Q1]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CY_T:.*]], %[[CY_C:.*]] = quantum.custom "CY"() %[[CNOT_T]] ctrls(%[[CNOT_C]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CZ_T:.*]], %[[CZ_C:.*]] = quantum.custom "CZ"() %[[CY_T]] ctrls(%[[CY_C]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[I_T:.*]], %[[I_C:.*]] = quantum.custom "Identity"() %[[CZ_T]] ctrls(%[[CZ_C]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[TOF_T:.*]], %[[TOF_C:.*]]:2 = quantum.custom "Toffoli"() %[[I_T]] ctrls(%[[I_C]], %[[Q2]]) ctrlvals(%[[TRUE]]{{.*}}, %[[TRUE]]{{.*}}) : !quantum.bit ctrls !quantum.bit, !quantum.bit

    // --- Reinsertion ----------------------------------------------------------------------------
    // CHECK: %[[C0_FINAL:.*]] = arith.index_cast %c0 : index to i64
    // CHECK: quantum.insert %[[QREG]][%[[C0_FINAL]]], %[[TOF_T]] : !quantum.reg, !quantum.bit
    // CHECK: %[[C1_FINAL:.*]] = arith.index_cast %c1 : index to i64
    // CHECK: quantum.insert %[[QREG]][%[[C1_FINAL]]], %[[TOF_C]]#0 : !quantum.reg, !quantum.bit
    // CHECK: %[[C2_FINAL:.*]] = arith.index_cast %c2 : index to i64
    // CHECK: quantum.insert %[[QREG]][%[[C2_FINAL]]], %[[TOF_C]]#1 : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc %[[QREG]] : !quantum.reg

    // Prepare qubits
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %r0_0 = memref.alloc() : memref<3x!mqtopt.Qubit>
    %q0_0 = memref.load %r0_0[%i0] : memref<3x!mqtopt.Qubit>
    %q1_0 = memref.load %r0_0[%i1] : memref<3x!mqtopt.Qubit>
    %q2_0 = memref.load %r0_0[%i2] : memref<3x!mqtopt.Qubit>

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
    memref.store %q0_9, %r0_0[%i0] : memref<3x!mqtopt.Qubit>
    memref.store %q1_5, %r0_0[%i1] : memref<3x!mqtopt.Qubit>
    memref.store %q2_1, %r0_0[%i2] : memref<3x!mqtopt.Qubit>
    memref.dealloc %r0_0 : memref<3x!mqtopt.Qubit>
    return
  }
}
