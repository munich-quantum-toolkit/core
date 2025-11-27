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
// Parameterized gates RX/RY/RZ, PhaseShift and controlled variants
// Groups: Allocation & extraction / Uncontrolled / Controlled / Reinsertion
// ============================================================================
module {
  // CHECK-LABEL: func.func @testMQTOptToCatalystQuantumParameterizedGates
  func.func @testMQTOptToCatalystQuantumParameterizedGates() {
    // --- Allocation & extraction ---------------------------------------------------------------
    // CHECK: %[[THETA:.*]] = arith.constant 3.000000e-01 : f64
    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[C1:.*]] = arith.constant 1 : index
    // CHECK: %[[QREG:.*]] = quantum.alloc( 2) : !quantum.reg
    // CHECK: %[[IDX0:.*]] = arith.index_cast %[[C0]] : index to i64
    // CHECK: %[[Q0:.*]] = quantum.extract %[[QREG]][%[[IDX0]]] : !quantum.reg -> !quantum.bit
    // CHECK: %[[IDX1:.*]] = arith.index_cast %[[C1]] : index to i64
    // CHECK: %[[Q1:.*]] = quantum.extract %[[QREG]][%[[IDX1]]] : !quantum.reg -> !quantum.bit

    // --- Uncontrolled -------------------------------------------------------------------------
    // CHECK: %[[RX:.*]] = quantum.custom "RX"(%[[THETA]]) %[[Q0]] : !quantum.bit
    // CHECK: %[[RY:.*]] = quantum.custom "RY"(%[[THETA]]) %[[RX]] : !quantum.bit
    // CHECK: %[[RZ:.*]] = quantum.custom "RZ"(%[[THETA]]) %[[RY]] : !quantum.bit
    // CHECK: %[[PS:.*]] = quantum.custom "PhaseShift"(%[[THETA]]) %[[RZ]] : !quantum.bit
    // CHECK: quantum.gphase(%[[THETA]]) :

    // --- Controlled ----------------------------------------------------------------------------
    // CHECK: %[[TRUE:.*]] = arith.constant true
    // CHECK: %[[CRX_T:.*]], %[[CRX_C:.*]] = quantum.custom "CRX"(%[[THETA]]) %[[PS]] ctrls(%[[Q1]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CRY_T:.*]], %[[CRY_C:.*]] = quantum.custom "CRY"(%[[THETA]]) %[[CRX_T]] ctrls(%[[CRX_C]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CRZ_T:.*]], %[[CRZ_C:.*]] = quantum.custom "CRZ"(%[[THETA]]) %[[CRY_T]] ctrls(%[[CRY_C]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CPS_T:.*]], %[[CPS_C:.*]] = quantum.custom "ControlledPhaseShift"(%[[THETA]]) %[[CRZ_T]] ctrls(%[[CRZ_C]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit ctrls !quantum.bit
    // --- Reinsertion ---------------------------------------------------------------------------
    // CHECK: %[[C0_FINAL:.*]] = arith.index_cast %c0 : index to i64
    // CHECK: quantum.insert %[[QREG]][%[[C0_FINAL]]], %[[CPS_T]] : !quantum.reg, !quantum.bit
    // CHECK: %[[C1_FINAL:.*]] = arith.index_cast %c1 : index to i64
    // CHECK: quantum.insert %[[QREG]][%[[C1_FINAL]]], %[[CPS_C]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc %[[QREG]] : !quantum.reg

    // Prepare qubits
    %cst = arith.constant 3.000000e-01 : f64
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %r0_0 = memref.alloc() : memref<2x!mqtopt.Qubit>
    %q0_0 = memref.load %r0_0[%i0] : memref<2x!mqtopt.Qubit>
    %q1_0 = memref.load %r0_0[%i1] : memref<2x!mqtopt.Qubit>

    // Non-controlled rotations
    %q0_1 = mqtopt.rx(%cst) %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.ry(%cst) %q0_1 : !mqtopt.Qubit
    %q0_3 = mqtopt.rz(%cst) %q0_2 : !mqtopt.Qubit
    %q0_4 = mqtopt.p(%cst) %q0_3 : !mqtopt.Qubit
    mqtopt.gphase(%cst) : ()

    // Controlled rotations
    %q0_5, %q1_1 = mqtopt.rx(%cst) %q0_4 ctrl %q1_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_6, %q1_2 = mqtopt.ry(%cst) %q0_5 ctrl %q1_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_7, %q1_3 = mqtopt.rz(%cst) %q0_6 ctrl %q1_2 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_8, %q1_4 = mqtopt.p(%cst) %q0_7 ctrl %q1_3 : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // Release qubits
    memref.store %q0_8, %r0_0[%i0] : memref<2x!mqtopt.Qubit>
    memref.store %q1_4, %r0_0[%i1] : memref<2x!mqtopt.Qubit>
    memref.dealloc %r0_0 : memref<2x!mqtopt.Qubit>
    return
  }
}
