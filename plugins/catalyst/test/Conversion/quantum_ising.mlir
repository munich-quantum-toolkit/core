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
// Ising-type gates and controlled variants
// Groups: Constants / Allocation & extraction / Uncontrolled chain / Controlled chain / Reinsertion
// ============================================================================
module {
  // CHECK-LABEL: func.func @testCatalystQuantumToMQTOptIsing
  func.func @testCatalystQuantumToMQTOptIsing() {
    // --- Constants & Allocation & extraction ---------------------------------------------------
    // CHECK: %cst = arith.constant 3.000000e-01 : f64
    // CHECK: %[[QREG:.*]] = memref.alloc() : memref<3x!mqtopt.Qubit>
    // CHECK: %[[CAST:.*]] = memref.cast %[[QREG]] : memref<3x!mqtopt.Qubit> to memref<?x!mqtopt.Qubit>
    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[Q0:.*]] = memref.load %[[QREG]][%[[C0]]] : memref<3x!mqtopt.Qubit>
    // CHECK: %[[C1:.*]] = arith.constant 1 : index
    // CHECK: %[[Q1:.*]] = memref.load %[[QREG]][%[[C1]]] : memref<3x!mqtopt.Qubit>
    // CHECK: %[[C2:.*]] = arith.constant 2 : index
    // CHECK: %[[Q2:.*]] = memref.load %[[QREG]][%[[C2]]] : memref<3x!mqtopt.Qubit>

    // --- Uncontrolled ---------------------------------------------------------------------------
    // CHECK: %[[XY:.*]]:2 = mqtopt.xx_plus_yy(%cst, %cst static [] mask [false, false]) %[[Q0]], %[[Q1]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[XX:.*]]:2 = mqtopt.rxx(%cst static [] mask [false]) %[[XY]]#0, %[[XY]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[YY:.*]]:2 = mqtopt.ryy(%cst static [] mask [false]) %[[XX]]#0, %[[XX]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[ZZ:.*]]:2 = mqtopt.rzz(%cst static [] mask [false]) %[[YY]]#0, %[[YY]]#1 : !mqtopt.Qubit, !mqtopt.Qubit

    // --- Controlled -----------------------------------------------------------------------------
    // CHECK: %[[CXY_T:.*]]:2, %[[CXY_C:.*]] = mqtopt.xx_plus_yy(%cst, %cst static [] mask [false, false]) %[[ZZ]]#0, %[[ZZ]]#1 ctrl %[[Q2]] : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CXX_T:.*]]:2, %[[CXX_C:.*]] = mqtopt.rxx(%cst static [] mask [false]) %[[CXY_T]]#0, %[[CXY_T]]#1 ctrl %[[CXY_C]] : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CYY_T:.*]]:2, %[[CYY_C:.*]] = mqtopt.ryy(%cst static [] mask [false]) %[[CXX_T]]#0, %[[CXX_T]]#1 ctrl %[[CXX_C]] : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CZZ_T:.*]]:2, %[[CZZ_C:.*]] = mqtopt.rzz(%cst static [] mask [false]) %[[CYY_T]]#0, %[[CYY_T]]#1 ctrl %[[CYY_C]] : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit

    // --- Reinsertion ---------------------------------------------------------------------------
    // CHECK: %[[C0_FINAL:.*]] = arith.constant 0 : index
    // CHECK: memref.store %[[CZZ_T]]#0, %[[QREG]][%[[C0_FINAL]]] : memref<3x!mqtopt.Qubit>
    // CHECK: %[[C1_FINAL:.*]] = arith.constant 1 : index
    // CHECK: memref.store %[[CZZ_T]]#1, %[[QREG]][%[[C1_FINAL]]] : memref<3x!mqtopt.Qubit>
    // CHECK: %[[C2_FINAL:.*]] = arith.constant 2 : index
    // CHECK: memref.store %[[CZZ_C]], %[[QREG]][%[[C2_FINAL]]] : memref<3x!mqtopt.Qubit>
    // CHECK: memref.dealloc %[[CAST]] : memref<?x!mqtopt.Qubit>

    // Prepare qubits
    %angle = arith.constant 3.000000e-01 : f64
    %qreg = quantum.alloc( 3) : !quantum.reg
    %q0 = quantum.extract %qreg[ 0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %qreg[ 1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %qreg[ 2] : !quantum.reg -> !quantum.bit

    // Uncontrolled Ising gates
    %q0_xy, %q1_xy = quantum.custom "IsingXY"(%angle, %angle) %q0, %q1 : !quantum.bit, !quantum.bit
    %q0_xx, %q1_xx = quantum.custom "IsingXX"(%angle) %q0_xy, %q1_xy : !quantum.bit, !quantum.bit
    %q0_yy, %q1_yy = quantum.custom "IsingYY"(%angle) %q0_xx, %q1_xx : !quantum.bit, !quantum.bit
    %q0_zz, %q1_zz = quantum.custom "IsingZZ"(%angle) %q0_yy, %q1_yy : !quantum.bit, !quantum.bit

    // Controlled Ising gates
    %true = arith.constant true
    %q0_cxy, %q1_cxy, %q2_cxy = quantum.custom "IsingXY"(%angle, %angle) %q0_zz, %q1_zz ctrls(%q2) ctrlvals(%true) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    %q0_cxx, %q1_cxx, %q2_cxx = quantum.custom "IsingXX"(%angle) %q0_cxy, %q1_cxy ctrls(%q2_cxy) ctrlvals(%true) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    %q0_cyy, %q1_cyy, %q2_cyy = quantum.custom "IsingYY"(%angle) %q0_cxx, %q1_cxx ctrls(%q2_cxx) ctrlvals(%true) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    %q0_czz, %q1_czz, %q2_czz = quantum.custom "IsingZZ"(%angle) %q0_cyy, %q1_cyy ctrls(%q2_cyy) ctrlvals(%true) : !quantum.bit, !quantum.bit ctrls !quantum.bit


    // Release qubits
    %qreg1 = quantum.insert %qreg[ 0], %q0_czz : !quantum.reg, !quantum.bit
    %qreg2 = quantum.insert %qreg1[ 1], %q1_czz : !quantum.reg, !quantum.bit
    %qreg3 = quantum.insert %qreg2[ 2], %q2_czz : !quantum.reg, !quantum.bit
    quantum.dealloc %qreg3 : !quantum.reg
    return
  }
}
