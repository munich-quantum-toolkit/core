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
// Ising-type gates and controlled variants
// Groups: Allocation & extraction / Uncontrolled / Controlled / Reinsertion
// ============================================================================
module {
  // CHECK-LABEL: func.func @testMQTOptToCatalystQuantumIsingGates
  func.func @testMQTOptToCatalystQuantumIsingGates() {
    // --- Allocation & extraction ---------------------------------------------------------------
    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[C1:.*]] = arith.constant 1 : index
    // CHECK: %[[C2:.*]] = arith.constant 2 : index
    // CHECK: %[[C3_I64:.*]] = arith.constant 3 : i64
    // CHECK: %[[QREG:.*]] = quantum.alloc(%[[C3_I64]]) : !quantum.reg
    // CHECK: %[[IDX0:.*]] = arith.index_cast %[[C0]] : index to i64
    // CHECK: %[[Q0:.*]] = quantum.extract %[[QREG]][%[[IDX0]]] : !quantum.reg -> !quantum.bit
    // CHECK: %[[IDX1:.*]] = arith.index_cast %[[C1]] : index to i64
    // CHECK: %[[Q1:.*]] = quantum.extract %[[QREG]][%[[IDX1]]] : !quantum.reg -> !quantum.bit
    // CHECK: %[[IDX2:.*]] = arith.index_cast %[[C2]] : index to i64
    // CHECK: %[[Q2:.*]] = quantum.extract %[[QREG]][%[[IDX2]]] : !quantum.reg -> !quantum.bit

    // --- Uncontrolled -------------------------------------------------------------------
    // CHECK: %[[XY_P:.*]]:2 = quantum.custom "IsingXY"(%cst, %cst) %[[Q0]], %[[Q1]] : !quantum.bit, !quantum.bit

    // CHECK: %[[XX_P:.*]]:2 = quantum.custom "IsingXX"(%cst) %[[XY_P]]#0, %[[XY_P]]#1 : !quantum.bit, !quantum.bit
    // CHECK: %[[YY_P:.*]]:2 = quantum.custom "IsingYY"(%cst) %[[XX_P]]#0, %[[XX_P]]#1 : !quantum.bit, !quantum.bit
    // CHECK: %[[ZZ_P1:.*]]:2 = quantum.custom "IsingZZ"(%cst) %[[YY_P]]#0, %[[YY_P]]#1 : !quantum.bit, !quantum.bit

    // CHECK: %[[H1:.*]] = quantum.custom "Hadamard"() %[[ZZ_P1]]#1 : !quantum.bit
    // CHECK: %[[ZZ_P2:.*]]:2 = quantum.custom "IsingZZ"(%cst) %[[ZZ_P1]]#0, %[[H1]] : !quantum.bit, !quantum.bit
    // CHECK: %[[H2:.*]] = quantum.custom "Hadamard"() %[[ZZ_P2]]#1 : !quantum.bit

    // --- Controlled ---------------------------------------------------------------------
    // CHECK: %[[XY_C:.*]]:2, %[[CTRL1:.*]] = quantum.custom "IsingXY"(%cst, %cst) %[[ZZ_P2]]#0, %[[H2]] ctrls(%[[Q2]]) ctrlvals(%true{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit

    // CHECK: %[[XX_C:.*]]:2, %[[CTRL2:.*]] = quantum.custom "IsingXX"(%cst) %[[XY_C]]#0, %[[XY_C]]#1 ctrls(%[[CTRL1]]) ctrlvals(%true{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[YY_C:.*]]:2, %[[CTRL3:.*]] = quantum.custom "IsingYY"(%cst) %[[XX_C]]#0, %[[XX_C]]#1 ctrls(%[[CTRL2]]) ctrlvals(%true{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[ZZ_C1:.*]]:2, %[[CTRL4:.*]] = quantum.custom "IsingZZ"(%cst) %[[YY_C]]#0, %[[YY_C]]#1 ctrls(%[[CTRL3]]) ctrlvals(%true{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit

    // CHECK: %[[H1:.*]], %[[CTRL5:.*]] = quantum.custom "Hadamard"() %[[ZZ_C1]]#1 ctrls(%[[CTRL4]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CZZ_P2:.*]]:2, %[[CTRL6:.*]] = quantum.custom "IsingZZ"(%cst) %[[ZZ_C1]]#0, %[[H1]] ctrls(%[[CTRL5]]) ctrlvals(%true{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[H2:.*]], %[[CTRL7:.*]] = quantum.custom "Hadamard"() %[[CZZ_P2]]#1 ctrls(%[[CTRL6]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit

    // --- Reinsertion ---------------------------------------------------------------------------
    // CHECK: quantum.insert %[[QREG]][{{.*}}], %[[CZZ_P2]]#0 : !quantum.reg, !quantum.bit
    // CHECK: quantum.insert %[[QREG]][{{.*}}], %[[H2]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.insert %[[QREG]][{{.*}}], %[[CTRL7]] : !quantum.reg, !quantum.bit
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
    %cst = arith.constant 3.000000e-01 : f64
    %q0_1, %q1_1 = mqtopt.xx_plus_yy(%cst, %cst) %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
    %q0_3, %q1_3 = mqtopt.rxx(%cst) %q0_1, %q1_1 : !mqtopt.Qubit, !mqtopt.Qubit
    %q0_4, %q1_4 = mqtopt.ryy(%cst) %q0_3, %q1_3 : !mqtopt.Qubit, !mqtopt.Qubit
    %q0_5, %q1_5 = mqtopt.rzz(%cst) %q0_4, %q1_4 : !mqtopt.Qubit, !mqtopt.Qubit
    %q0_6, %q1_6 = mqtopt.rzx(%cst) %q0_5, %q1_5 : !mqtopt.Qubit, !mqtopt.Qubit

    // Controlled
    %q0_7,  %q1_7,  %q2_1 = mqtopt.xx_plus_yy(%cst, %cst) %q0_6, %q1_6 ctrl %q2_0 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_9,  %q1_9,  %q2_3 = mqtopt.rxx(%cst) %q0_7, %q1_7 ctrl %q2_1 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_10, %q1_10, %q2_4 = mqtopt.ryy(%cst) %q0_9, %q1_9 ctrl %q2_3 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_11, %q1_11, %q2_5 = mqtopt.rzz(%cst) %q0_10, %q1_10 ctrl %q2_4 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_12, %q1_12, %q2_6 = mqtopt.rzx(%cst) %q0_11, %q1_11 ctrl %q2_5 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit

    // Release qubits
    memref.store %q0_12, %r0_0[%i0] : memref<3x!mqtopt.Qubit>
    memref.store %q1_12, %r0_0[%i1] : memref<3x!mqtopt.Qubit>
    memref.store %q2_6, %r0_0[%i2] : memref<3x!mqtopt.Qubit>
    memref.dealloc %r0_0 : memref<3x!mqtopt.Qubit>
    return
  }
}
