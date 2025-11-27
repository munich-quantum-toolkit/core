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
    // CHECK: %[[QREG:.*]] = quantum.alloc( 3) : !quantum.reg
    // CHECK: %[[IDX0:.*]] = arith.index_cast %[[C0]] : index to i64
    // CHECK: %[[Q0:.*]] = quantum.extract %[[QREG]][%[[IDX0]]] : !quantum.reg -> !quantum.bit
    // CHECK: %[[IDX1:.*]] = arith.index_cast %[[C1]] : index to i64
    // CHECK: %[[Q1:.*]] = quantum.extract %[[QREG]][%[[IDX1]]] : !quantum.reg -> !quantum.bit
    // CHECK: %[[IDX2:.*]] = arith.index_cast %[[C2]] : index to i64
    // CHECK: %[[Q2:.*]] = quantum.extract %[[QREG]][%[[IDX2]]] : !quantum.reg -> !quantum.bit

    // --- Uncontrolled -------------------------------------------------------------------
    // CHECK: %[[CST:.*]] = arith.constant {{.*}} : f64
    // CHECK: %[[RZ0:.*]] = quantum.custom "RZ"(%{{.*}}) %[[Q1]] : !quantum.bit
    // CHECK: %[[XY_P:.*]]:2 = quantum.custom "IsingXY"(%[[CST]]) %[[Q0]], %[[RZ0]] : !quantum.bit, !quantum.bit
    // CHECK: %[[RZ1:.*]] = quantum.custom "RZ"(%{{.*}}) %[[XY_P]]#1 : !quantum.bit

    // CHECK: %[[X1:.*]] = quantum.custom "PauliX"() %[[XY_P]]#0 : !quantum.bit
    // CHECK: %[[RZ2:.*]] = quantum.custom "RZ"(%{{.*}}) %[[RZ1]] : !quantum.bit
    // CHECK: %[[XY_M:.*]]:2 = quantum.custom "IsingXY"(%[[CST]]) %[[X1]], %[[RZ2]] : !quantum.bit, !quantum.bit
    // CHECK: %[[RZ3:.*]] = quantum.custom "RZ"(%{{.*}}) %[[XY_M]]#1 : !quantum.bit
    // CHECK: %[[X2:.*]] = quantum.custom "PauliX"() %[[XY_M]]#0 : !quantum.bit

    // CHECK: %[[XX_P:.*]]:2 = quantum.custom "IsingXX"(%[[CST]]) %[[X2]], %[[RZ3]] : !quantum.bit, !quantum.bit
    // CHECK: %[[YY_P:.*]]:2 = quantum.custom "IsingYY"(%[[CST]]) %[[XX_P]]#0, %[[XX_P]]#1 : !quantum.bit, !quantum.bit
    // CHECK: %[[ZZ_P1:.*]]:2 = quantum.custom "IsingZZ"(%[[CST]]) %[[YY_P]]#0, %[[YY_P]]#1 : !quantum.bit, !quantum.bit

    // CHECK: %[[H1U:.*]] = quantum.custom "Hadamard"() %[[ZZ_P1]]#1 : !quantum.bit
    // CHECK: %[[ZZ_P2:.*]]:2 = quantum.custom "IsingZZ"(%[[CST]]) %[[ZZ_P1]]#0, %[[H1U]] : !quantum.bit, !quantum.bit
    // CHECK: %[[H2U:.*]] = quantum.custom "Hadamard"() %[[ZZ_P2]]#1 : !quantum.bit

    // --- Controlled ---------------------------------------------------------------------
    // CHECK: %[[TRUE:.*]] = arith.constant true
    // CHECK: %[[RZ2:.*]], %[[CTRL1A:.*]] = quantum.custom "RZ"(%{{.*}}) %[[H2U]] ctrls(%[[Q2]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[XY_C:.*]]:2, %[[CTRL1B:.*]] = quantum.custom "IsingXY"(%[[CST]]) %[[ZZ_P2]]#0, %[[RZ2]] ctrls(%[[CTRL1A]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[RZ3:.*]], %[[CTRL1:.*]] = quantum.custom "RZ"(%{{.*}}) %[[XY_C]]#1 ctrls(%[[CTRL1B]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit ctrls !quantum.bit

    // CHECK: %[[X1C:.*]], %[[CTRL2A:.*]] = quantum.custom "PauliX"() %[[XY_C]]#0 ctrls(%[[CTRL1]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[RZ4:.*]], %[[CTRL2B:.*]] = quantum.custom "RZ"(%{{.*}}) %[[RZ3]] ctrls(%[[CTRL2A]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[XY_CM:.*]]:2, %[[CTRL2C:.*]] = quantum.custom "IsingXY"(%[[CST]]) %[[X1C]], %[[RZ4]] ctrls(%[[CTRL2B]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[RZ5:.*]], %[[CTRL2D:.*]] = quantum.custom "RZ"(%{{.*}}) %[[XY_CM]]#1 ctrls(%[[CTRL2C]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[X2C:.*]], %[[CTRL2:.*]] = quantum.custom "PauliX"() %[[XY_CM]]#0 ctrls(%[[CTRL2D]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit ctrls !quantum.bit

    // CHECK: %[[XX_C:.*]]:2, %[[CTRL3:.*]] = quantum.custom "IsingXX"(%[[CST]]) %[[X2C]], %[[RZ5]] ctrls(%[[CTRL2]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[YY_C:.*]]:2, %[[CTRL4:.*]] = quantum.custom "IsingYY"(%[[CST]]) %[[XX_C]]#0, %[[XX_C]]#1 ctrls(%[[CTRL3]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[ZZ_C1:.*]]:2, %[[CTRL5:.*]] = quantum.custom "IsingZZ"(%[[CST]]) %[[YY_C]]#0, %[[YY_C]]#1 ctrls(%[[CTRL4]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit

    // CHECK: %[[H1C:.*]] = quantum.custom "Hadamard"() %[[ZZ_C1]]#1 : !quantum.bit
    // CHECK: %[[CZZ_P2:.*]]:2, %[[CTRL7:.*]] = quantum.custom "IsingZZ"(%[[CST]]) %[[ZZ_C1]]#0, %[[H1C]] ctrls(%[[CTRL5]]) ctrlvals(%[[TRUE]]{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[H2C:.*]] = quantum.custom "Hadamard"() %[[CZZ_P2]]#1 : !quantum.bit

    // --- Reinsertion ---------------------------------------------------------------------------
    // CHECK: %[[C0_FINAL:.*]] = arith.index_cast %c0 : index to i64
    // CHECK: quantum.insert %[[QREG]][%[[C0_FINAL]]], %[[CZZ_P2]]#0 : !quantum.reg, !quantum.bit
    // CHECK: %[[C1_FINAL:.*]] = arith.index_cast %c1 : index to i64
    // CHECK: quantum.insert %[[QREG]][%[[C1_FINAL]]], %[[H2C]] : !quantum.reg, !quantum.bit
    // CHECK: %[[C2_FINAL:.*]] = arith.index_cast %c2 : index to i64
    // CHECK: quantum.insert %[[QREG]][%[[C2_FINAL]]], %[[CTRL7]] : !quantum.reg, !quantum.bit
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
    %q0_2, %q1_2 = mqtopt.xx_minus_yy(%cst, %cst) %q0_1, %q1_1 : !mqtopt.Qubit, !mqtopt.Qubit
    %q0_3, %q1_3 = mqtopt.rxx(%cst) %q0_2, %q1_2 : !mqtopt.Qubit, !mqtopt.Qubit
    %q0_4, %q1_4 = mqtopt.ryy(%cst) %q0_3, %q1_3 : !mqtopt.Qubit, !mqtopt.Qubit
    %q0_5, %q1_5 = mqtopt.rzz(%cst) %q0_4, %q1_4 : !mqtopt.Qubit, !mqtopt.Qubit
    %q0_6, %q1_6 = mqtopt.rzx(%cst) %q0_5, %q1_5 : !mqtopt.Qubit, !mqtopt.Qubit

    // Controlled
    %q0_7,  %q1_7,  %q2_1 = mqtopt.xx_plus_yy(%cst, %cst) %q0_6, %q1_6 ctrl %q2_0 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_8,  %q1_8,  %q2_2 = mqtopt.xx_minus_yy(%cst, %cst) %q0_7, %q1_7 ctrl %q2_1 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_9,  %q1_9,  %q2_3 = mqtopt.rxx(%cst) %q0_8, %q1_8 ctrl %q2_2 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
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
