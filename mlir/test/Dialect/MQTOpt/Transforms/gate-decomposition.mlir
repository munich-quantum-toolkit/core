// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --gate-decomposition | FileCheck %s

// -----
// This test checks if a series equal to the identity will be cancelled out.

module {
  // CHECK-LABEL: func.func @testIdentitySeries
  func.func @testIdentitySeries() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit

    // CHECK-NOT: mqtopt.[[ANY:.*]]

    // CHECK: mqtopt.deallocQubit %[[Q0_0]]
    // CHECK: mqtopt.deallocQubit %[[Q1_0]]

    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    %q0_1, %q1_1 = mqtopt.x() %q0_0 ctrl %q1_0: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_2, %q1_2 = mqtopt.x() %q0_1 ctrl %q1_1: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_3, %q1_3 = mqtopt.x() %q0_2 ctrl %q1_2: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_4, %q1_4 = mqtopt.x() %q0_3 ctrl %q1_3: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_5, %q1_5 = mqtopt.x() %q0_4 ctrl %q1_4: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_6, %q1_6 = mqtopt.x() %q0_5 ctrl %q1_5: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_7, %q1_7 = mqtopt.x() %q0_6 ctrl %q1_6: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_8, %q1_8 = mqtopt.x() %q0_7 ctrl %q1_7: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_9, %q1_9 = mqtopt.x() %q0_8 ctrl %q1_8: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_10, %q1_10 = mqtopt.x() %q0_9 ctrl %q1_9: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_11, %q1_11 = mqtopt.x() %q0_10 ctrl %q1_10: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_12, %q1_12 = mqtopt.x() %q0_11 ctrl %q1_11: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_13, %q1_13 = mqtopt.x() %q0_12 ctrl %q1_12: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_14, %q1_14 = mqtopt.x() %q0_13 ctrl %q1_13: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_15, %q1_15 = mqtopt.x() %q0_14 ctrl %q1_14: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_16, %q1_16 = mqtopt.x() %q0_15 ctrl %q1_15: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_17, %q1_17 = mqtopt.x() %q0_16 ctrl %q1_16: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_18, %q1_18 = mqtopt.x() %q0_17 ctrl %q1_17: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_19, %q1_19 = mqtopt.x() %q0_18 ctrl %q1_18: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_20, %q1_20 = mqtopt.x() %q0_19 ctrl %q1_19: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_21, %q1_21 = mqtopt.x() %q0_20 ctrl %q1_20: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_22, %q1_22 = mqtopt.x() %q0_21 ctrl %q1_21: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_23, %q1_23 = mqtopt.x() %q0_22 ctrl %q1_22: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_24, %q1_24 = mqtopt.x() %q0_23 ctrl %q1_23: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_25, %q1_25 = mqtopt.x() %q0_24 ctrl %q1_24: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_27, %q1_27 = mqtopt.x() %q0_25 ctrl %q1_25: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_28, %q1_28 = mqtopt.x() %q0_27 ctrl %q1_27: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_29, %q1_29 = mqtopt.x() %q0_28 ctrl %q1_28: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_30, %q1_30 = mqtopt.x() %q0_29 ctrl %q1_29: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_31, %q1_31 = mqtopt.x() %q0_30 ctrl %q1_30: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_32, %q1_32 = mqtopt.x() %q0_31 ctrl %q1_31: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_33, %q1_33 = mqtopt.x() %q0_32 ctrl %q1_32: !mqtopt.Qubit ctrl !mqtopt.Qubit

    mqtopt.deallocQubit %q0_33
    mqtopt.deallocQubit %q1_33

    return
  }
}

// -----
// This test checks if an odd number of CNOT gates will be reduced to a single CNOT.

module {
  // CHECK-LABEL: func.func @testOddNegationSeries
  func.func @testOddNegationSeries() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]], %[[Q1_1:.*]] = mqtopt.x() %[[Q0_0]] ctrl %[[Q1_0]]

    // ensure no other operations are inserted
    // CHECK-NOT: mqtopt.[[ANY:.*]](%[[ANY:.*]])

    // CHECK: mqtopt.deallocQubit %[[Q0_1]]
    // CHECK: mqtopt.deallocQubit %[[Q1_1]]

    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    %q0_1, %q1_1 = mqtopt.x() %q0_0 ctrl %q1_0: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_2, %q1_2 = mqtopt.x() %q0_1 ctrl %q1_1: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_3, %q1_3 = mqtopt.x() %q0_2 ctrl %q1_2: !mqtopt.Qubit ctrl !mqtopt.Qubit

    mqtopt.deallocQubit %q0_3
    mqtopt.deallocQubit %q1_3

    return
  }
}

// -----
// This test checks if an odd number of CNOT gates with ctrl/target swapped will be reduced to a single CNOT.

module {
  // CHECK-LABEL: func.func @testCNotOtherDirection
  func.func @testCNotOtherDirection() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit

    // CHECK-NOT: mqtopt.i(%[[ANY:.*]])
    // CHECK: %[[Q1_1:.*]], %[[Q0_1:.*]] = mqtopt.x() %[[Q1_0]] ctrl %[[Q0_0]]

    // CHECK: mqtopt.deallocQubit %[[Q0_1]]
    // CHECK: mqtopt.deallocQubit %[[Q1_1]]

    // ensure no other operations are inserted
    // CHECK-NOT: mqtopt.[[ANY:.*]]

    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    %q1_1, %q0_1 = mqtopt.i() %q1_0 ctrl %q0_0: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q1_2, %q0_2 = mqtopt.x() %q1_1 ctrl %q0_1: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q1_3, %q0_3 = mqtopt.i() %q1_2 ctrl %q0_2: !mqtopt.Qubit ctrl !mqtopt.Qubit

    mqtopt.deallocQubit %q0_3
    mqtopt.deallocQubit %q1_3

    return
  }
}

// -----
// This test checks if a two-qubit series containing a single-qubit gate is decomposed correctly.

module {
  // CHECK-LABEL: func.func @testSeriesOneQubitOpInbetween
  func.func @testSeriesOneQubitOpInbetween() {
    // CHECK-DAG: %[[C0:.*]] = arith.constant 3.14159
    // CHECK-DAG: %[[C1:.*]] = arith.constant 1.5707

    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit

    // CHECK: mqtopt.gphase(%[[C1]])
    // CHECK: %[[Q0_1:.*]] = mqtopt.rx(%[[C0]]) %[[Q0_0]]

    // ensure no other operations are inserted
    // CHECK-NOT: mqtopt.[[ANY:.*]](%[[ANY:.*]])

    // CHECK: mqtopt.deallocQubit %[[Q0_1]]
    // CHECK: mqtopt.deallocQubit %[[Q1_0]]

    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    %q0_1, %q1_1 = mqtopt.x() %q0_0 ctrl %q1_0: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_2 = mqtopt.x() %q0_1: !mqtopt.Qubit
    %q0_3, %q1_2 = mqtopt.x() %q0_2 ctrl %q1_1: !mqtopt.Qubit ctrl !mqtopt.Qubit

    mqtopt.deallocQubit %q0_3
    mqtopt.deallocQubit %q1_2

    return
  }
}

// -----
// This test checks if a two-qubit series starting on a single-qubit gate is decomposed correctly.

module {
  // CHECK-LABEL: func.func @testSeriesStartingOneQubitOp
  func.func @testSeriesStartingOneQubitOp() {
    // CHECK-DAG: %[[C0:.*]] = arith.constant 3.14159
    // CHECK-DAG: %[[C1:.*]] = arith.constant 1.5707

    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit

    // CHECK: mqtopt.gphase(%[[C1]])
    // CHECK: %[[Q0_1:.*]] = mqtopt.rx(%[[C0]]) %[[Q0_0]]

    // ensure no other operations are inserted
    // CHECK-NOT: mqtopt.[[ANY:.*]](%[[ANY:.*]])

    // CHECK: mqtopt.deallocQubit %[[Q0_1]]
    // CHECK: mqtopt.deallocQubit %[[Q1_0]]

    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    %q0_1 = mqtopt.x() %q0_0: !mqtopt.Qubit
    %q0_2, %q1_1 = mqtopt.x() %q0_1 ctrl %q1_0: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_3, %q1_2 = mqtopt.x() %q0_2 ctrl %q1_1: !mqtopt.Qubit ctrl !mqtopt.Qubit

    mqtopt.deallocQubit %q0_3
    mqtopt.deallocQubit %q1_2

    return
  }
}

// -----
// This test checks if a two-qubit series ending on a single-qubit gate is decomposed correctly.

module {
  // CHECK-LABEL: func.func @testSeriesEndingOneQubitOp
  func.func @testSeriesEndingOneQubitOp() {
    // CHECK-DAG: %[[C0:.*]] = arith.constant 3.14159
    // CHECK-DAG: %[[C1:.*]] = arith.constant 1.5707

    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit

    // CHECK: mqtopt.gphase(%[[C1]])
    // CHECK: %[[Q0_1:.*]] = mqtopt.rx(%[[C0]]) %[[Q0_0]]

    // ensure no other operations are inserted
    // CHECK-NOT: mqtopt.[[ANY:.*]](%[[ANY:.*]])

    // CHECK: mqtopt.deallocQubit %[[Q0_1]]
    // CHECK: mqtopt.deallocQubit %[[Q1_0]]

    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    %q0_1, %q1_1 = mqtopt.x() %q0_0 ctrl %q1_0: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_2, %q1_2 = mqtopt.x() %q0_1 ctrl %q1_1: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_3 = mqtopt.x() %q0_2: !mqtopt.Qubit

    mqtopt.deallocQubit %q0_3
    mqtopt.deallocQubit %q1_2

    return
  }
}

// -----
// This test checks if an interrupted series is ignored correctly.

module {
  // CHECK-LABEL: func.func @testInterruptedSeries
  func.func @testInterruptedSeries() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q2_0:.*]] = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]], %[[Q1_1:.*]] = mqtopt.x() %[[Q0_0]] ctrl %[[Q1_0]]
    // CHECK: %[[Q1_2:.*]], %[[Q0_2:.*]] = mqtopt.x() %[[Q1_1]] ctrl %[[Q0_1]]
    // CHECK: %[[Q0_3:.*]], %[[Q2_1:.*]] = mqtopt.x() %[[Q0_2]] ctrl %[[Q2_0]]
    // CHECK: %[[Q1_3:.*]], %[[Q0_4:.*]] = mqtopt.x() %[[Q1_2]] ctrl %[[Q0_3]]
    // CHECK: %[[Q0_5:.*]], %[[Q1_4:.*]] = mqtopt.x() %[[Q0_4]] ctrl %[[Q1_3]]

    // ensure no other operations are inserted
    // CHECK-NOT: mqtopt.[[ANY:.*]](%[[ANY:.*]])

    // CHECK: mqtopt.deallocQubit %[[Q0_5]]
    // CHECK: mqtopt.deallocQubit %[[Q1_4]]
    // CHECK: mqtopt.deallocQubit %[[Q2_1]]

    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit
    %q2_0 = mqtopt.allocQubit

    %q0_1, %q1_1 = mqtopt.x() %q0_0 ctrl %q1_0: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q1_2, %q0_2 = mqtopt.x() %q1_1 ctrl %q0_1: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_3, %q2_1 = mqtopt.x() %q0_2 ctrl %q2_0: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q1_3, %q0_4 = mqtopt.x() %q1_2 ctrl %q0_3: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_5, %q1_4 = mqtopt.x() %q0_4 ctrl %q1_3: !mqtopt.Qubit ctrl !mqtopt.Qubit

    mqtopt.deallocQubit %q0_5
    mqtopt.deallocQubit %q1_4
    mqtopt.deallocQubit %q2_1

    return
  }
}

// -----
// This test checks if a complex series containing two basis gates is decomposed correctly.

module {
  // CHECK-LABEL: func.func @testTwoBasisGateDecomposition
  func.func @testTwoBasisGateDecomposition() {
    // CHECK-DAG: %[[C0:.*]] = arith.constant -1.5707963267
    // CHECK-DAG: %[[C1:.*]] = arith.constant -3.141592653
    // CHECK-DAG: %[[C2:.*]] = arith.constant 1.2502369157
    // CHECK-DAG: %[[C3:.*]] = arith.constant 1.8299117708
    // CHECK-DAG: %[[C4:.*]] = arith.constant 2.8733113535
    // CHECK-DAG: %[[C5:.*]] = arith.constant 3.0097427712
    // CHECK-DAG: %[[C6:.*]] = arith.constant 0.2614370492
    // CHECK-DAG: %[[C7:.*]] = arith.constant -0.131849882
    // CHECK-DAG: %[[C8:.*]] = arith.constant 2.500000e+00
    // CHECK-DAG: %[[C9:.*]] = arith.constant -1.570796326
    // CHECK-DAG: %[[C10:.*]] = arith.constant 1.570796326
    // CHECK-DAG: %[[C11:.*]] = arith.constant 0.785398163

    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit

    // CHECK-NOT: mqtopt.gphase(%[[ANY:.*]])
    // CHECK: %[[Q0_1:.*]] = mqtopt.ry(%[[C11]]) %[[Q0_0]]
    // CHECK: %[[Q0_2:.*]] = mqtopt.rz(%[[C10]]) %[[Q0_1]]
    // CHECK: %[[Q1_1:.*]] = mqtopt.rx(%[[C9]]) %[[Q1_0]]
    // CHECK: %[[Q1_2:.*]] = mqtopt.ry(%[[C8]]) %[[Q1_1]]
    // CHECK: %[[Q0_3:.*]], %[[Q1_3:.*]] = mqtopt.x() %[[Q0_2]] ctrl %[[Q1_2]]
    // CHECK: %[[Q0_4:.*]] = mqtopt.rz(%[[C7]]) %[[Q0_3]]
    // CHECK: %[[Q0_5:.*]] = mqtopt.ry(%[[C6]]) %[[Q0_4]]
    // CHECK: %[[Q0_6:.*]] = mqtopt.rz(%[[C5]]) %[[Q0_5]]
    // CHECK: %[[Q1_4:.*]] = mqtopt.rz(%[[C10]]) %[[Q1_3]]
    // CHECK: %[[Q1_5:.*]] = mqtopt.ry(%[[C10]]) %[[Q1_4]]
    // CHECK: %[[Q0_7:.*]], %[[Q1_6:.*]] = mqtopt.x() %[[Q0_6]] ctrl %[[Q1_5]]
    // CHECK: %[[Q0_8:.*]] = mqtopt.rz(%[[C4]]) %[[Q0_7]]
    // CHECK: %[[Q0_9:.*]] = mqtopt.ry(%[[C3]]) %[[Q0_8]]
    // CHECK: %[[Q0_10:.*]] = mqtopt.rz(%[[C2]]) %[[Q0_9]]
    // CHECK: %[[Q1_7:.*]] = mqtopt.rz(%[[C1]]) %[[Q1_6]]
    // CHECK: %[[Q1_8:.*]] = mqtopt.rx(%[[C0]]) %[[Q1_7]]

    // ensure no other operations are inserted
    // CHECK-NOT: mqtopt.[[ANY:.*]](%[[ANY:.*]])

    // CHECK: mqtopt.deallocQubit %[[Q0_10]]
    // CHECK: mqtopt.deallocQubit %[[Q1_8]]

    %cst0 = arith.constant 2.5 : f64
    %cst1 = arith.constant 1.2 : f64
    %cst2 = arith.constant 0.5 : f64

    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    %q0_x, %q1_x = mqtopt.i() %q0_0 ctrl %q1_0: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_1 = mqtopt.h() %q0_x: !mqtopt.Qubit
    %q1_1, %q0_2 = mqtopt.x() %q1_x ctrl %q0_1: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_3, %q1_2 = mqtopt.rzz(%cst0) %q0_2, %q1_1: !mqtopt.Qubit, !mqtopt.Qubit
    %q1_3 = mqtopt.ry(%cst1) %q1_2: !mqtopt.Qubit
    %q0_4 = mqtopt.rx(%cst1) %q0_3: !mqtopt.Qubit
    %q0_5, %q1_4 = mqtopt.x() %q0_4 ctrl %q1_3: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_6 = mqtopt.rz(%cst2) %q0_5: !mqtopt.Qubit
    // make series longer to enforce decomposition
    %q0_7, %q1_5 = mqtopt.i() %q0_6 ctrl %q1_4: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_8, %q1_6 = mqtopt.i() %q0_7 ctrl %q1_5: !mqtopt.Qubit ctrl !mqtopt.Qubit

    mqtopt.deallocQubit %q0_8
    mqtopt.deallocQubit %q1_6

    return
  }
}

// -----
// This test checks if a complex series containing at least three basis gates is decomposed correctly.

module {
  // CHECK-LABEL: func.func @testThreeBasisGateDecomposition
  func.func @testThreeBasisGateDecomposition() {
    // CHECK-DAG: %[[C0:.*]] = arith.constant 2.5762316133
    // CHECK-DAG: %[[C1:.*]] = arith.constant 2.3561944901
    // CHECK-DAG: %[[C2:.*]] = arith.constant -1.570796326
    // CHECK-DAG: %[[C3:.*]] = arith.constant 0.8584959089
    // CHECK-DAG: %[[C4:.*]] = arith.constant 0.4362460494
    // CHECK-DAG: %[[C5:.*]] = arith.constant 2.4341917193
    // CHECK-DAG: %[[C6:.*]] = arith.constant -0.615479708
    // CHECK-DAG: %[[C7:.*]] = arith.constant 1.0471975511
    // CHECK-DAG: %[[C8:.*]] = arith.constant 2.5261129449
    // CHECK-DAG: %[[C9:.*]] = arith.constant -0.259380512
    // CHECK-DAG: %[[C10:.*]] = arith.constant 1.570796326
    // CHECK-DAG: %[[C11:.*]] = arith.constant -2.52611294
    // CHECK-DAG: %[[C12:.*]] = arith.constant 2.094395102
    // CHECK-DAG: %[[C13:.*]] = arith.constant -0.61547970
    // CHECK-DAG: %[[C14:.*]] = arith.constant 0.694804217
    // CHECK-DAG: %[[C15:.*]] = arith.constant 0.220207934
    // CHECK-DAG: %[[C16:.*]] = arith.constant 1.125358392
    // CHECK-DAG: %[[C17:.*]] = arith.constant -0.03425899
    // CHECK-DAG: %[[C18:.*]] = arith.constant -1.57079632
    // CHECK-DAG: %[[C19:.*]] = arith.constant -2.39270110
    // CHECK-DAG: %[[C20:.*]] = arith.constant -0.78539816

    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit

    // CHECK: mqtopt.gphase(%[[C20]])
    // CHECK: %[[Q0_1:.*]] = mqtopt.ry(%[[C19]]) %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]] = mqtopt.rx(%[[C18]]) %[[Q0_1]] : !mqtopt.Qubit
    // CHECK: %[[Q1_1:.*]] = mqtopt.rz(%[[C17]]) %[[Q1_0]] : !mqtopt.Qubit
    // CHECK: %[[Q1_2:.*]] = mqtopt.ry(%[[C16]]) %[[Q1_1]] : !mqtopt.Qubit
    // CHECK: %[[Q1_3:.*]] = mqtopt.rz(%[[C15]]) %[[Q1_2]] : !mqtopt.Qubit
    // CHECK: %[[Q1_4:.*]], %[[Q0_3:.*]] = mqtopt.x() %[[Q1_3]] ctrl %[[Q0_2]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[Q0_4:.*]] = mqtopt.rx(%[[C14]]) %[[Q0_3]] : !mqtopt.Qubit
    // CHECK: %[[Q1_5:.*]] = mqtopt.rz(%[[C13]]) %[[Q1_4]] : !mqtopt.Qubit
    // CHECK: %[[Q1_6:.*]] = mqtopt.ry(%[[C12]]) %[[Q1_5]] : !mqtopt.Qubit
    // CHECK: %[[Q1_7:.*]] = mqtopt.rz(%[[C11]]) %[[Q1_6]] : !mqtopt.Qubit
    // CHECK: %[[Q1_8:.*]], %[[Q0_5:.*]] = mqtopt.x() %[[Q1_7]] ctrl %[[Q0_4]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[Q0_6:.*]] = mqtopt.rz(%[[C10]]) %[[Q0_5]] : !mqtopt.Qubit
    // CHECK: %[[Q0_7:.*]] = mqtopt.ry(%[[C9]]) %[[Q0_6]] : !mqtopt.Qubit
    // CHECK: %[[Q1_9:.*]] = mqtopt.rz(%[[C8]]) %[[Q1_8]] : !mqtopt.Qubit
    // CHECK: %[[Q1_10:.*]] = mqtopt.ry(%[[C7]]) %[[Q1_9]] : !mqtopt.Qubit
    // CHECK: %[[Q1_11:.*]] = mqtopt.rz(%[[C6]]) %[[Q1_10]] : !mqtopt.Qubit
    // CHECK: %[[Q1_12:.*]], %[[Q0_8:.*]] = mqtopt.x() %[[Q1_11]] ctrl %[[Q0_7]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[Q0_9:.*]] = mqtopt.rz(%[[C5]]) %[[Q0_8]] : !mqtopt.Qubit
    // CHECK: %[[Q0_10:.*]] = mqtopt.ry(%[[C4]]) %[[Q0_9]] : !mqtopt.Qubit
    // CHECK: %[[Q0_11:.*]] = mqtopt.rz(%[[C3]]) %[[Q0_10]] : !mqtopt.Qubit
    // CHECK: %[[Q1_13:.*]] = mqtopt.rz(%[[C2]]) %[[Q1_12]] : !mqtopt.Qubit
    // CHECK: %[[Q1_14:.*]] = mqtopt.ry(%[[C1]]) %[[Q1_13]] : !mqtopt.Qubit
    // CHECK: %[[Q1_15:.*]] = mqtopt.rz(%[[C0]]) %[[Q1_14]] : !mqtopt.Qubit

    // CHECK: mqtopt.deallocQubit %[[Q0_11]]
    // CHECK: mqtopt.deallocQubit %[[Q1_15]]

    %cst0 = arith.constant 2.5 : f64
    %cst1 = arith.constant 1.2 : f64
    %cst2 = arith.constant 0.5 : f64

    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit
    %q2_0 = mqtopt.allocQubit

    %q0_x, %q1_x = mqtopt.i() %q0_0 ctrl %q1_0: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_1 = mqtopt.h() %q0_x: !mqtopt.Qubit
    %q1_1, %q0_2 = mqtopt.x() %q1_x ctrl %q0_1: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_3, %q1_2 = mqtopt.rzz(%cst0) %q0_2, %q1_1: !mqtopt.Qubit, !mqtopt.Qubit
    %q1_3 = mqtopt.ry(%cst1) %q1_2: !mqtopt.Qubit
    %q0_4 = mqtopt.rx(%cst1) %q0_3: !mqtopt.Qubit
    %q0_5, %q1_4 = mqtopt.x() %q0_4 ctrl %q1_3: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_6 = mqtopt.rz(%cst2) %q0_5: !mqtopt.Qubit
    %q0_7, %q1_5 = mqtopt.rxx(%cst0) %q0_6, %q1_4: !mqtopt.Qubit, !mqtopt.Qubit
    %q0_8, %q1_6 = mqtopt.ryy(%cst2) %q0_7, %q1_5: !mqtopt.Qubit, !mqtopt.Qubit
    // make series longer to enforce decomposition
    %q0_9, %q1_7 = mqtopt.i() %q0_8 ctrl %q1_6: !mqtopt.Qubit ctrl !mqtopt.Qubit

    mqtopt.deallocQubit %q0_9
    mqtopt.deallocQubit %q1_7
    mqtopt.deallocQubit %q2_0

    return
  }
}

// -----
// This test checks if the repeated application of the decomposition works by "interrupting" the first series and then having a second one.

module {
  // CHECK-LABEL: func.func @testRepeatedDecomposition
  func.func @testRepeatedDecomposition() {
    return
  }
}

// -----
// This test checks if two single-qubit series (connected by an identity) remain separate without the insertion of a basis gate.

module {
  // CHECK-LABEL: func.func @testSingleQubitSeries
  func.func @testSingleQubitSeries() {
    // CHECK-DAG: %[[C0:.*]] = arith.constant 2.400000
    // CHECK-DAG: %[[C1:.*]] = arith.constant 2.500000

    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]] = mqtopt.ry(%[[C1]]) %[[Q0_0]]
    // CHECK: %[[Q1_1:.*]] = mqtopt.rx(%[[C0]]) %[[Q1_0]]

    // ensure no other operations are inserted
    // CHECK-NOT: mqtopt.[[ANY:.*]](%[[ANY:.*]])

    // CHECK: mqtopt.deallocQubit %[[Q0_1]]
    // CHECK: mqtopt.deallocQubit %[[Q1_1]]

    %cst0 = arith.constant 2.5 : f64
    %cst1 = arith.constant 1.2 : f64

    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    %q0_1, %q1_1 = mqtopt.i() %q0_0 ctrl %q1_0: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_2 = mqtopt.ry(%cst0) %q0_1: !mqtopt.Qubit
    %q1_2 = mqtopt.rx(%cst1) %q1_1: !mqtopt.Qubit
    %q1_3 = mqtopt.rx(%cst1) %q1_2: !mqtopt.Qubit

    mqtopt.deallocQubit %q0_2
    mqtopt.deallocQubit %q1_3

    return
  }
}

// -----
// This test checks if a two-qubit series starting with two separate single-qubit chains is fully decomposed.

module {
  // CHECK-LABEL: func.func @testSeriesSingleQubitBacktracking
  func.func @testSeriesSingleQubitBacktracking() {
    // CHECK-DAG: %[[C0:.*]] = arith.constant -2.4269908
    // CHECK-DAG: %[[C1:.*]] = arith.constant 3.14159265
    // CHECK-DAG: %[[C2:.*]] = arith.constant 0.85619449
    // CHECK-DAG: %[[C3:.*]] = arith.constant -2.3561944
    // CHECK-DAG: %[[C4:.*]] = arith.constant -1.5707963
    // CHECK-DAG: %[[C5:.*]] = arith.constant -1.5707963
    // CHECK-DAG: %[[C6:.*]] = arith.constant 2.35619449
    // CHECK-DAG: %[[C7:.*]] = arith.constant -3.1415926

    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit

    // CHECK: mqtopt.gphase(%[[C7]])
    // CHECK: %[[Q0_1:.*]] = mqtopt.rz(%[[C6]]) %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]] = mqtopt.ry(%[[C5]]) %[[Q0_1]] : !mqtopt.Qubit
    // CHECK: %[[Q1_1:.*]] = mqtopt.ry(%[[C4]]) %[[Q1_0]] : !mqtopt.Qubit
    // CHECK: %[[Q1_2:.*]] = mqtopt.rz(%[[C3]]) %[[Q1_1]] : !mqtopt.Qubit
    // CHECK: %[[Q0_3:.*]], %[[Q1_3:.*]] = mqtopt.x() %[[Q0_2]] ctrl %[[Q1_2]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[Q0_4:.*]] = mqtopt.rx(%[[C2]]) %[[Q0_3]] : !mqtopt.Qubit
    // CHECK: %[[Q0_5:.*]] = mqtopt.ry(%[[C1]]) %[[Q0_4]] : !mqtopt.Qubit
    // CHECK: %[[Q1_4:.*]] = mqtopt.rz(%[[C0]]) %[[Q1_3]] : !mqtopt.Qubit

    // CHECK: mqtopt.deallocQubit %[[Q0_5]]
    // CHECK: mqtopt.deallocQubit %[[Q1_4]]

    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    %cst0 = arith.constant 1.5 : f64
    %cst1 = arith.constant 0.2 : f64
    %cst2 = arith.constant 0.9 : f64

    %q0_1 = mqtopt.h() %q0_0: !mqtopt.Qubit
    %q0_2 = mqtopt.rx(%cst0) %q0_1: !mqtopt.Qubit
    %q0_3 = mqtopt.x() %q0_2: !mqtopt.Qubit
    %q1_1 = mqtopt.h() %q1_0: !mqtopt.Qubit
    %q1_2 = mqtopt.rz(%cst0) %q1_1: !mqtopt.Qubit
    %q0_4, %q1_3 = mqtopt.x() %q0_3 ctrl %q1_2: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q1_4, %q0_5 = mqtopt.i() %q1_3 ctrl %q0_4: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q1_5, %q0_6 = mqtopt.i() %q1_4 ctrl %q0_5: !mqtopt.Qubit ctrl !mqtopt.Qubit

    mqtopt.deallocQubit %q0_6
    mqtopt.deallocQubit %q1_5

    return
  }
}
