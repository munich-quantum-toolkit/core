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
    // CHECK: %[[Q0_0]], %[[Q1_0]] = mqtopt.x() %[[Q0_0]] ctrl %[[Q1_0]]

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
    // CHECK: %[[C2:.*]] = arith.constant -1.5707
    // CHECK: %[[C1:.*]] = arith.constant 3.14159
    // CHECK: %[[C0:.*]] = arith.constant 1.57079

    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit

    // CHECK: mqtopt.gphase(%[[C0:.*]])
    // CHECK: %[[Q0_1:.*]] = mqtopt.rz(%[[C0]]) %[[Q0_0]]
    // CHECK: %[[Q0_2:.*]] = mqtopt.ry(%[[C1:.*]]) %[[Q0_1]]
    // CHECK: %[[Q0_3:.*]] = mqtopt.rz(%[[C2:.*]]) %[[Q0_2]]

    // CHECK: mqtopt.deallocQubit %[[Q0_3]]
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
    // CHECK: %[[C2:.*]] = arith.constant -1.5707
    // CHECK: %[[C1:.*]] = arith.constant 3.14159
    // CHECK: %[[C0:.*]] = arith.constant 1.57079

    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit

    // CHECK: mqtopt.gphase(%[[C0:.*]])
    // CHECK: %[[Q0_1:.*]] = mqtopt.rz(%[[C0]]) %[[Q0_0]]
    // CHECK: %[[Q0_2:.*]] = mqtopt.ry(%[[C1:.*]]) %[[Q0_1]]
    // CHECK: %[[Q0_3:.*]] = mqtopt.rz(%[[C2:.*]]) %[[Q0_2]]

    // CHECK: mqtopt.deallocQubit %[[Q0_3]]
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
    // CHECK: %[[C2:.*]] = arith.constant -1.5707
    // CHECK: %[[C1:.*]] = arith.constant 3.14159
    // CHECK: %[[C0:.*]] = arith.constant 1.57079

    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit

    // CHECK: mqtopt.gphase(%[[C0:.*]])
    // CHECK: %[[Q0_1:.*]] = mqtopt.rz(%[[C0]]) %[[Q0_0]]
    // CHECK: %[[Q0_2:.*]] = mqtopt.ry(%[[C1:.*]]) %[[Q0_1]]
    // CHECK: %[[Q0_3:.*]] = mqtopt.rz(%[[C2:.*]]) %[[Q0_2]]

    // CHECK: mqtopt.deallocQubit %[[Q0_3]]
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

    // CHECK-NOT: mqtopt.ry
    // CHECK-NOT: mqtopt.rz
    // CHECK-NOT: mqtopt.gphase(%[[ANY:.*]])

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

module {
  // CHECK-LABEL: func.func @testTwoBasisGateDecomposition
  func.func @testTwoBasisGateDecomposition() {
    // CHECK: %[[C0:.*]] = arith.constant -2.356194490
    // CHECK: %[[C1:.*]] = arith.constant -1.070796326
    // CHECK: %[[C2:.*]] = arith.constant -0.370796326
    // CHECK: %[[C3:.*]] = arith.constant -2.526112944
    // CHECK: %[[C4:.*]] = arith.constant 1.0471975511
    // CHECK: %[[C5:.*]] = arith.constant 0.6154797086
    // CHECK: %[[C6:.*]] = arith.constant -3.141592653
    // CHECK: %[[C7:.*]] = arith.constant 2.7707963267
    // CHECK: %[[C8:.*]] = arith.constant -1.570796326
    // CHECK: %[[C9:.*]] = arith.constant 0.7853981633
    // CHECK: %[[C10:.*]] = arith.constant 2.5000
    // CHECK: %[[C11:.*]] = arith.constant 1.570796326
    // CHECK: %[[C12:.*]] = arith.constant -1.57079632
    // CHECK: %[[C13:.*]] = arith.constant 8.881784197

    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit

    // CHECK: mqtopt.gphase(%[[C13]])
    // CHECK: %[[Q0_1:.*]] = mqtopt.rz(%[[C12]]) %[[Q0_0]]
    // CHECK: %[[Q0_2:.*]] = mqtopt.ry(%[[C11]]) %[[Q0_1]]
    // CHECK: %[[Q0_3:.*]] = mqtopt.rz(%[[C11]]) %[[Q0_2]]
    // CHECK: %[[Q1_1:.*]] = mqtopt.rz(%[[C10]]) %[[Q1_0]]
    // CHECK: %[[Q1_2:.*]] = mqtopt.ry(%[[C9]]) %[[Q1_1]]
    // CHECK: %[[Q1_3:.*]] = mqtopt.rz(%[[C8]]) %[[Q1_2]]
    // CHECK: %[[Q1_4:.*]], %[[Q0_4:.*]] = mqtopt.x() %[[Q1_3]] ctrl %[[Q0_3]]
    // CHECK: %[[Q0_5:.*]] = mqtopt.rz(%[[C11]]) %[[Q0_4]]
    // CHECK: %[[Q0_6:.*]] = mqtopt.ry(%[[C7]]) %[[Q0_5]]
    // CHECK: %[[Q0_7:.*]] = mqtopt.rz(%[[C6]]) %[[Q0_6]]
    // CHECK: %[[Q1_5:.*]] = mqtopt.rz(%[[C5]]) %[[Q1_4]]
    // CHECK: %[[Q1_6:.*]] = mqtopt.ry(%[[C4]]) %[[Q1_5]]
    // CHECK: %[[Q1_7:.*]] = mqtopt.rz(%[[C3]]) %[[Q1_6]]
    // CHECK: %[[Q1_8:.*]], %[[Q0_8:.*]] = mqtopt.x() %[[Q1_7]] ctrl %[[Q0_7]]
    // CHECK: %[[Q0_9:.*]] = mqtopt.ry(%[[C2]]) %[[Q0_8]]
    // CHECK: %[[Q0_10:.*]] = mqtopt.rz(%[[C1]]) %[[Q0_9]]
    // CHECK: %[[Q1_9:.*]] = mqtopt.rz(%[[C12]]) %[[Q1_8]]
    // CHECK: %[[Q1_10:.*]] = mqtopt.ry(%[[C0]]) %[[Q1_9]]

    // CHECK: mqtopt.deallocQubit %[[Q0_10]]
    // CHECK: mqtopt.deallocQubit %[[Q1_10]]

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
    %q0_7 = mqtopt.i() %q0_6: !mqtopt.Qubit
    %q0_8 = mqtopt.i() %q0_7: !mqtopt.Qubit
    %q0_9 = mqtopt.i() %q0_8: !mqtopt.Qubit
    %q1_5 = mqtopt.i() %q1_4: !mqtopt.Qubit
    %q1_6 = mqtopt.i() %q1_5: !mqtopt.Qubit
    %q1_7 = mqtopt.i() %q1_6: !mqtopt.Qubit
    %q1_8 = mqtopt.i() %q1_7: !mqtopt.Qubit
    %q1_9 = mqtopt.i() %q1_8: !mqtopt.Qubit

    mqtopt.deallocQubit %q0_9
    mqtopt.deallocQubit %q1_9

    return
  }
}

// -----
// This test checks if two single-qubit series (connected by an identity) remain separate without the insertion of a basis gate.

module {
  // CHECK-LABEL: func.func @testSingleQubitSeries
  func.func @testSingleQubitSeries() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]], %[[Q1_1:.*]] = mqtopt.x() %[[Q0_0]] ctrl %[[Q1_0]]

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
// This test checks if two single-qubit series (connected by an identity) remain separate without the insertion of a basis gate.

module {
  // CHECK-LABEL: func.func @testThreeBasisGateDecomposition
  func.func @testThreeBasisGateDecomposition() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q2_0:.*]] = mqtopt.allocQubit

    // CHECK-NOT: mqtopt.gphase(%[[ANY:.*]])

    // CHECK: mqtopt.deallocQubit %[[Q0_5]]
    // CHECK: mqtopt.deallocQubit %[[Q1_4]]
    // CHECK: mqtopt.deallocQubit %[[Q2_1]]

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
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q2_0:.*]] = mqtopt.allocQubit

    // CHECK-NOT: mqtopt.gphase(%[[ANY:.*]])

    // CHECK: mqtopt.deallocQubit %[[Q0_5]]
    // CHECK: mqtopt.deallocQubit %[[Q1_4]]
    // CHECK: mqtopt.deallocQubit %[[Q2_1]]

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
