// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --gate-decomposition | FileCheck %s

// -----
// This test checks if two-qubit consecutive controlled self-inverses are canceled correctly.

module {
  // CHECK-LABEL: func.func @testEvenNegationSeries
  func.func @testEvenNegationSeries() {
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

module {
  // CHECK-LABEL: func.func @testSeriesOneQubitOpInbetween
  func.func @testSeriesOneQubitOpInbetween() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit

    // CHECK: mqtopt.gphase(%[[C0:.*]])
    // CHECK: %[[Q0_1:.*]] = mqtopt.ry(%[[C1:.*]]) %[[Q0_0]]
    // CHECK: %[[Q0_2:.*]] = mqtopt.rz(%[[C1]]) %[[Q0_1]]

    // CHECK: mqtopt.deallocQubit %[[Q0_2]]
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

module {
  // CHECK-LABEL: func.func @testSeriesStartingOneQubitOp
  func.func @testSeriesStartingOneQubitOp() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]], %[[Q1_1:.*]] = mqtopt.x() %[[Q0_0]] ctrl %[[Q1_0]]

    // CHECK: mqtopt.deallocQubit %[[Q0_1]]
    // CHECK: mqtopt.deallocQubit %[[Q1_1]]

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

module {
  // CHECK-LABEL: func.func @testSeriesEndingOneQubitOp
  func.func @testSeriesEndingOneQubitOp() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit

    // CHECK: %[[Q0_1:.*]], %[[Q1_1:.*]] = mqtopt.x() %[[Q0_0]]

    // CHECK: mqtopt.deallocQubit %[[Q0_1]]
    // CHECK: mqtopt.deallocQubit %[[Q1_1]]

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
