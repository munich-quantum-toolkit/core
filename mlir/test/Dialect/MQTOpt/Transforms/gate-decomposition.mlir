// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --gate-decomposition | FileCheck %s

// -----
// This test checks if single-qubit consecutive self-inverses are canceled correctly.

module {
  // CHECK-LABEL: func.func @testNegationSeries
  func.func @testNegationSeries() {
    // CHECK: %[[Q0_0]] = mqtopt.allocQubit

    // CHECK: %[[C_0:.*]] = arith.constant 0.000000e+00 : f64
    // CHECK: %[[Q0_1:.*]] = mqtopt.rz

    // CHECK-NOT: %[[ANY:.*]] = mqtopt.x

    // CHECK: mqtopt.deallocQubit %[[Q0_1]]

    %q0_0 = mqtopt.allocQubit

    %q0_1 = mqtopt.x() %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.x() %q0_1 : !mqtopt.Qubit
    %q0_3 = mqtopt.x() %q0_2 : !mqtopt.Qubit
    %q0_4 = mqtopt.x() %q0_3 : !mqtopt.Qubit
    %q0_5 = mqtopt.x() %q0_4 : !mqtopt.Qubit
    %q0_6 = mqtopt.x() %q0_5 : !mqtopt.Qubit

    mqtopt.deallocQubit %q0_6

    return
  }
}


module {
  // CHECK-LABEL: func.func @testNoMergeSmallSeries
  func.func @testNoMergeSmallSeries() {
    // CHECK: %[[Q0_0]] = mqtopt.allocQubit

    // CHECK: %[[C_0:.*]] = arith.constant 3.000000e+00 : f64
    // CHECK: %[[Q0_1:.*]] = mqtopt.rx(%[[C_0]]) %[[ANY:.*]] : !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]] = mqtopt.rx(%[[C_0]]) %[[ANY:.*]] : !mqtopt.Qubit
    // CHECK: %[[Q0_3:.*]] = mqtopt.rx(%[[C_0]]) %[[ANY:.*]] : !mqtopt.Qubit

    // CHECK-NOT: %[[ANY:.*]] = mqtopt.rz
    // CHECK-NOT: %[[ANY:.*]] = mqtopt.ry

    // CHECK: mqtopt.deallocQubit %[[Q0_3]]

    %q0_0 = mqtopt.allocQubit

    %c_0 = arith.constant 1.000000e+00 : f64
    %q0_1 = mqtopt.rx(%c_0) %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.rz(%c_0) %q0_1 : !mqtopt.Qubit
    %q0_3 = mqtopt.ry(%c_0) %q0_2 : !mqtopt.Qubit

    mqtopt.deallocQubit %q0_3

    return
  }
}
