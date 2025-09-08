// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --swap-reconstruction-and-elision | FileCheck %s

// -----
// This test checks that a single SWAP gate is removed correctly.

module {
  // CHECK-LABEL: func.func @testSingleElidePermutation
  func.func @testSingleElidePermutation() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit

    // ========================== Check for operations that should be canceled ==============================
    // CHECK-NOT: %[[ANY:.*]] = mqtopt.swap()

    // ========================== Check for operations that should not be canceled ==============================
    // CHECK: %[[Q0_1:.*]] = mqtopt.x() %[[Q0_0]]

    // CHECK: mqtopt.deallocQubit %[[Q1_0]]
    // CHECK: mqtopt.deallocQubit %[[Q0_1]]

    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    %q0_1, %q1_1 = mqtopt.swap() %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
    %q1_2 = mqtopt.x() %q1_1 : !mqtopt.Qubit

    mqtopt.deallocQubit %q0_1
    mqtopt.deallocQubit %q1_2

    return
  }
}


// -----
// This test checks that a controlled SWAP gate is not removed.

module {
  // CHECK-LABEL: func.func @testControlledSwapNoElidePermutation
  func.func @testControlledSwapNoElidePermutation() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q2_0:.*]] = mqtopt.allocQubit

    // ========================== Check for operations that should not be canceled ==============================
    // CHECK: %[[Q01_1:.*]]:2, %[[Q2_1:.*]] = mqtopt.swap() %[[Q0_0]], %[[Q1_0]] ctrl %[[Q2_0]]

    // CHECK: mqtopt.deallocQubit %[[Q01_1]]#0
    // CHECK: mqtopt.deallocQubit %[[Q01_1]]#1
    // CHECK: mqtopt.deallocQubit %[[Q2_1]]

    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit
    %q2_0 = mqtopt.allocQubit

    %q0_1, %q1_1, %q2_1 = mqtopt.swap() %q0_0, %q1_0 ctrl %q2_0 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit

    mqtopt.deallocQubit %q0_1
    mqtopt.deallocQubit %q1_1
    mqtopt.deallocQubit %q2_1

    return
  }
}


// -----
// This test checks that all removable SWAP gates are removed.

module {
  // CHECK-LABEL: func.func @testMultiSwapElidePermutation
  func.func @testMultiSwapElidePermutation() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q2_0:.*]] = mqtopt.allocQubit

    // ========================== Check for operations that should not be canceled ==============================
    // CHECK-NOT: %[[ANY:.*]] = mqtopt.swap()

    // ========================== Check for operations that should not be canceled ==============================
    // CHECK: %[[Q12_1:.*]]:2, %[[Q0_1:.*]] = mqtopt.swap() %[[Q1_0]], %[[Q2_0]] ctrl %[[Q0_0]]

    // CHECK: mqtopt.deallocQubit %[[Q12_1]]#0
    // CHECK: mqtopt.deallocQubit %[[Q0_1]]
    // CHECK: mqtopt.deallocQubit %[[Q12_1]]#1

    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit
    %q2_0 = mqtopt.allocQubit

    %q0_1, %q1_1 = mqtopt.swap() %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
    %q1_2, %q2_1 = mqtopt.swap() %q1_1, %q2_0 : !mqtopt.Qubit, !mqtopt.Qubit
    %q0_2, %q1_3, %q2_2 = mqtopt.swap() %q0_1, %q1_2 ctrl %q2_1 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q1_4, %q2_3 = mqtopt.swap() %q1_3, %q2_2 : !mqtopt.Qubit, !mqtopt.Qubit

    mqtopt.deallocQubit %q0_2
    mqtopt.deallocQubit %q1_4
    mqtopt.deallocQubit %q2_3

    return
  }
}
