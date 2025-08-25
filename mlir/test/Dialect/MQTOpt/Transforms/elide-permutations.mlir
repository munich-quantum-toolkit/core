// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --elide-permutations | FileCheck %s

// -----
// This test checks that a single SWAP gate is removed correctly.

module {
  // CHECK-LABEL: func.func @testSingleElidePermutation
  func.func @testSingleElidePermutation() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit

    // ========================== Check for operations that should be canceled ==============================
    // CHECK-NOT: %[[ANY:.*]] = mqtopt.swap()

    // CHECK: mqtopt.deallocQubit %[[Q1_0]]
    // CHECK: mqtopt.deallocQubit %[[Q0_0]]

    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    %q0_1, %q1_1 = mqtopt.swap() %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit

    mqtopt.deallocQubit %q0_1
    mqtopt.deallocQubit %q1_1

    return
  }
}
