// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --swap-reconstruction | FileCheck %s

// -----
// This test checks that consecutive CNOT gates which match a SWAP gate are merged correctly.

module {
  // CHECK-LABEL: func.func @testSingleSwapReconstruction
  func.func @testSingleSwapReconstruction() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit

    // ============================ Check for operations that should be inserted ============================
    // CHECK: %[[Q01_1:.*]]:2 = mqtopt.swap() %[[Q0_0]], %[[Q1_0]] : !mqtopt.Qubit, !mqtopt.Qubit

    // ========================== Check for operations that should be canceled ==============================
    // CHECK-NOT: %[[ANY:.*]] = mqtopt.x()

    // CHECK: mqtopt.deallocQubit %[[Q01_1]]#1
    // CHECK: mqtopt.deallocQubit %[[Q01_1]]#0

    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    %q0_1, %q1_1 = mqtopt.x() %q0_0 ctrl %q1_0: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q1_2, %q0_2 = mqtopt.x() %q1_1 ctrl %q0_1: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_3, %q1_3 = mqtopt.x() %q0_2 ctrl %q1_2: !mqtopt.Qubit ctrl !mqtopt.Qubit

    mqtopt.deallocQubit %q0_3
    mqtopt.deallocQubit %q1_3

    return
  }
}
