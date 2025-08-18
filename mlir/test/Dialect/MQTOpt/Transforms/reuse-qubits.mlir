// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --reuse-qubits | FileCheck %s

// -----
// This test checks that measurements can be lifted over controlled gates if the measurement target is a positive control and the only control of the gate.
// In this case, the controlled gate is replaced by a conditional gate that uses the measurement outcome as a condition.

module {
  // CHECK-LABEL: func.func @testSimple
  func.func @testSimple() -> (i1, i1) {
    // CHECK: %[[Q0_0:.*]] = "mqtopt.allocQubit"() : () -> !mqtopt.Qubit
    // CHECK: %[[Q0_1:.*]] = mqtopt.h() %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]], %[[C0:.*]] = "mqtopt.measure"(%[[Q0_1]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    // CHECK: %[[Q0_3:.*]] = "mqtopt.reset"(%[[Q0_2]]) : (!mqtopt.Qubit) -> !mqtopt.Qubit
    // CHECK: %[[Q0_4:.*]] = "mqtopt.h"(%[[Q0_3]]) : (!mqtopt.Qubit) -> !mqtopt.Qubit
    // CHECK: %[[Q0_5:.*]], %[[C1:.*]] = "mqtopt.measure"(%[[Q0_4]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    // CHECK: "mqtopt.deallocQubit"(%[[Q0_5]]) : (!mqtopt.Qubit) -> ()
    // CHECK: return %[[C0]], %[[C1]] : i1, i1
    %q0_0 = "mqtopt.allocQubit"() : () -> !mqtopt.Qubit
    %q1_0 = "mqtopt.allocQubit"() : () -> !mqtopt.Qubit

    %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
    %q1_1 = mqtopt.h() %q1_0 : !mqtopt.Qubit

    %q0_2, %c0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    %q1_2, %c1 = "mqtopt.measure"(%q1_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

    "mqtopt.deallocQubit"(%q0_2) : (!mqtopt.Qubit) -> ()
    "mqtopt.deallocQubit"(%q1_2) : (!mqtopt.Qubit) -> ()
    return %c0, %c1 : i1, i1
  }
}
