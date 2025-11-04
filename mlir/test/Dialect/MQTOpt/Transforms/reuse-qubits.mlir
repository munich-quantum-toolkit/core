// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --reuse-qubits | FileCheck %s

// -----
// This test checks that qubits are reused if they are disjointly used.

module {
  // CHECK-LABEL: func.func @testSimple
  func.func @testSimple() -> (i1, i1) {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q0_1:.*]] = mqtopt.h() %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]], %[[C0:.*]] = mqtopt.measure %[[Q0_1]]
    // CHECK: %[[Q0_3:.*]] = mqtopt.reset %[[Q0_2]]
    // CHECK: %[[Q0_4:.*]] = mqtopt.h() %[[Q0_3]] : !mqtopt.Qubit
    // CHECK: %[[Q0_5:.*]], %[[C1:.*]] = mqtopt.measure %[[Q0_4]]
    // CHECK: mqtopt.deallocQubit %[[Q0_5]]
    // CHECK: return %[[C0]], %[[C1]] : i1, i1
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
    %q1_1 = mqtopt.h() %q1_0 : !mqtopt.Qubit

    %q0_2, %c0 = mqtopt.measure %q0_1
    %q1_2, %c1 = mqtopt.measure %q1_1

    mqtopt.deallocQubit %q0_2
    mqtopt.deallocQubit %q1_2
    return %c0, %c1 : i1, i1
  }
}

// -----
// This test checks that qubit reuse is not applied if the used qubits intersect.

module {
  // CHECK-LABEL: func.func @testNoReuse
  func.func @testNoReuse() -> (i1, i1) {
    // CHECK: %[[q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[q0_1:.*]] = mqtopt.h() %[[q0_0]] : !mqtopt.Qubit
    // CHECK: %[[q1_1:.*]], %[[q0_2:.*]] = mqtopt.h() %[[q1_0]] ctrl %[[q0_1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[q0_3:.*]], %[[c0:.*]] = mqtopt.measure %[[q0_2]]
    // CHECK: %[[q1_2:.*]], %[[c1:.*]] = mqtopt.measure %[[q1_1]]
    // CHECK: mqtopt.deallocQubit %[[q0_3]]
    // CHECK: mqtopt.deallocQubit %[[q1_2]]
    // CHECK: return %[[c0]], %[[c1]] : i1, i1
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
    %q1_1, %q0_2 = mqtopt.h() %q1_0 ctrl %q0_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit

    %q0_3, %c0 = mqtopt.measure %q0_2
    %q1_2, %c1 = mqtopt.measure %q1_1

    mqtopt.deallocQubit %q0_3
    mqtopt.deallocQubit %q1_2
    return %c0, %c1 : i1, i1
  }
}

// -----
// This test checks that qubit reuse is applied correctly in a context with several qubits.

module {
  // CHECK-LABEL: func.func @testReuseOne
  func.func @testReuseOne() -> (i1, i1, i1) {
    // CHECK: %[[q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[q0_1:.*]] = mqtopt.h() %[[q0_0]] : !mqtopt.Qubit
    // CHECK: %[[q1_1:.*]], %[[q0_2:.*]] = mqtopt.h() %[[q1_0]] ctrl %[[q0_1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[q0_3:.*]], %[[c0:.*]] = mqtopt.measure %[[q0_2]]
    // CHECK: %[[q1_2:.*]], %[[c1:.*]] = mqtopt.measure %[[q1_1]]
    // CHECK: mqtopt.deallocQubit %[[q0_3]]
    // CHECK: %[[q2_0:.*]] = mqtopt.reset %[[q1_2]]
    // CHECK: %[[q2_1:.*]] = mqtopt.h() %[[q2_0]] : !mqtopt.Qubit
    // CHECK: %[[q2_2:.*]], %[[c2:.*]] = mqtopt.measure %[[q2_1]]
    // CHECK: mqtopt.deallocQubit %[[q2_2]]
    // CHECK: return %[[c0]], %[[c1]], %[[c2]] : i1, i1, i1
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit
    %q2_0 = mqtopt.allocQubit

    %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
    %q1_1, %q0_2 = mqtopt.h() %q1_0 ctrl %q0_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q2_1 = mqtopt.h() %q2_0 : !mqtopt.Qubit

    %q0_3, %c0 = mqtopt.measure %q0_2
    %q1_2, %c1 = mqtopt.measure %q1_1
    %q2_2, %c2 = mqtopt.measure %q2_1

    mqtopt.deallocQubit %q0_3
    mqtopt.deallocQubit %q1_2
    mqtopt.deallocQubit %q2_2
    return %c0, %c1, %c2 : i1, i1, i1
  }
}

// -----
// This test checks that qubit reuse is even applied correctly if the reused qubit appears after the alloc but could be moved above it.

module {
  // CHECK-LABEL: func.func @testReuseOneWithBadOrdering
  func.func @testReuseOneWithBadOrdering() -> (i1, i1, i1) {
    // CHECK: %[[q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[q0_1:.*]] = mqtopt.h() %[[q0_0]] : !mqtopt.Qubit
    // CHECK: %[[q1_1:.*]], %[[q0_2:.*]] = mqtopt.h() %[[q1_0]] ctrl %[[q0_1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[q0_3:.*]], %[[c0:.*]] = mqtopt.measure %[[q0_2]]
    // CHECK: %[[q1_2:.*]], %[[c1:.*]] = mqtopt.measure %[[q1_1]]
    // CHECK: mqtopt.deallocQubit %[[q0_3]]
    // CHECK: %[[q2_0:.*]] = mqtopt.reset %[[q1_2]]
    // CHECK: %[[q2_1:.*]] = mqtopt.h() %[[q2_0]] : !mqtopt.Qubit
    // CHECK: %[[q2_2:.*]], %[[c2:.*]] = mqtopt.measure %[[q2_1]]
    // CHECK: mqtopt.deallocQubit %[[q2_2]]
    // CHECK: return %[[c0]], %[[c1]], %[[c2]] : i1, i1, i1
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit
    %q2_0 = mqtopt.allocQubit

    %q2_1 = mqtopt.h() %q2_0 : !mqtopt.Qubit

    %q2_2, %c2 = mqtopt.measure %q2_1

    mqtopt.deallocQubit %q2_2

    %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
    %q1_1, %q0_2 = mqtopt.h() %q1_0 ctrl %q0_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_3, %c0 = mqtopt.measure %q0_2
    %q1_2, %c1 = mqtopt.measure %q1_1

    mqtopt.deallocQubit %q0_3
    mqtopt.deallocQubit %q1_2
    return %c0, %c1, %c2 : i1, i1, i1
  }
}

// -----
// This test checks that qubit reuse can be applied multiple times.

module {
  // CHECK-LABEL: func.func @testReuseMultiple
  func.func @testReuseMultiple() -> (i1, i1, i1) {
    // CHECK: %[[q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[q0_1:.*]] = mqtopt.h() %[[q0_0]] : !mqtopt.Qubit
    // CHECK: %[[q0_2:.*]], %[[c0:.*]] = mqtopt.measure %[[q0_1]]
    // q2 is reused first because we start from the last dealloc.
    // CHECK: %[[q2_0:.*]] = mqtopt.reset %[[q0_2]]
    // CHECK: %[[q2_1:.*]] = mqtopt.h() %[[q2_0]] : !mqtopt.Qubit
    // CHECK: %[[q2_2:.*]], %[[c1:.*]] = mqtopt.measure %[[q2_1]]
    // CHECK: %[[q1_0:.*]] = mqtopt.reset %[[q2_2]]
    // CHECK: %[[q1_1:.*]] = mqtopt.h() %[[q1_0]] : !mqtopt.Qubit
    // CHECK: %[[q1_2:.*]], %[[c2:.*]] = mqtopt.measure %[[q1_1]]
    // CHECK: mqtopt.deallocQubit %[[q1_2]]
    // CHECK: return %[[c0]], %[[c1]], %[[c2]] : i1, i1, i1
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit
    %q2_0 = mqtopt.allocQubit

    %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
    %q1_1 = mqtopt.h() %q1_0 : !mqtopt.Qubit
    %q2_1 = mqtopt.h() %q2_0 : !mqtopt.Qubit

    %q0_2, %c0 = mqtopt.measure %q0_1
    %q1_2, %c1 = mqtopt.measure %q1_1
    %q2_2, %c2 = mqtopt.measure %q2_1

    mqtopt.deallocQubit %q0_2
    mqtopt.deallocQubit %q1_2
    mqtopt.deallocQubit %q2_2

    return %c0, %c1, %c2 : i1, i1, i1
  }
}

// -----
// This test checks that qubit reuse can be applied even if a path exists between the qubits in the interaction graph

module {
  // CHECK-LABEL: func.func @testReuseWhenPathExists
  func.func @testReuseWhenPathExists() -> (i1, i1, i1) {
    // CHECK: %[[q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[q0_1:.*]] = mqtopt.h() %[[q0_0]] : !mqtopt.Qubit
    // CHECK: %[[q1_1:.*]], %[[q0_2:.*]] = mqtopt.h() %[[q1_0]] ctrl %[[q0_1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[q1_2:.*]], %[[c1:.*]] = mqtopt.measure %[[q1_1]]
    // CHECK: %[[q2_0:.*]] = mqtopt.reset %[[q1_2]]
    // CHECK: %[[q2_1:.*]], %[[q0_3:.*]] = mqtopt.h() %[[q2_0]] ctrl %[[q0_2]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[q0_4:.*]], %[[c0:.*]] = mqtopt.measure %[[q0_3]]
    // CHECK: mqtopt.deallocQubit %[[q0_4]]
    // CHECK: %[[q2_2:.*]], %[[c2:.*]] = mqtopt.measure %[[q2_1]]
    // CHECK: mqtopt.deallocQubit %[[q2_2]]
    // CHECK: return %[[c0]], %[[c1]], %[[c2]] : i1, i1, i1
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit
    %q2_0 = mqtopt.allocQubit

    %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
    %q1_1, %q0_2 = mqtopt.h() %q1_0 ctrl %q0_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q2_1, %q0_3 = mqtopt.h() %q2_0 ctrl %q0_2 : !mqtopt.Qubit ctrl !mqtopt.Qubit

    %q0_4, %c0 = mqtopt.measure %q0_3
    %q1_2, %c1 = mqtopt.measure %q1_1
    %q2_2, %c2 = mqtopt.measure %q2_1

    mqtopt.deallocQubit %q0_4
    mqtopt.deallocQubit %q1_2
    mqtopt.deallocQubit %q2_2
    return %c0, %c1, %c2 : i1, i1, i1
  }
}
