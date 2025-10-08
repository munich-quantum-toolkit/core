// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --lift-measurements --reuse-qubits | FileCheck %s

// -----
// This test checks that qubit reuse and measurement lifting are compatible in a simple case, where measurements are
// lifted above a single-qubit gate.

module {
  // CHECK-LABEL: func.func @testSingleQubitLift
  func.func @testSingleQubitLift() -> (i1, i1) {
    // CHECK: %[[true:.*]] = arith.constant true
    // CHECK: %[[q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[q0_1:.*]], %[[c0_0:.*]] = mqtopt.measure %[[q0_0]]
    // CHECK: %[[c0_1:.*]] = arith.xori %[[c0_0]], %[[true]] : i1
    // CHECK: %[[q1_0:.*]] = mqtopt.reset %[[q0_1]]
    // CHECK: %[[q1_1:.*]] = mqtopt.h() %[[q1_0]] : !mqtopt.Qubit
    // CHECK: %[[q1_2:.*]], %[[c1:.*]] = mqtopt.measure %[[q1_1]]
    // CHECK: mqtopt.deallocQubit %[[q1_2]]
    // CHECK: return %[[c0_1]], %[[c1]] : i1, i1
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    %q0_1 = mqtopt.x() %q0_0 : !mqtopt.Qubit
    %q1_1 = mqtopt.h() %q1_0 : !mqtopt.Qubit

    %q0_2, %c0 = mqtopt.measure %q0_1
    %q1_2, %c1 = mqtopt.measure %q1_1

    mqtopt.deallocQubit %q0_2
    mqtopt.deallocQubit %q1_2

    return %c0, %c1 : i1, i1
  }
}

// -----
// This test checks that qubit reuse can be applied after lifting measurements over controlled gates
// in a case where the measurement lifting does not influence the applicability of qubit reuse.

module {
  // CHECK-LABEL: func.func @testReuseAfterUnneededControlLift
  func.func @testReuseAfterUnneededControlLift() -> (i1, i1, i1) {
    // CHECK: %[[q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[q0_1:.*]] = mqtopt.h() %[[q0_0]] : !mqtopt.Qubit
    // CHECK: %[[q1_1:.*]], %[[c1:.*]] = mqtopt.measure %[[q1_0]]
    // CHECK: %[[q0_2:.*]] = scf.if %[[c1]] -> (!mqtopt.Qubit) {
    // CHECK:     %[[q0_2_if:.*]] = mqtopt.h() %[[q0_1]] : !mqtopt.Qubit
    // CHECK:     scf.yield %[[q0_2_if]] : !mqtopt.Qubit
    // CHECK: } else {
    // CHECK:     scf.yield %[[q0_1]] : !mqtopt.Qubit
    // CHECK: }
    // CHECK: %[[q2_0:.*]] = mqtopt.reset %[[q1_1]]
    // CHECK: %[[q2_1:.*]], %[[q0_3:.*]] = mqtopt.h() %[[q2_0]] ctrl %[[q0_2]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[q0_4:.*]] = mqtopt.h() %[[q0_3]] : !mqtopt.Qubit
    // CHECK: %[[q0_5:.*]], %[[c0:.*]] = mqtopt.measure %[[q0_4]]
    // CHECK: mqtopt.deallocQubit %[[q0_5]]
    // CHECK: %[[q2_2:.*]], %[[c2:.*]] = mqtopt.measure %[[q2_1]]
    // CHECK: mqtopt.deallocQubit %[[q2_2]]
    // CHECK: return %[[c0]], %[[c1]], %[[c2]] : i1, i1, i1
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit
    %q2_0 = mqtopt.allocQubit

    %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
    %q0_2, %q1_1 = mqtopt.h() %q0_1 ctrl %q1_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q2_1, %q0_3 = mqtopt.h() %q2_0 ctrl %q0_2 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_4 = mqtopt.h() %q0_3 : !mqtopt.Qubit

    %q0_5, %c0 = mqtopt.measure %q0_4
    %q1_2, %c1 = mqtopt.measure %q1_1
    %q2_2, %c2 = mqtopt.measure %q2_1

    mqtopt.deallocQubit %q0_5
    mqtopt.deallocQubit %q1_2
    mqtopt.deallocQubit %q2_2
    return %c0, %c1, %c2 : i1, i1, i1
  }
}

// -----
// This test checks that qubit reuse can be applied with the help of measurement lifting
// over a control.

module {
  // CHECK-LABEL: func.func @testReuseThroughMeasurementLift
  func.func @testReuseThroughMeasurementLift() -> (i1, i1) {
    // CHECK: %[[q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[q0_1:.*]] = mqtopt.h() %[[q0_0]] : !mqtopt.Qubit
    // CHECK: %[[q0_2:.*]], %[[c0:.*]] = mqtopt.measure %[[q0_1]]
    // CHECK: %[[q1_0:.*]] = mqtopt.reset %[[q0_2]]
    // CHECK: %[[q1_1:.*]] = scf.if %[[c0]] -> (!mqtopt.Qubit) {
    // CHECK:     %[[q1_1_then:.*]] = mqtopt.x() %[[q1_0]] : !mqtopt.Qubit
    // CHECK:     scf.yield %[[q1_1_then]] : !mqtopt.Qubit
    // CHECK: } else {
    // CHECK:     scf.yield %[[q1_0]] : !mqtopt.Qubit
    // CHECK: }
    // CHECK: %[[q1_2:.*]], %[[c1:.*]] = mqtopt.measure %[[q1_1]]
    // CHECK: mqtopt.deallocQubit %[[q1_2]]
    // CHECK: return %[[c0]], %[[c1]] : i1, i1
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
    %q1_1, %q0_2 = mqtopt.x() %q1_0 ctrl %q0_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit

    %q0_3, %c0 = mqtopt.measure %q0_2
    %q1_2, %c1 = mqtopt.measure %q1_1

    mqtopt.deallocQubit %q0_3
    mqtopt.deallocQubit %q1_2
    return %c0, %c1 : i1, i1
  }
}

// -----
// This test checks that qubit reuse can be applied with the help of measurement lifting
// over a control and over a single-qubit gate.

module {
  // CHECK-LABEL: func.func @testReuseThroughComplexMeasurementLift
  func.func @testReuseThroughComplexMeasurementLift() -> (i1, i1) {
    // CHECK: %[[q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[q0_1:.*]] = mqtopt.h() %[[q0_0]] : !mqtopt.Qubit
    // CHECK: %[[q0_2:.*]], %[[c0:.*]] = mqtopt.measure %[[q0_1]]
    // CHECK: %[[c0_1:.*]] = arith.xori %[[c0]], %[[true:.*]] : i1
    // CHECK: %[[q1_0:.*]] = mqtopt.reset %[[q0_2]]
    // CHECK: %[[q1_1:.*]] = scf.if %[[c0]] -> (!mqtopt.Qubit) {
    // CHECK:     %[[q1_1_then:.*]] = mqtopt.x() %[[q1_0]] : !mqtopt.Qubit
    // CHECK:     scf.yield %[[q1_1_then]] : !mqtopt.Qubit
    // CHECK: } else {
    // CHECK:     scf.yield %[[q1_0]] : !mqtopt.Qubit
    // CHECK: }
    // CHECK: %[[q1_2:.*]], %[[c1:.*]] = mqtopt.measure %[[q1_1]]
    // CHECK: mqtopt.deallocQubit %[[q1_2]]
    // CHECK: return %[[c0_1]], %[[c1]] : i1, i1
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
    %q1_1, %q0_2 = mqtopt.x() %q1_0 ctrl %q0_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_3 = mqtopt.x() %q0_2 : !mqtopt.Qubit

    %q0_4, %c0 = mqtopt.measure %q0_3
    %q1_2, %c1 = mqtopt.measure %q1_1

    mqtopt.deallocQubit %q0_4
    mqtopt.deallocQubit %q1_2
    return %c0, %c1 : i1, i1
  }
}

// -----
// This test checks that qubit reuse can be applied after lifting measurements over controlled gates if the
// qubit for which reuse should be applied was pulled into an if/else block.

module {
  // CHECK-LABEL: func.func @testReuseOutOfIf
  func.func @testReuseOutOfIf() -> (i1, i1, i1) {
    // CHECK: %[[q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[q1_1:.*]], %[[c1:.*]] = mqtopt.measure %[[q1_0]]
    // CHECK: %[[q0_0:.*]] = mqtopt.reset %[[q1_1]]
    // CHECK: %[[q0_1:.*]] = mqtopt.h() %[[q0_0]] : !mqtopt.Qubit
    // CHECK: %[[q0_2:.*]] = scf.if %[[c1]] -> (!mqtopt.Qubit) {
    // CHECK:     %[[q0_2_if:.*]] = mqtopt.h() %[[q0_1]] : !mqtopt.Qubit
    // CHECK:     scf.yield %[[q0_2_if]] : !mqtopt.Qubit
    // CHECK: } else {
    // CHECK:     scf.yield %[[q0_1]] : !mqtopt.Qubit
    // CHECK: }
    // CHECK: %[[q0_3:.*]], %[[c0:.*]] = mqtopt.measure %[[q0_2]]
    // CHECK: %[[q2_0:.*]] = mqtopt.reset %[[q0_3]]
    // CHECK: %[[q2_1:.*]] = scf.if %[[c0]] -> (!mqtopt.Qubit) {
    // CHECK:     %[[q2_1_if:.*]] = mqtopt.h() %[[q2_0]] : !mqtopt.Qubit
    // CHECK:     scf.yield %[[q2_1_if]] : !mqtopt.Qubit
    // CHECK: } else {
    // CHECK:     scf.yield %[[q2_0]] : !mqtopt.Qubit
    // CHECK: }
    // CHECK: %[[q2_2:.*]], %[[c2:.*]] = mqtopt.measure %[[q2_1]]
    // CHECK: mqtopt.deallocQubit %[[q2_2]]
    // CHECK: return %[[c0]], %[[c1]], %[[c2]] : i1, i1, i1
    %q1_0 = mqtopt.allocQubit
    %q2_0 = mqtopt.allocQubit
    %q0_0 = mqtopt.allocQubit

    %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
    %q0_2, %q1_1 = mqtopt.h() %q0_1 ctrl %q1_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit
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
