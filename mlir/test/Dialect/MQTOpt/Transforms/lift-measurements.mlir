// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --lift-measurements | FileCheck %s

// =================================================== Lift over Controls =========================================================================
// -----
// This test checks that measurements can be lifted over controlled gates if the measurement target is a positive control and the only control of the gate.
// In this case, the controlled gate is replaced by a conditional gate that uses the measurement outcome as a condition.

module {
  // CHECK-LABEL: func.func @testLiftOverPositiveControl
  func.func @testLiftOverPositiveControl() -> (i1, i1) {
    // CHECK: %[[Q0_1:.*]], %[[Q1_1:.*]] = mqtopt.x() %[[ANY:.*]] ctrl %[[ANY:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]], %[[C0:.*]] = "mqtopt.measure"(%[[Q0_1:.*]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    // CHECK: %[[Q1_2:.*]] = scf.if %[[C0]] -> (!mqtopt.Qubit) {
    // CHECK-NEXT: %[[Q1_2IF:.*]] = mqtopt.h() %[[Q1_1]] : !mqtopt.Qubit
    // CHECK-NEXT: scf.yield %[[Q1_2IF]]
    // CHECK-NEXT: } else {
    // CHECK-NEXT: scf.yield %[[Q1_1]]
    // CHECK-NEXT: }
    // CHECK: %[[Q1_3:.*]] = scf.if %[[C0]] -> (!mqtopt.Qubit) {
    // CHECK-NEXT: %[[Q1_3IF:.*]] = mqtopt.x() %[[Q1_2]] : !mqtopt.Qubit
    // CHECK-NEXT: scf.yield %[[Q1_3IF]]
    // CHECK-NEXT: } else {
    // CHECK-NEXT: scf.yield %[[Q1_2]]
    // CHECK-NEXT: }
    // CHECK: mqtopt.deallocQubit %[[Q0_2]]

    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    %q0_1, %q1_1 = mqtopt.x() %q0_0 ctrl %q1_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q1_2, %q0_2 = mqtopt.h() %q1_1 ctrl %q0_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q1_3, %q0_3 = mqtopt.x() %q1_2 ctrl %q0_2 : !mqtopt.Qubit ctrl !mqtopt.Qubit

    %q0_4, %c0 = "mqtopt.measure"(%q0_3) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    %q1_4, %c1 = "mqtopt.measure"(%q1_3) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

    mqtopt.deallocQubit %q0_4
    mqtopt.deallocQubit %q1_4
    return %c0, %c1 : i1, i1
  }
}

// -----
// This test checks that measurements can be lifted over controlled gates if the measurement target is a negative control and the only control of the gate.
// In this case, the controlled gate is replaced by a conditional gate that uses the negated measurement outcome as a condition.

module {
  // CHECK-LABEL: func.func @testLiftOverNegativeControl
  func.func @testLiftOverNegativeControl() -> (i1, i1) {
    // CHECK: %[[Q0_1:.*]], %[[Q1_1:.*]] = mqtopt.x() %[[ANY:.*]] nctrl %[[ANY:.*]] : !mqtopt.Qubit nctrl !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]], %[[C0:.*]] = "mqtopt.measure"(%[[Q0_1]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    // CHECK: %[[Q1_2:.*]] = scf.if %[[C0]] -> (!mqtopt.Qubit) {
    // CHECK-NEXT: scf.yield %[[Q1_1]]
    // CHECK-NEXT: } else {
    // CHECK-NEXT: %[[Q1_2IF:.*]] = mqtopt.h() %[[Q1_1]] : !mqtopt.Qubit
    // CHECK-NEXT: scf.yield %[[Q1_2IF]]
    // CHECK-NEXT: }
    // CHECK: %[[Q1_3:.*]] = scf.if %[[C0]] -> (!mqtopt.Qubit) {
    // CHECK-NEXT: scf.yield %[[Q1_2]]
    // CHECK-NEXT: } else {
    // CHECK-NEXT: %[[Q1_3IF:.*]] = mqtopt.x() %[[Q1_2]] : !mqtopt.Qubit
    // CHECK-NEXT: scf.yield %[[Q1_3IF]]
    // CHECK-NEXT: }
    // CHECK: mqtopt.deallocQubit %[[Q0_2]]

    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    %q0_1, %q1_1 = mqtopt.x() %q0_0 nctrl %q1_0 : !mqtopt.Qubit nctrl !mqtopt.Qubit
    %q1_2, %q0_2 = mqtopt.h() %q1_1 nctrl %q0_1 : !mqtopt.Qubit nctrl !mqtopt.Qubit
    %q1_3, %q0_3 = mqtopt.x() %q1_2 nctrl %q0_2 : !mqtopt.Qubit nctrl !mqtopt.Qubit

    %q0_4, %c0 = "mqtopt.measure"(%q0_3) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    %q1_4, %c1 = "mqtopt.measure"(%q1_3) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

    mqtopt.deallocQubit %q0_4
    mqtopt.deallocQubit %q1_4
    return %c0, %c1 : i1, i1
  }
}

// -----
// This test checks that measurements can be lifted over controlled gates if the measurement target is one of multiple controls of the gate.
// In this case, the controlled gate is replaced by a conditional gate that uses the measurement outcome as a condition and still uses the remaining controls.

module {
  // CHECK-LABEL: func.func @testLiftOverOneOfMultipleControls
  func.func @testLiftOverOneOfMultipleControls() -> (i1) {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q2_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_1:.*]], %[[C1:.*]] = "mqtopt.measure"(%[[Q1_0]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    // CHECK: %[[Q02_1:.*]]:2 = scf.if %[[C1]] ->
    // CHECK-NEXT: %[[Q0_1IF:.*]], %[[Q2_1IF:.*]] = mqtopt.x() %[[Q0_0]] ctrl %[[Q2_0]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK-NEXT: scf.yield %[[Q0_1IF:.*]], %[[Q2_1IF:.*]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK-NEXT: } else {
    // CHECK-NEXT: scf.yield %[[Q0_0]], %[[Q2_0]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK-NEXT: }
    // CHECK: %[[Q02_2:.*]]:2 = scf.if %[[C1]] -> (!mqtopt.Qubit, !mqtopt.Qubit) {
    // CHECK-NEXT: scf.yield %[[Q02_1]]#0, %[[Q02_1]]#1
    // CHECK-NEXT: } else {
    // CHECK-NEXT: %[[Q0_2IF:.*]], %[[Q2_2IF:.*]] = mqtopt.x() %[[Q02_1]]#0 nctrl %[[Q02_1]]#1 : !mqtopt.Qubit nctrl !mqtopt.Qubit
    // CHECK-NEXT: scf.yield %[[Q0_2IF]], %[[Q2_2IF]]
    // CHECK-NEXT: }
    // CHECK: %[[Q02_3:.*]]:2 = scf.if %[[C1]] -> (!mqtopt.Qubit, !mqtopt.Qubit) {
    // CHECK-NEXT: %[[Q0_3IF:.*]], %[[Q2_3IF:.*]] = mqtopt.x() %[[Q02_2]]#0 nctrl %[[Q02_2]]#1 : !mqtopt.Qubit nctrl !mqtopt.Qubit
    // CHECK-NEXT: scf.yield %[[Q0_3IF]], %[[Q2_3IF]]
    // CHECK-NEXT: } else {
    // CHECK-NEXT: scf.yield %[[Q02_2]]#0, %[[Q02_2]]#1
    // CHECK-NEXT: }
    // CHECK: mqtopt.deallocQubit %[[Q1_1]]

    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit
    %q2_0 = mqtopt.allocQubit

    %q0_1, %q12_0:2 = mqtopt.x() %q0_0 ctrl %q1_0, %q2_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit
    %q0_2, %q12_1:2 = mqtopt.x() %q0_1 nctrl %q12_0#1, %q12_0#0 : !mqtopt.Qubit nctrl !mqtopt.Qubit, !mqtopt.Qubit
    %q0_3, %q1_3, %q2_3 = mqtopt.x() %q0_2 ctrl %q12_1#1 nctrl %q12_1#0 : !mqtopt.Qubit ctrl !mqtopt.Qubit nctrl !mqtopt.Qubit

    %q1_4, %c1 = "mqtopt.measure"(%q1_3) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

    %q0_4 = mqtopt.h() %q0_3 : !mqtopt.Qubit
    %q2_4 = mqtopt.h() %q2_3 : !mqtopt.Qubit
    %q0_5, %c0 = "mqtopt.measure"(%q0_4) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    %q2_5, %c2 = "mqtopt.measure"(%q2_4) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

    mqtopt.deallocQubit %q0_5
    mqtopt.deallocQubit %q1_4
    mqtopt.deallocQubit %q2_5
    return %c1 : i1
  }
}

// -----
// This test checks that several measurements can be lifted over the same controlled gate if the measurement targets are controls of the gate.
// In this case, the controlled gate is replaced by a conditional gate that uses the measurement outcomes as conditions.

module {
  // CHECK-LABEL: func.func @testLiftMultipleMeasuresOverControlledGate
  func.func @testLiftMultipleMeasuresOverControlledGate() -> (i1, i1) {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q2_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q2_1:.*]], %[[C2:.*]] = "mqtopt.measure"(%[[Q2_0]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    // CHECK: %[[Q1_1:.*]], %[[C1:.*]] = "mqtopt.measure"(%[[Q1_0]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    // CHECK: %[[Q02_1:.*]]:2 = scf.if %[[C1]] -> (!mqtopt.Qubit, !mqtopt.Qubit) {
      // CHECK-NEXT: %[[Q0_1IF:.*]] = scf.if %[[C2]] -> (!mqtopt.Qubit) {
        // CHECK-NEXT: %[[Q0_1IFIF:.*]] = mqtopt.x() %[[ANY:.*]] : !mqtopt.Qubit
        // CHECK-NEXT: scf.yield %[[Q0_1IFIF]]
      // CHECK-NEXT: } else {
        // CHECK-NEXT: scf.yield %[[ANY:.*]]
      // CHECK-NEXT: }
      // CHECK-NEXT: scf.yield %[[Q0_1IF]], %[[Q2_1]]
    // CHECK-NEXT: } else {
      // CHECK-NEXT: scf.yield %[[ANY:.*]], %[[Q2_1]]
    // CHECK-NEXT: }
    // CHECK: mqtopt.deallocQubit %[[Q02_1]]#0
    // CHECK: mqtopt.deallocQubit %[[Q1_1]]
    // CHECK: mqtopt.deallocQubit %[[Q02_1]]#1

    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit
    %q2_0 = mqtopt.allocQubit

    %q0_1, %q12_0:2 = mqtopt.x() %q0_0 ctrl %q1_0, %q2_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit

    %q1_1, %c1 = "mqtopt.measure"(%q12_0#0) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    %q2_1, %c2 = "mqtopt.measure"(%q12_0#1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

    mqtopt.deallocQubit %q0_1
    mqtopt.deallocQubit %q1_1
    mqtopt.deallocQubit %q2_1
    return %c1, %c2 : i1, i1
  }
}

// -----
// This test checks that measurements can be lifted over controlled parametrized gates and the resulting classically controlled parametrized gates
// still have the correct parameters.

module {
  // CHECK-LABEL: func.func @testLiftControlledParametrizedGate
  func.func @testLiftControlledParametrizedGate() -> (i1, i1) {
    // CHECK: %[[Q1_1:.*]], %[[C1:.*]] = "mqtopt.measure"(%[[Q1_0:.*]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    // CHECK: %[[Q0_1:.*]] = scf.if %[[C1]] -> (!mqtopt.Qubit) {
    // CHECK-NEXT: %[[Q0_1IF:.*]] = mqtopt.rx(%[[ANGLE:.*]]) %[[Q0_0:.*]] : !mqtopt.Qubit
    // CHECK-NEXT: scf.yield %[[Q0_1IF]]
    // CHECK-NEXT: } else {
    // CHECK-NEXT: scf.yield %[[Q0_0]]
    // CHECK-NEXT: }

    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    %angle = arith.constant 3.000000e-01 : f64

    %q0_1, %q1_1 = mqtopt.rx(%angle) %q0_0 ctrl %q1_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit

    %q0_2, %c0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    %q1_2, %c1 = "mqtopt.measure"(%q1_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

    mqtopt.deallocQubit %q0_2
    mqtopt.deallocQubit %q1_2
    return %c0, %c1 : i1, i1
  }
}

// =================================================== Lift over 1Q Gates =========================================================================

// -----
// This test checks that measurements can be lifted over a single X Gate by adding a classical negation afterwards.

module {
  // CHECK-LABEL: func.func @testLiftOverSingleX
  func.func @testLiftOverSingleX() -> i1 {
    // CHECK: %[[Q0_1:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[ANY:.*]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    // CHECK-NOT: mqtopt.x
    // CHECK: %[[C0_1:.*]] = arith.xori %[[C0_0]], %[[TRUE:.*]] : i1
    // CHECK: mqtopt.deallocQubit %[[Q0_1]]
    // CHECK: return %[[C0_1]]

    %q0_0 = mqtopt.allocQubit

    %q0_1 = mqtopt.x() %q0_0 : !mqtopt.Qubit

    %q0_2, %c0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

    mqtopt.deallocQubit %q0_2
    return %c0 : i1
  }
}

// -----
// This test checks that measurements can be lifted over a single Y Gate by adding a classical negation afterwards.
// Essentially, a Y Gate before a measurement can be treated just like an X Gate

module {
  // CHECK-LABEL: func.func @testLiftOverSingleY
  func.func @testLiftOverSingleY() -> i1 {
    // CHECK: %[[Q0_1:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[ANY:.*]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    // CHECK-NOT: mqtopt.y
    // CHECK: %[[C0_1:.*]] = arith.xori %[[C0_0]], %[[TRUE:.*]] : i1
    // CHECK: mqtopt.deallocQubit %[[Q0_1]]
    // CHECK: return %[[C0_1]]

    %q0_0 = mqtopt.allocQubit

    %q0_1 = mqtopt.y() %q0_0 : !mqtopt.Qubit

    %q0_2, %c0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

    mqtopt.deallocQubit %q0_2
    return %c0 : i1
  }
}

// -----
// This test checks that measurements can be lifted over local phase gates (Z, etc.) without further classical operations.

module {
  // CHECK-LABEL: func.func @testLiftOverZ
  func.func @testLiftOverZ() -> i1 {
    // CHECK: %[[Q0_1:.*]], %[[C0:.*]] = "mqtopt.measure"(%[[ANY:.*]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    // CHECK: mqtopt.deallocQubit %[[Q0_1]]
    // CHECK-NOT: mqtopt.z
    // CHECK: return %[[C0]]

    %q0_0 = mqtopt.allocQubit
    %angle = arith.constant 3.000000e-01 : f64

    %q0_1 = mqtopt.i() %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.z() %q0_1 : !mqtopt.Qubit
    %q0_3 = mqtopt.s() %q0_2 : !mqtopt.Qubit
    %q0_4 = mqtopt.sdg() %q0_3 : !mqtopt.Qubit
    %q0_5 = mqtopt.t() %q0_4 : !mqtopt.Qubit
    %q0_6 = mqtopt.tdg() %q0_5 : !mqtopt.Qubit
    %q0_7 = mqtopt.p(%angle) %q0_6 : !mqtopt.Qubit
    %q0_8 = mqtopt.rz(%angle) %q0_7 : !mqtopt.Qubit

    %q0_9, %c0 = "mqtopt.measure"(%q0_8) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

    mqtopt.deallocQubit %q0_9
    return %c0 : i1
  }
}

// -----
// This test checks that measurements can be lifted over multiple X Gates by adding classical negations afterwards.

module {
  // CHECK-LABEL: func.func @testLiftOverMultipleX
  func.func @testLiftOverMultipleX() -> i1 {
    // CHECK: %[[Q0_1:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[ANY:.*]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    // CHECK: mqtopt.deallocQubit %[[Q0_1]]
    // CHECK-NOT: mqtopt.x
    // CHECK: return %[[C0_0]]

    %q0_0 = mqtopt.allocQubit

    %q0_1 = mqtopt.x() %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.x() %q0_1 : !mqtopt.Qubit

    %q0_3, %c0 = "mqtopt.measure"(%q0_2) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

    mqtopt.deallocQubit %q0_3
    return %c0 : i1
  }
}

// -----
// This test checks that measurements can be lifted over X Gates as well as controlled gates where the measurement target is a control at the same time.

module {
  // CHECK-LABEL: func.func @testLiftOverXAndControlledGates
  func.func @testLiftOverXAndControlledGates() -> i1 {
    // CHECK: %[[Q0_1:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[ANY:.*]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    // CHECK: %[[Q1_1:.*]] = scf.if %[[C0_0]] -> (!mqtopt.Qubit) {
    // CHECK-NEXT: %[[Q1_1IF:.*]] = mqtopt.y() %[[ANY:.*]] : !mqtopt.Qubit
    // CHECK-NEXT: scf.yield %[[Q1_1IF]]
    // CHECK-NEXT: } else {
    // CHECK-NEXT: scf.yield %[[ANY:.*]]
    // CHECK-NEXT: }
    // CHECK: %[[Q1_2:.*]] = scf.if %[[C0_0]] -> (!mqtopt.Qubit) {
    // CHECK-NEXT: scf.yield %[[Q1_1]]
    // CHECK-NEXT: } else {
    // CHECK-NEXT: %[[Q1_2IF:.*]] = mqtopt.y() %[[Q1_1]] : !mqtopt.Qubit
    // CHECK-NEXT: scf.yield %[[Q1_2IF]]
    // CHECK-NEXT: }
    // CHECK-NOT: mqtopt.x
    // CHECK: mqtopt.deallocQubit %[[Q0_1]]
    // CHECK: mqtopt.deallocQubit %[[Q1_2]]
    // CHECK: return %[[C0_0]]

    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit


    %q1_1, %q0_1 = mqtopt.y() %q1_0 ctrl %q0_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_2 = mqtopt.x() %q0_1 : !mqtopt.Qubit
    %q1_2, %q0_3 = mqtopt.y() %q1_1 ctrl %q0_2 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_4 = mqtopt.x() %q0_3 : !mqtopt.Qubit

    %q0_5, %c0 = "mqtopt.measure"(%q0_4) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

    mqtopt.deallocQubit %q0_5
    mqtopt.deallocQubit %q1_2
    return %c0 : i1
  }
}

// -----
// This test checks that measurements can be lifted over
// the targets of controlled gates if target and control are intercheangable.

module {
  // CHECK-LABEL: func.func @testLiftOverTargetAsControlInDiagonalGate
  func.func @testLiftOverTargetAsControlInDiagonalGate() -> i1 {
    // CHECK: %[[q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[q0_1:.*]], %[[c0:.*]] = "mqtopt.measure"(%[[q0_0]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    // CHECK: %[[q1_1:.*]] = scf.if %[[c0]] -> (!mqtopt.Qubit) {
    // CHECK:   %[[q1_1_then:.*]] = mqtopt.z() %[[q1_0]] : !mqtopt.Qubit
    // CHECK:   scf.yield %[[q1_1_then]] : !mqtopt.Qubit
    // CHECK: } else {
    // CHECK:   scf.yield %[[q1_0]] : !mqtopt.Qubit
    // CHECK: }
    // CHECK: mqtopt.deallocQubit %[[q1_1]]
    // CHECK: mqtopt.deallocQubit %[[q0_1]]
    // CHECK: return %[[c0]] : i1
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    %q0_1, %q1_1 = mqtopt.z() %q0_0 ctrl %q1_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_2, %c0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

    mqtopt.deallocQubit %q1_1
    mqtopt.deallocQubit %q0_1
    return %c0 : i1
  }
}

// -----
// This test checks that measurements can be lifted over
// the targets of controlled gates if target and negative control are intercheangable.

module {
  // CHECK-LABEL: func.func @testLiftOverTargetAsNegativeControlInDiagonalGate
  func.func @testLiftOverTargetAsNegativeControlInDiagonalGate() -> i1 {
    // CHECK: %[[q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[q0_1:.*]], %[[c0:.*]] = "mqtopt.measure"(%[[q0_0]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    // CHECK: %[[q1_1:.*]] = mqtopt.x() %[[q1_0]] : !mqtopt.Qubit
    // CHECK: %[[q1_2:.*]] = scf.if %[[c0]] -> (!mqtopt.Qubit) {
    // CHECK:   %[[q1_2_then:.*]] = mqtopt.z() %[[q1_1]] : !mqtopt.Qubit
    // CHECK:   scf.yield %[[q1_2_then]] : !mqtopt.Qubit
    // CHECK: } else {
    // CHECK:   scf.yield %[[q1_1]] : !mqtopt.Qubit
    // CHECK: }
    // CHECK: mqtopt.deallocQubit %[[q1_2]]
    // CHECK: mqtopt.deallocQubit %[[q0_1]]
    // CHECK: return %[[c0]] : i1
    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    %q0_1, %q1_1 = mqtopt.z() %q0_0 nctrl %q1_0 : !mqtopt.Qubit nctrl !mqtopt.Qubit
    %q0_2, %c0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

    mqtopt.deallocQubit %q1_1
    mqtopt.deallocQubit %q0_1
    return %c0 : i1
  }
}
