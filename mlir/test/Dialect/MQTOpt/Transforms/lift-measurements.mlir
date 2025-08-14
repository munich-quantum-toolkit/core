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
  func.func @testLiftOverPositiveControl() -> i1 {
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
    // CHECK: %[[ANY:.*]] = "mqtopt.insertQubit"(%[[Reg_2:.*]], %[[Q0_2]]) <{index_attr = 0 : i64}>

    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    %q0_1, %q1_1 = mqtopt.x() %q0_0 ctrl %q1_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q1_2, %q0_2 = mqtopt.h() %q1_1 ctrl %q0_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q1_3, %q0_3 = mqtopt.x() %q1_2 ctrl %q0_2 : !mqtopt.Qubit ctrl !mqtopt.Qubit

    %q0_4, %c0 = "mqtopt.measure"(%q0_3) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

    %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_4) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_3) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    "mqtopt.deallocQubitRegister"(%reg_4) : (!mqtopt.QubitRegister) -> ()
    return %c0 : i1
  }
}

// -----
// This test checks that measurements can be lifted over controlled gates if the measurement target is a negative control and the only control of the gate.
// In this case, the controlled gate is replaced by a conditional gate that uses the negated measurement outcome as a condition.

module {
  // CHECK-LABEL: func.func @testLiftOverNegativeControl
  func.func @testLiftOverNegativeControl() -> i1 {
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
    // CHECK: %[[ANY:.*]] = "mqtopt.insertQubit"(%[[Reg_2:.*]], %[[Q0_2]]) <{index_attr = 0 : i64}>

    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    %q0_1, %q1_1 = mqtopt.x() %q0_0 nctrl %q1_0 : !mqtopt.Qubit nctrl !mqtopt.Qubit
    %q1_2, %q0_2 = mqtopt.h() %q1_1 nctrl %q0_1 : !mqtopt.Qubit nctrl !mqtopt.Qubit
    %q1_3, %q0_3 = mqtopt.x() %q1_2 nctrl %q0_2 : !mqtopt.Qubit nctrl !mqtopt.Qubit

    %q0_4, %c0 = "mqtopt.measure"(%q0_3) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

    %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_4) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_3) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    "mqtopt.deallocQubitRegister"(%reg_4) : (!mqtopt.QubitRegister) -> ()
    return %c0 : i1
  }
}

// -----
// This test checks that measurements can be lifted over controlled gates if the measurement target is one of multiple controls of the gate.
// In this case, the controlled gate is replaced by a conditional gate that uses the measurement outcome as a condition and still uses the remaining controls.

module {
  // CHECK-LABEL: func.func @testLiftOverOneOfMultipleControls
  func.func @testLiftOverOneOfMultipleControls() -> i1 {
    // CHECK: %[[Q1_1:.*]], %[[C1:.*]] = "mqtopt.measure"(%[[Q1_0:.*]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    // CHECK: %[[Q02_1:.*]]:2 = scf.if %[[C1]] ->
    // CHECK-NEXT: %[[Q0_1IF:.*]], %[[Q2_1IF:.*]] = mqtopt.x() %[[Q0_0:.*]] ctrl %[[Q2_0:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
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
    // CHECK: %[[ANY:.*]] = "mqtopt.insertQubit"(%[[Reg_2:.*]], %[[Q02_3]]#0) <{index_attr = 0 : i64}>
    // CHECK: %[[ANY:.*]] = "mqtopt.insertQubit"(%[[Reg_2:.*]], %[[Q1_1]]) <{index_attr = 1 : i64}>
    // CHECK: %[[ANY:.*]] = "mqtopt.insertQubit"(%[[Reg_2:.*]], %[[Q02_3]]#1) <{index_attr = 2 : i64}>

    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister

    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    %reg_3, %q2_0 = "mqtopt.extractQubit"(%reg_2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    %q0_1, %q12_0:2 = mqtopt.x() %q0_0 ctrl %q1_0, %q2_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit
    %q0_2, %q12_1:2 = mqtopt.x() %q0_1 nctrl %q12_0#1, %q12_0#0 : !mqtopt.Qubit nctrl !mqtopt.Qubit, !mqtopt.Qubit
    %q0_3, %q1_3, %q2_3 = mqtopt.x() %q0_2 ctrl %q12_1#1 nctrl %q12_1#0 : !mqtopt.Qubit ctrl !mqtopt.Qubit nctrl !mqtopt.Qubit

    %q1_4, %c0 = "mqtopt.measure"(%q1_3) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

    %reg_4 = "mqtopt.insertQubit"(%reg_3, %q0_3) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    %reg_5 = "mqtopt.insertQubit"(%reg_4, %q1_4) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    %reg_6 = "mqtopt.insertQubit"(%reg_5, %q2_3) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    "mqtopt.deallocQubitRegister"(%reg_6) : (!mqtopt.QubitRegister) -> ()
    return %c0 : i1
  }
}

// -----
// This test checks that several measurements can be lifted over the same controlled gate if the measurement targets are controls of the gate.
// In this case, the controlled gate is replaced by a conditional gate that uses the measurement outcomes as conditions.

module {
  // CHECK-LABEL: func.func @testLiftMultipleMeasuresOverControlledGate
  func.func @testLiftMultipleMeasuresOverControlledGate() -> (i1, i1) {
    // CHECK: %[[Reg_ANY:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[REG_ANY:.*]]) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[Reg_ANY:.*]], %[[Q2_0:.*]] = "mqtopt.extractQubit"(%[[REG_ANY:.*]]) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
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
    // CHECK: %[[ANY:.*]] = "mqtopt.insertQubit"(%[[Reg_2:.*]], %[[Q02_1]]#0) <{index_attr = 0 : i64}>
    // CHECK: %[[ANY:.*]] = "mqtopt.insertQubit"(%[[Reg_3:.*]], %[[Q1_1]]) <{index_attr = 1 : i64}>
    // CHECK: %[[ANY:.*]] = "mqtopt.insertQubit"(%[[Reg_4:.*]], %[[Q02_1]]#1) <{index_attr = 2 : i64}>

    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister

    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    %reg_3, %q2_0 = "mqtopt.extractQubit"(%reg_2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    %q0_1, %q12_0:2 = mqtopt.x() %q0_0 ctrl %q1_0, %q2_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit

    %q1_1, %c1 = "mqtopt.measure"(%q12_0#0) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    %q2_1, %c2 = "mqtopt.measure"(%q12_0#1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

    %reg_4 = "mqtopt.insertQubit"(%reg_3, %q0_1) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    %reg_5 = "mqtopt.insertQubit"(%reg_4, %q1_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    %reg_6 = "mqtopt.insertQubit"(%reg_5, %q2_1) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    "mqtopt.deallocQubitRegister"(%reg_6) : (!mqtopt.QubitRegister) -> ()
    return %c1, %c2 : i1, i1
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
    // CHECK: %[[ANY:.*]] = "mqtopt.insertQubit"(%[[Reg_2:.*]], %[[Q0_1]]) <{index_attr = 0 : i64}>
    // CHECK: return %[[C0_1]]

    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister

    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    %q0_1 = mqtopt.x() %q0_0 : !mqtopt.Qubit

    %q0_2, %c0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

    %reg_2 = "mqtopt.insertQubit"(%reg_1, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    "mqtopt.deallocQubitRegister"(%reg_2) : (!mqtopt.QubitRegister) -> ()
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
    // CHECK: %[[ANY:.*]] = "mqtopt.insertQubit"(%[[Reg_2:.*]], %[[Q0_1]]) <{index_attr = 0 : i64}>
    // CHECK: return %[[C0_1]]

    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister

    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    %q0_1 = mqtopt.y() %q0_0 : !mqtopt.Qubit

    %q0_2, %c0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

    %reg_2 = "mqtopt.insertQubit"(%reg_1, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    "mqtopt.deallocQubitRegister"(%reg_2) : (!mqtopt.QubitRegister) -> ()
    return %c0 : i1
  }
}

// -----
// This test checks that measurements can be lifted over local phase gates (Z, etc.) without further classical operations.

module {
  // CHECK-LABEL: func.func @testLiftOverZ
  func.func @testLiftOverZ() -> i1 {
    // CHECK: %[[Q0_1:.*]], %[[C0:.*]] = "mqtopt.measure"(%[[ANY:.*]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    // CHECK: %[[ANY:.*]] = "mqtopt.insertQubit"(%[[Reg_2:.*]], %[[Q0_1]]) <{index_attr = 0 : i64}>
    // CHECK-NOT: mqtopt.z
    // CHECK: return %[[C0]]

    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
    %angle = arith.constant 3.000000e-01 : f64

    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    %q0_1 = mqtopt.i() %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.z() %q0_1 : !mqtopt.Qubit
    %q0_3 = mqtopt.s() %q0_2 : !mqtopt.Qubit
    %q0_4 = mqtopt.sdg() %q0_3 : !mqtopt.Qubit
    %q0_5 = mqtopt.t() %q0_4 : !mqtopt.Qubit
    %q0_6 = mqtopt.tdg() %q0_5 : !mqtopt.Qubit
    %q0_7 = mqtopt.p(%angle) %q0_6 : !mqtopt.Qubit
    %q0_8 = mqtopt.rz(%angle) %q0_7 : !mqtopt.Qubit

    %q0_9, %c0 = "mqtopt.measure"(%q0_8) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

    %reg_2 = "mqtopt.insertQubit"(%reg_1, %q0_9) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    "mqtopt.deallocQubitRegister"(%reg_2) : (!mqtopt.QubitRegister) -> ()
    return %c0 : i1
  }
}

// -----
// This test checks that measurements can be lifted over multiple X Gates by adding classical negations afterwards.

module {
  // CHECK-LABEL: func.func @testLiftOverMultipleX
  func.func @testLiftOverMultipleX() -> i1 {
    // CHECK: %[[Q0_1:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[ANY:.*]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    // CHECK: %[[ANY:.*]] = "mqtopt.insertQubit"(%[[Reg_2:.*]], %[[Q0_1]]) <{index_attr = 0 : i64}>
    // CHECK-NOT: mqtopt.x
    // CHECK: return %[[C0_0]]

    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister

    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    %q0_1 = mqtopt.x() %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.x() %q0_1 : !mqtopt.Qubit

    %q0_3, %c0 = "mqtopt.measure"(%q0_2) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

    %reg_2 = "mqtopt.insertQubit"(%reg_1, %q0_3) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    "mqtopt.deallocQubitRegister"(%reg_2) : (!mqtopt.QubitRegister) -> ()
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
    // CHECK: %[[ANY:.*]] = "mqtopt.insertQubit"(%[[ANY:.*]], %[[Q0_1]]) <{index_attr = 0 : i64}>
    // CHECK: %[[ANY:.*]] = "mqtopt.insertQubit"(%[[ANY:.*]], %[[Q1_2]]) <{index_attr = 1 : i64}>
    // CHECK: return %[[C0_0]]

    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    %q1_1, %q0_1 = mqtopt.y() %q1_0 ctrl %q0_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_2 = mqtopt.x() %q0_1 : !mqtopt.Qubit
    %q1_2, %q0_3 = mqtopt.y() %q1_1 ctrl %q0_2 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_4 = mqtopt.x() %q0_3 : !mqtopt.Qubit

    %q0_5, %c0 = "mqtopt.measure"(%q0_4) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

    %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_5) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_2) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    "mqtopt.deallocQubitRegister"(%reg_4) : (!mqtopt.QubitRegister) -> ()
    return %c0 : i1
  }
}
