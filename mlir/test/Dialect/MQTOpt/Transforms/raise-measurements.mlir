// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --raise-measurements | FileCheck %s

// =================================================== Raise over Controls =========================================================================
// -----
// This test checks that measurements can be raised over controlled gates if the measurement target is a positive control and the only control of the gate.
// In this case, the controlled gate is replaced by a conditional gate that uses the measurement outcome as a condition.

module {
  // CHECK-LABEL: func.func @testRaiseOverPositiveControl
  func.func @testRaiseOverPositiveControl() -> i1 {
    // CHECK: %[[Q0_1:.*]], %[[Q1_1:.*]] = mqtopt.x() %[[ANY:.*]] ctrl %[[ANY:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]], %[[C0]] = "mqtopt.measure"(%[[Q0_1:.*]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    // CHECK: %[[Q1_2:.*]] = mqtopt.h() %[[Q1_1]] if %[[C0]] : !mqtopt.Qubit if i1
    // CHECK: %[[Q1_3:.*]] = mqtopt.x() %[[Q1_2]] if %[[C0]] : !mqtopt.Qubit if i1
    // CHECK: %[[ANY:.*]] = "mqtopt.insertQubit"(%[[Reg_2:.*]], %[[Q0_2]]) <{index_attr = 0 : i64}>

    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    %q0_1, %q1_1 = mqtopt.x() %q0_0 ctrl %q1_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q1_2, %q0_2 = mqtopt.h() %q1_1 ctrl %q0_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q1_3, %q0_3 = mqtopt.x() %q1_2 ctrl %q1_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit

    %q0_4, %c0 = "mqtopt.measure"(%q0_3) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

    %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_4) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_3) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    "mqtopt.deallocQubitRegister"(%reg_4) : (!mqtopt.QubitRegister) -> ()
    return %c0 : i1
  }
}

// -----
// This test checks that measurements can be raised over controlled gates if the measurement target is a negative control and the only control of the gate.
// In this case, the controlled gate is replaced by a conditional gate that uses the negated measurement outcome as a condition.

module {
  // CHECK-LABEL: func.func @testRaiseOverNegativeControl
  func.func @testRaiseOverNegativeControl() -> i1 {
    // CHECK: %[[Q0_1:.*]], %[[Q1_1:.*]] = mqtopt.x() %[[ANY:.*]] nctrl %[[ANY:.*]] : !mqtopt.Qubit nctrl !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]], %[[C0_0]] = "mqtopt.measure"(%[[Q0_1:.*]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    // CHECK: %[[C0_1]] = arith.xori %[[C0_0]], %[[TRUE:.*]] : i1
    // CHECK: %[[Q1_2:.*]] = mqtopt.h() %[[Q1_1]] if %[[C0_1]] : !mqtopt.Qubit if i1
    // CHECK: %[[Q1_3:.*]] = mqtopt.x() %[[Q1_2]] if %[[C0_1]] : !mqtopt.Qubit if i1
    // CHECK: %[[ANY:.*]] = "mqtopt.insertQubit"(%[[Reg_2:.*]], %[[Q0_2]]) <{index_attr = 0 : i64}>

    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    %q0_1, %q1_1 = mqtopt.x() %q0_0 nctrl %q1_0 : !mqtopt.Qubit nctrl !mqtopt.Qubit
    %q1_2, %q0_2 = mqtopt.h() %q1_1 nctrl %q0_1 : !mqtopt.Qubit nctrl !mqtopt.Qubit
    %q1_3, %q0_3 = mqtopt.x() %q1_2 nctrl %q1_1 : !mqtopt.Qubit nctrl !mqtopt.Qubit

    %q0_4, %c0 = "mqtopt.measure"(%q0_3) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

    %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_4) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_3) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    "mqtopt.deallocQubitRegister"(%reg_4) : (!mqtopt.QubitRegister) -> ()
    return %c0 : i1
  }
}

// -----
// This test checks that measurements can be raised over controlled gates if the measurement target is one of multiple controls of the gate.
// In this case, the controlled gate is replaced by a conditional gate that uses the measurement outcome as a condition and still uses the remaining controls.

module {
  // CHECK-LABEL: func.func @testRaiseOverOneOfMultipleControls
  func.func @testRaiseOverOneOfMultipleControls() -> i1 {
    // CHECK: %[[Q1_1:.*]], %[[C1_0]] = "mqtopt.measure"(%[[Q1_0:.*]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    // CHECK: %[[Q0_1:.*]], %[[Q2_1:.*]] = mqtopt.x() %[[ANY:.*]] ctrl %[[ANY:.*]] if %[[C1_0]]: !mqtopt.Qubit ctrl !mqtopt.Qubit if i1
    // CHECK: %[[C1_1]] = arith.xori %[[C1_0]], %[[TRUE:.*]] : i1
    // CHECK: %[[Q0_2:.*]], %[[Q2_2:.*]] = mqtopt.x() %[[Q0_1]] nctrl %[[Q2_1]] if %[[C1_1]] : !mqtopt.Qubit nctrl !mqtopt.Qubit if i1
    // CHECK: %[[Q0_3:.*]], %[[Q2_3:.*]] = mqtopt.x() %[[Q0_2]] nctrl %[[Q2_2]] if %[[C0_1]] : !mqtopt.Qubit nctrl !mqtopt.Qubit if i1
    // CHECK: %[[ANY:.*]] = "mqtopt.insertQubit"(%[[Reg_2:.*]], %[[Q1_1]]) <{index_attr = 1 : i64}>

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
// This test checks that several measurements can be raised over the same controlled gate if the measurement targets are controls of the gate.
// In this case, the controlled gate is replaced by a conditional gate that uses the measurement outcomes as conditions.

module {
  // CHECK-LABEL: func.func @testRaiseMultipleMeasuresOverControlledGate
  func.func @testRaiseMultipleMeasuresOverControlledGate() -> (i1, i1) {
    // CHECK: %[[Q1_1:.*]], %[[C1]] = "mqtopt.measure"(%[[Q1_0:.*]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    // CHECK: %[[Q2_1:.*]], %[[C2]] = "mqtopt.measure"(%[[Q2_0:.*]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    // CHECK: %[[Q0_1:.*]] = mqtopt.x() %[[ANY:.*]] if %[[C1]], %[[C2]]: !mqtopt.Qubit ctrl !mqtopt.Qubit if i1, i1
    // CHECK: %[[ANY:.*]] = "mqtopt.insertQubit"(%[[Reg_2:.*]], %[[Q1_1]]) <{index_attr = 1 : i64}>
    // CHECK: %[[ANY:.*]] = "mqtopt.insertQubit"(%[[Reg_2:.*]], %[[Q2_1]]) <{index_attr = 2 : i64}>

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

// =================================================== Raise over 1Q Gates =========================================================================

// -----
// This test checks that measurements can be raised over a single X Gate by adding a classical negation afterwards.

module {
  // CHECK-LABEL: func.func @testRaiseOverSingleX
  func.func @testRaiseOverSingleX() -> i1 {
    // CHECK: %[[Q0_1:.*]], %[[C0_0]] = "mqtopt.measure"(%[[ANY.*]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    // CHECK-NOT: mqtopt.x
    // CHECK: %[[ANY:.*]] = "mqtopt.insertQubit"(%[[Reg_2:.*]], %[[Q0_1]]) <{index_attr = 0 : i64}>
    // CHECK: %[[C0_1:.*]] = arith.xori %[[C0_0]], %[[TRUE:.*]] : i1
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
// This test checks that measurements can be raised over a single Y Gate by adding a classical negation afterwards.
// Essentially, a Y Gate before a measurement can be treated just like an X Gate

module {
  // CHECK-LABEL: func.func @testRaiseOverSingleY
  func.func @testRaiseOverSingleY() -> i1 {
    // CHECK: %[[Q0_1:.*]], %[[C0_0]] = "mqtopt.measure"(%[[ANY.*]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    // CHECK-NOT: mqtopt.y
    // CHECK: %[[ANY:.*]] = "mqtopt.insertQubit"(%[[Reg_2:.*]], %[[Q0_1]]) <{index_attr = 0 : i64}>
    // CHECK: %[[C0_1:.*]] = arith.xori %[[C0_0]], %[[TRUE:.*]] : i1
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
// This test checks that measurements can be raised over local phase gates (Z, etc.) without further classical operations.

module {
  // CHECK-LABEL: func.func @testRaiseOverLocalPhaseGates
  func.func @testRaiseOverZ() -> i1 {
    // CHECK: %[[Q0_1:.*]], %[[C0]] = "mqtopt.measure"(%[[ANY.*]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
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
// This test checks that measurements can be raised over multiple X Gates by adding classical negations afterwards.

module {
  // CHECK-LABEL: func.func @testRaiseOverMultipleX
  func.func @testRaiseOverMultipleX() -> i1 {
    // CHECK: %[[Q0_1:.*]], %[[C0_0]] = "mqtopt.measure"(%[[ANY.*]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
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
// This test checks that measurements can be raised over X Gates as well as controlled gates where the measurement target is a control at the same time.

module {
  // CHECK-LABEL: func.func @testRaiseOverXAndControlledGates
  func.func @testRaiseOverXAndControlledGates() -> i1 {
    // CHECK: %[[Q0_1:.*]], %[[C0_0]] = "mqtopt.measure"(%[[ANY.*]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    // CHECK: %[[Q1_1:.*]] = mqtopt.x() %[[ANY:.*]] if %[[C0_0]]: !mqtopt.Qubit ctrl !mqtopt.Qubit if i1
    // CHECK: %[[C0_1]] = arith.xori %[[C0_0]] : i1
    // CHECK: %[[Q1_2:.*]] = mqtopt.x() %[[Q1_1]] if %[[C0_1]]: !mqtopt.Qubit ctrl !mqtopt.Qubit if i1
    // CHECK-NOT: mqtopt.x
    // CHECK: %[[ANY:.*]] = "mqtopt.insertQubit"(%[[ANY:.*]], %[[Q0_1]]) <{index_attr = 0 : i64}>
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