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

    // ========================== Check for operations that should be inserted ==============================
    // CHECK: %[[Q1_1:.*]], %[[Q0_1:.*]] = mqtopt.x() %[[Q1_0]] ctrl %[[Q0_0]]
    // CHECK: %[[Q01_2:.*]]:2 = mqtopt.swap() %[[Q0_1]], %[[Q1_1]] : !mqtopt.Qubit, !mqtopt.Qubit

    // ========================== Check for operations that should be canceled ==============================
    // CHECK-NOT: %[[ANY:.*]] = mqtopt.x()

    // CHECK: mqtopt.deallocQubit %[[Q01_2]]#0
    // CHECK: mqtopt.deallocQubit %[[Q01_2]]#1

    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    %q0_1, %q1_1 = mqtopt.x() %q0_0 ctrl %q1_0: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q1_2, %q0_2 = mqtopt.x() %q1_1 ctrl %q0_1: !mqtopt.Qubit ctrl !mqtopt.Qubit

    mqtopt.deallocQubit %q0_2
    mqtopt.deallocQubit %q1_2

    return
  }
}


// -----
// This test checks that consecutive CNOT gates with more than one control qubit are not merged.

module {
  // CHECK-LABEL: func.func @testTooManyControlsNoSwapReconstruction
  func.func @testTooManyControlsNoSwapReconstruction() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q2_0:.*]] = mqtopt.allocQubit

    // ========================== Check for operations that should not be canceled ===========================
    // CHECK: %[[Q0_1:.*]], %[[Q1_1:.*]] = mqtopt.x() %[[Q0_0]] ctrl %[[Q1_0]]
    // CHECK: %[[Q1_2:.*]], %[[Q02_2:.*]]:2 = mqtopt.x() %[[Q1_1]] ctrl %[[Q0_1]], %[[Q2_0]]

    // ========================== Check for operations that should not be inserted ===========================
    // CHECK-NOT: %[[ANY:.*]] = mqtopt.swap()

    // CHECK: mqtopt.deallocQubit %[[Q02_2]]#0
    // CHECK: mqtopt.deallocQubit %[[Q1_2]]
    // CHECK: mqtopt.deallocQubit %[[Q02_2]]#1

    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit
    %q2_0 = mqtopt.allocQubit

    %q0_1, %q1_1 = mqtopt.x() %q0_0 ctrl %q1_0: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q1_2, %q0_2, %q2_1 = mqtopt.x() %q1_1 ctrl %q0_1, %q2_0: !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit

    mqtopt.deallocQubit %q0_2
    mqtopt.deallocQubit %q1_2
    mqtopt.deallocQubit %q2_1

    return
  }
}


// -----
// This test checks that consecutive CNOT gates with same target and control are not merged.

module {
  // CHECK-LABEL: func.func @testWrongPatternNoSwapReconstruction
  func.func @testWrongPatternNoSwapReconstruction() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit

    // ========================== Check for operations that should be canceled ==============================
    // CHECK: %[[Q0_1:.*]], %[[Q1_1:.*]] = mqtopt.x() %[[Q0_0]] ctrl %[[Q1_0]]
    // CHECK: %[[Q0_2:.*]], %[[Q1_2:.*]] = mqtopt.x() %[[Q0_1]] ctrl %[[Q1_1]]
    // CHECK: %[[Q0_3:.*]], %[[Q1_3:.*]] = mqtopt.x() %[[Q0_2]] ctrl %[[Q1_2]]

    // ========================== Check for operations that should not be inserted ===========================
    // CHECK-NOT: %[[ANY:.*]] = mqtopt.swap()

    // CHECK: mqtopt.deallocQubit %[[Q0_3]]
    // CHECK: mqtopt.deallocQubit %[[Q1_3]]

    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    %q0_1, %q1_1 = mqtopt.x() %q0_0 ctrl %q1_0: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_2, %q1_2 = mqtopt.x() %q0_1 ctrl %q1_1: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_3, %q1_3 = mqtopt.x() %q0_2 ctrl %q1_2: !mqtopt.Qubit ctrl !mqtopt.Qubit

    mqtopt.deallocQubit %q0_3
    mqtopt.deallocQubit %q1_3

    return
  }
}


// -----
// This test checks that controlled CNOT gates are merged into a controlled SWAP gate.

module {
  // CHECK-LABEL: func.func @testControlledSwapReconstruction
  func.func @testControlledSwapReconstruction() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q2_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q3_0:.*]] = mqtopt.allocQubit

    // ========================== Check for operations that should be canceled ==============================
    // CHECK-NOT: %[[Q0_1:.*]], %[[Q1_1:.*]] = mqtopt.x() %[[Q0_0]] ctrl %[[Q1_0]]
    // CHECK-NOT: %[[Q0_2:.*]], %[[Q1_2:.*]] = mqtopt.x() %[[Q1_1]] ctrl %[[Q0_1]]

    // ========================== Check for operations that should be inserted ==============================
    // CHECK: %[[Q01_1:.*]]:2, %[[Q2_1:.*]], %[[Q3_1:.*]] = mqtopt.swap() %[[Q0_0]], %[[Q1_0]]

    // CHECK: mqtopt.deallocQubit %[[Q01_1]]#0
    // CHECK: mqtopt.deallocQubit %[[Q01_1]]#1
    // CHECK: mqtopt.deallocQubit %[[Q2_1]]
    // CHECK: mqtopt.deallocQubit %[[Q3_1]]

    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit
    %q2_0 = mqtopt.allocQubit
    %q3_0 = mqtopt.allocQubit

    %q0_1, %q1_1, %q2_1, %q3_1 = mqtopt.x() %q0_0 ctrl %q1_0, %q2_0 nctrl %q3_0: !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit nctrl !mqtopt.Qubit
    %q1_2, %q0_2, %q2_2, %q3_2 = mqtopt.x() %q1_1 ctrl %q0_1, %q2_1 nctrl %q3_1: !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit nctrl !mqtopt.Qubit
    %q0_3, %q1_3, %q2_3, %q3_3 = mqtopt.x() %q0_2 ctrl %q1_2, %q2_2 nctrl %q3_2: !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit nctrl !mqtopt.Qubit

    mqtopt.deallocQubit %q0_3
    mqtopt.deallocQubit %q1_3
    mqtopt.deallocQubit %q2_3
    mqtopt.deallocQubit %q3_3

    return
  }
}


// -----
// This test checks that controlled CNOT gates with simpler controls on the surrounding CNOT gates are merged into a controlled SWAP gate.

module {
  // CHECK-LABEL: func.func @testControlledSwapReconstructionSimplerSurroundingCNots
  func.func @testControlledSwapReconstructionSimplerSurroundingCNots() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q2_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q3_0:.*]] = mqtopt.allocQubit

    // ========================== Check for operations that should be canceled ==============================
    // CHECK-NOT: %[[ANY:.*]] = mqtopt.x()

    // ========================== Check for operations that should be inserted ==============================
    // CHECK: %[[Q0123_1:.*]]:4 = mqtopt.swap() %[[Q0_0]], %[[Q1_0]] ctrl %[[Q2_0]] nctrl %[[Q3_0]]

    // CHECK: mqtopt.deallocQubit %[[Q0123_1]]#0
    // CHECK: mqtopt.deallocQubit %[[Q0123_1]]#1
    // CHECK: mqtopt.deallocQubit %[[Q0123_1]]#2
    // CHECK: mqtopt.deallocQubit %[[Q0123_1]]#3

    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit
    %q2_0 = mqtopt.allocQubit
    %q3_0 = mqtopt.allocQubit

    %q0_1, %q1_1 = mqtopt.x() %q0_0 ctrl %q1_0: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q1_2, %q0_2, %q2_1, %q3_1 = mqtopt.x() %q1_1 ctrl %q0_1, %q2_0 nctrl %q3_0: !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit nctrl !mqtopt.Qubit
    %q0_3, %q1_3 = mqtopt.x() %q0_2 ctrl %q1_2: !mqtopt.Qubit ctrl !mqtopt.Qubit

    mqtopt.deallocQubit %q0_3
    mqtopt.deallocQubit %q1_3
    mqtopt.deallocQubit %q2_1
    mqtopt.deallocQubit %q3_1

    return
  }
}


// -----
// This test checks that controlled CNOT gates with differing controls are merged into a controlled SWAP gate.

module {
  // CHECK-LABEL: func.func @testNoControlledSwapReconstruction
  func.func @testNoControlledSwapReconstruction() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q2_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q3_0:.*]] = mqtopt.allocQubit

    // ========================= Check for operations that should be kept as-is =============================
    // CHECK: %[[Q0_1:.*]], %[[Q1_1:.*]], %[[Q2_1:.*]], %[[Q3_1:.*]] = mqtopt.x() %[[Q0_0]] ctrl %[[Q1_0]], %[[Q2_0]] nctrl %[[Q3_0]]
    // CHECK: %[[Q0_2:.*]], %[[Q1_2:.*]], %[[Q3_2:.*]] = mqtopt.x() %[[Q1_1]] ctrl %[[Q0_1]] nctrl %[[Q3_1]]
    // CHECK: %[[Q0_3:.*]], %[[Q1_3:.*]], %[[Q2_2:.*]], %[[Q3_3:.*]] = mqtopt.x() %[[Q0_0]] ctrl %[[Q1_0]], %[[Q2_0]] nctrl %[[Q3_0]]

    // ======================== Check for operations that should not be inserted ============================
    // CHECK-NOT: %[[ANY:.*]] = mqtopt.swap()

    // CHECK: mqtopt.deallocQubit %[[Q0_3]]
    // CHECK: mqtopt.deallocQubit %[[Q1_3]]
    // CHECK: mqtopt.deallocQubit %[[Q2_2]]
    // CHECK: mqtopt.deallocQubit %[[Q3_3]]

    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit
    %q2_0 = mqtopt.allocQubit
    %q3_0 = mqtopt.allocQubit

    %q0_1, %q1_1, %q2_1, %q3_1 = mqtopt.x() %q0_0 ctrl %q1_0, %q2_0 nctrl %q3_0: !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit nctrl !mqtopt.Qubit
    %q1_2, %q0_2, %q3_2 = mqtopt.x() %q1_1 ctrl %q0_1 nctrl %q3_1: !mqtopt.Qubit ctrl !mqtopt.Qubit nctrl !mqtopt.Qubit
    %q0_3, %q1_3, %q2_2, %q3_3 = mqtopt.x() %q0_2 ctrl %q1_2, %q2_1 nctrl %q3_2: !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit nctrl !mqtopt.Qubit

    mqtopt.deallocQubit %q0_3
    mqtopt.deallocQubit %q1_3
    mqtopt.deallocQubit %q2_2
    mqtopt.deallocQubit %q3_3

    return
  }
}

