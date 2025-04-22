// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --cancel-consecutive-inverses | FileCheck %s

// -----
// This test checks if single-qubit consecutive self-inverses are canceled correctly.
// In this example, most operations should be canceled including cases where:
//   - The operations are directly consecutive
//   - There are operations on other qubits interleaved between them
//   - There are operations on the same qubits interleaved between them that will also get canceled,
//     allowing the outer consecutive pair to be canceled as well

module {
  func.func @testCancelSingleQubitGates() {
    // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

    // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
    %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // ========================== Check for operations that should not be canceled ==========================
    // CHECK: %[[Q0_1:.*]] = mqtopt.z() %[[Q0_0]] : !mqtopt.Qubit

    // ========================== Check for operations that should be canceled ==============================
    // CHECK-NOT: %[[ANY:.*]] = mqtopt.x() %[[ANY:.*]] : !mqtopt.Qubit
    // CHECK-NOT: %[[ANY:.*]] = mqtopt.z() %[[ANY:.*]] : !mqtopt.Qubit

    %q1_1 = mqtopt.x() %q1_0 : !mqtopt.Qubit
    %q1_2 = mqtopt.x() %q1_1 : !mqtopt.Qubit
    %q1_3 = mqtopt.x() %q1_2 : !mqtopt.Qubit
    %q1_4 = mqtopt.z() %q1_3 : !mqtopt.Qubit
    %q0_1 = mqtopt.z() %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.x() %q0_1 : !mqtopt.Qubit
    %q0_3 = mqtopt.x() %q0_2 : !mqtopt.Qubit
    %q1_5 = mqtopt.z() %q1_4 : !mqtopt.Qubit
    %q1_6 = mqtopt.x() %q1_5 : !mqtopt.Qubit

    // CHECK: %[[Reg_3:.*]] = "mqtopt.insertQubit"(%[[Reg_2]], %[[Q0_1]])  <{index_attr = 0 : i64}>
    %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_3) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q1_0]])  <{index_attr = 1 : i64}>
    %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_6) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: "mqtopt.deallocQubitRegister"(%[[Reg_4]])
    "mqtopt.deallocQubitRegister"(%reg_4) : (!mqtopt.QubitRegister) -> ()
    return
  }
}

// -----
// This test checks if two-qubit consecutive self-inverses are canceled correctly.
// For this, the operations must involve exactly the same qubits.

module {
  func.func @testCancelMultiQubitGates() {
    // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister

    // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
    %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[Reg_3:.*]], %[[Q2_0:.*]] = "mqtopt.extractQubit"(%[[Reg_2]]) <{index_attr = 2 : i64}>
    %reg_3, %q2_0 = "mqtopt.extractQubit"(%reg_2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    //===----------------------------------------------------------------===//
    //            INPUT                  OUTPUT
    // q_0: ──■────■────────────  >>>  ──────────
    //      ┌─┴─┐┌─┴─┐┌───┐       >>>  ┌───┐
    // q_1: ┤ X ├┤ X ├┤ X ├──■──  >>>  ┤ X ├──■──
    //      └───┘└───┘└─┬─┘┌─┴─┐  >>>  └─┬─┘┌─┴─┐
    // q_2: ────────────■──┤ X ├  >>>  ──■──┤ X ├
    //                     └───┘  >>>       └───┘
    //===----------------------------------------------------------------===//
    // Check for operations that should not be cancelled
    //===----------------------------------------------------------------===//
    // CHECK: %[[Q1_1:.*]], %[[Q2_1:.*]] = mqtopt.x() %[[Q1_0]] ctrl %[[Q2_0]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[Q2_2:.*]], %[[Q1_2:.*]] = mqtopt.x() %[[Q2_1]] ctrl %[[Q1_1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit

    //===----------------------------------------------------------------===//
    // Check for operations that should be canceled
    //===----------------------------------------------------------------===//
    // CHECK-NOT: %[[ANY:.*]], %[[ANY:.*]] = mqtopt.x() %[[ANY:.*]] ctrl %[[Q0_0]] : !mqtopt.Qubit, !mqtopt.Qubit

    %q1_1, %q0_1 = mqtopt.x() %q1_0 ctrl %q0_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q1_2, %q0_2 = mqtopt.x() %q1_1 ctrl %q0_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q1_3, %q2_1 = mqtopt.x() %q1_2 ctrl %q2_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q2_2, %q1_4 = mqtopt.x() %q2_1 ctrl %q1_3 : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q0_0]])  <{index_attr = 0 : i64}>
    %reg_4 = "mqtopt.insertQubit"(%reg_3, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %[[Reg_5:.*]] = "mqtopt.insertQubit"(%[[Reg_4]], %[[Q1_2]])  <{index_attr = 1 : i64}>
    %reg_5 = "mqtopt.insertQubit"(%reg_4, %q1_4) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %[[Reg_6:.*]] = "mqtopt.insertQubit"(%[[Reg_5]], %[[Q2_2]])  <{index_attr = 2 : i64}>
    %reg_6 = "mqtopt.insertQubit"(%reg_5, %q2_2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: "mqtopt.deallocQubitRegister"(%[[Reg_6]])
    "mqtopt.deallocQubitRegister"(%reg_6) : (!mqtopt.QubitRegister) -> ()
    return
  }
}

// -----
// Checks if `dagger` gates correctly cancel their inverses, too

module {
  func.func @testCancelMultiQubitGates() {
    // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister

    // CHECK: %[[Reg_1:.*]], %[[Q_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
    %reg_1, %q_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // ========================== Check for operations that should not be canceled ==========================
    // CHECK: %[[Q_1:.*]] = mqtopt.sx() %[[Q_0]]
    // CHECK: %[[Q_2:.*]] = mqtopt.sx() %[[Q_1]]

    // ========================== Check for operations that should be canceled ==============================
    // CHECK-NOT: %[[ANY:.*]] = mqtopt.s()
    // CHECK-NOT: %[[ANY:.*]] = mqtopt.tdg()
    // CHECK-NOT: %[[ANY:.*]] = mqtopt.t()
    // CHECK-NOT: %[[ANY:.*]] = mqtopt.sdg()

    %q_1 = mqtopt.s() %q_0 : !mqtopt.Qubit
    %q_2 = mqtopt.tdg() %q_1 : !mqtopt.Qubit
    %q_3 = mqtopt.t() %q_2 : !mqtopt.Qubit
    %q_4 = mqtopt.sdg() %q_3 : !mqtopt.Qubit
    %q_5 = mqtopt.sx() %q_4 : !mqtopt.Qubit
    %q_6 = mqtopt.sx() %q_5 : !mqtopt.Qubit


    // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[Reg_1]], %[[Q_2]])  <{index_attr = 0 : i64}>
    %reg_2 = "mqtopt.insertQubit"(%reg_1, %q_6) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    "mqtopt.deallocQubitRegister"(%reg_2) : (!mqtopt.QubitRegister) -> ()
    return
  }
}


// -----
// Checks that controlled gates with different control polarities are not canceled

module {
  func.func @testDontCancelDifferingControlPolarities() {
    // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

    // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
    %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // CHECK: %[[Q1_1:.*]], %[[Q0_1:.*]] = mqtopt.x() %[[Q1_0]] ctrl %[[Q0_0]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[Q1_2:.*]], %[[Q2_2:.*]] = mqtopt.x() %[[Q1_1]] nctrl %[[Q2_1]] : !mqtopt.Qubit nctrl !mqtopt.Qubit

    %q1_1, %q0_1 = mqtopt.x() %q1_0 ctrl %q0_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q1_2, %q0_2 = mqtopt.x() %q1_1 nctrl %q0_1 : !mqtopt.Qubit nctrl !mqtopt.Qubit


    %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_2) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    "mqtopt.deallocQubitRegister"(%reg_4) : (!mqtopt.QubitRegister) -> ()
    return
  }
}


// -----
// Checks that controlled gates with different numbers of controls are not canceled

module {
  func.func @testDontCancelDifferentNumberOfQubits() {
    // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister

    // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
    %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // CHECK: %[[Reg_3:.*]], %[[Q2_0:.*]] = "mqtopt.extractQubit"(%[[Reg_2]]) <{index_attr = 2 : i64}>
    %reg_3, %q2_0 = "mqtopt.extractQubit"(%reg_2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    //===------------------------------------------------------------------===//
    //        INPUT            OUTPUT
    // q_0: ──■────■──  >>>  ──■────■──
    //      ┌─┴─┐┌─┴─┐  >>>  ┌─┴─┐┌─┴─┐
    // q_1: ┤ X ├┤ X ├  >>>  ┤ X ├┤ X ├
    //      └───┘└─┬─┘  >>>  └───┘└─┬─┘
    // q_2: ───────■──  >>>  ───────■──
    //===----------------------------------------------------------------===//
    // Check for operations that should not be cancelled
    //===----------------------------------------------------------------===//
    // CHECK: %[[Q1_1:.*]], %[[Q0_1:.*]] = mqtopt.x() %[[Q1_0]] ctrl %[[Q0_0]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[Q1_2:.*]], %[[Q02_2:.*]]:2 = mqtopt.x() %[[Q1_1]] ctrl %[[Q0_1]], %[[Q2_0]] : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit

    %q1_1, %q0_1 = mqtopt.x() %q1_0 ctrl %q0_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q1_2, %q02_1:2 = mqtopt.x() %q1_1 ctrl %q0_1, %q2_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit

    %reg_4 = "mqtopt.insertQubit"(%reg_3, %q02_1#0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    %reg_5 = "mqtopt.insertQubit"(%reg_4, %q1_2) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    %reg_6 = "mqtopt.insertQubit"(%reg_5, %q02_1#1) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    "mqtopt.deallocQubitRegister"(%reg_6) : (!mqtopt.QubitRegister) -> ()
    return
  }
}
