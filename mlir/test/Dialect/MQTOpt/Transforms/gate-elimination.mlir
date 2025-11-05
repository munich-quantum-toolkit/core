// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --gate-elimination | FileCheck %s

// -----
// This test checks if single-qubit consecutive self-inverses are canceled correctly.
// In this example, most operations should be canceled including cases where:
//   - The operations are directly consecutive
//   - There are operations on other qubits interleaved between them
//   - There are operations on the same qubits interleaved between them that will also get canceled,
//     allowing the outer consecutive pair to be canceled as well

module {
  func.func @testCancelSingleQubitGates() {
    // CHECK: %[[I1:.*]] = arith.constant 1 : index
    // CHECK: %[[I0:.*]] = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i0 = arith.constant 0 : index

    // CHECK: %[[Qreg:.*]] = memref.alloc() : memref<2x!mqtopt.Qubit>
    %qreg = memref.alloc() : memref<2x!mqtopt.Qubit>

    // CHECK: %[[Q0_0:.*]] =  memref.load %[[Qreg]][%[[I0]]] : memref<2x!mqtopt.Qubit>
    %q0_0 = memref.load %qreg[%i0] : memref<2x!mqtopt.Qubit>
    // CHECK: %[[Q1_0:.*]] =  memref.load %[[Qreg]][%[[I1]]] : memref<2x!mqtopt.Qubit>
    %q1_0 = memref.load %qreg[%i1] : memref<2x!mqtopt.Qubit>

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

    // CHECK: memref.store %[[Q0_1]], %[[Qreg]][%[[I0]]] : memref<2x!mqtopt.Qubit>
    memref.store %q0_3, %qreg[%i0] : memref<2x!mqtopt.Qubit>
    // CHECK: memref.store %[[Q1_0]], %[[Qreg]][%[[I1]]] : memref<2x!mqtopt.Qubit>
    memref.store %q1_6, %qreg[%i1] : memref<2x!mqtopt.Qubit>
    // CHECK: memref.dealloc %[[Qreg]] : memref<2x!mqtopt.Qubit>
    memref.dealloc %qreg : memref<2x!mqtopt.Qubit>

    return
  }
}

// -----
// This test checks if two-qubit consecutive self-inverses are canceled correctly.
// For this, the operations must involve exactly the same qubits.

module {
  func.func @testCancelMultiQubitGates() {
    // CHECK: %[[I2:.*]] = arith.constant 2 : index
    // CHECK: %[[I1:.*]] = arith.constant 1 : index
    // CHECK: %[[I0:.*]] = arith.constant 0 : index
    %i2 = arith.constant 2 : index
    %i1 = arith.constant 1 : index
    %i0 = arith.constant 0 : index

    // CHECK: %[[Qreg:.*]] = memref.alloc() : memref<3x!mqtopt.Qubit>
    %qreg = memref.alloc() : memref<3x!mqtopt.Qubit>

    // CHECK: %[[Q0_0:.*]] =  memref.load %[[Qreg]][%[[I0]]] : memref<3x!mqtopt.Qubit>
    %q0_0 = memref.load %qreg[%i0] : memref<3x!mqtopt.Qubit>
    // CHECK: %[[Q1_0:.*]] =  memref.load %[[Qreg]][%[[I1]]] : memref<3x!mqtopt.Qubit>
    %q1_0 = memref.load %qreg[%i1] : memref<3x!mqtopt.Qubit>
    // CHECK: %[[Q2_0:.*]] =  memref.load %[[Qreg]][%[[I2]]] : memref<3x!mqtopt.Qubit>
    %q2_0 = memref.load %qreg[%i2] : memref<3x!mqtopt.Qubit>

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

    // CHECK: memref.store %[[Q0_0]], %[[Qreg]][%[[I0]]] : memref<3x!mqtopt.Qubit>
    memref.store %q0_2, %qreg[%i0] : memref<3x!mqtopt.Qubit>
    // CHECK: memref.store %[[Q1_2]], %[[Qreg]][%[[I1]]] : memref<3x!mqtopt.Qubit>
    memref.store %q1_4, %qreg[%i1] : memref<3x!mqtopt.Qubit>
    // CHECK: memref.store %[[Q2_2]], %[[Qreg]][%[[I2]]] : memref<3x!mqtopt.Qubit>
    memref.store %q2_2, %qreg[%i2] : memref<3x!mqtopt.Qubit>
    // CHECK: memref.dealloc %[[Qreg]] : memref<3x!mqtopt.Qubit>
    memref.dealloc %qreg : memref<3x!mqtopt.Qubit>

    return
  }
}

// -----
// Checks if `dagger` gates correctly cancel their inverses, too

module {
  func.func @testCancelMultiQubitGates() {
    // CHECK: %[[I0:.*]] = arith.constant 0 : index
    %i0 = arith.constant 0 : index

    // CHECK: %[[Qreg:.*]] = memref.alloc() : memref<1x!mqtopt.Qubit>
    %qreg = memref.alloc() : memref<1x!mqtopt.Qubit>

    // CHECK: %[[Q_0:.*]] =  memref.load %[[Qreg]][%[[I0]]] : memref<1x!mqtopt.Qubit>
    %q_0 = memref.load %qreg[%i0] : memref<1x!mqtopt.Qubit>

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

    memref.store %q_6, %qreg[%i0] : memref<1x!mqtopt.Qubit>
    memref.dealloc %qreg : memref<1x!mqtopt.Qubit>

    return
  }
}


// -----
// Checks that controlled gates with different control polarities are not canceled

module {
  func.func @testDontCancelDifferingControlPolarities() {
    // CHECK: %[[I1:.*]] = arith.constant 1 : index
    // CHECK: %[[I0:.*]] = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i0 = arith.constant 0 : index

    // CHECK: %[[Qreg:.*]] = memref.alloc() : memref<2x!mqtopt.Qubit>
    %qreg = memref.alloc() : memref<2x!mqtopt.Qubit>

    // CHECK: %[[Q0_0:.*]] =  memref.load %[[Qreg]][%[[I0]]] : memref<2x!mqtopt.Qubit>
    %q0_0 = memref.load %qreg[%i0] : memref<2x!mqtopt.Qubit>
    // CHECK: %[[Q1_0:.*]] =  memref.load %[[Qreg]][%[[I1]]] : memref<2x!mqtopt.Qubit>
    %q1_0 = memref.load %qreg[%i1] : memref<2x!mqtopt.Qubit>

    // CHECK: %[[Q1_1:.*]], %[[Q0_1:.*]] = mqtopt.x() %[[Q1_0]] ctrl %[[Q0_0]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[Q1_2:.*]], %[[Q2_2:.*]] = mqtopt.x() %[[Q1_1]] nctrl %[[Q2_1]] : !mqtopt.Qubit nctrl !mqtopt.Qubit

    %q1_1, %q0_1 = mqtopt.x() %q1_0 ctrl %q0_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q1_2, %q0_2 = mqtopt.x() %q1_1 nctrl %q0_1 : !mqtopt.Qubit nctrl !mqtopt.Qubit

    memref.store %q0_2, %qreg[%i0] : memref<2x!mqtopt.Qubit>
    memref.store %q1_2, %qreg[%i1] : memref<2x!mqtopt.Qubit>
    memref.dealloc %qreg : memref<2x!mqtopt.Qubit>

    return
  }
}


// -----
// Checks that controlled gates with different numbers of controls are not canceled

module {
  func.func @testDontCancelDifferentNumberOfQubits() {
    // CHECK: %[[I2:.*]] = arith.constant 2 : index
    // CHECK: %[[I1:.*]] = arith.constant 1 : index
    // CHECK: %[[I0:.*]] = arith.constant 0 : index
    %i2 = arith.constant 2 : index
    %i1 = arith.constant 1 : index
    %i0 = arith.constant 0 : index

    // CHECK: %[[Qreg:.*]] = memref.alloc() : memref<3x!mqtopt.Qubit>
    %qreg = memref.alloc() : memref<3x!mqtopt.Qubit>

    // CHECK: %[[Q0_0:.*]] =  memref.load %[[Qreg]][%[[I0]]] : memref<3x!mqtopt.Qubit>
    %q0_0 = memref.load %qreg[%i0] : memref<3x!mqtopt.Qubit>
    // CHECK: %[[Q1_0:.*]] =  memref.load %[[Qreg]][%[[I1]]] : memref<3x!mqtopt.Qubit>
    %q1_0 = memref.load %qreg[%i1] : memref<3x!mqtopt.Qubit>
    // CHECK: %[[Q2_0:.*]] =  memref.load %[[Qreg]][%[[I2]]] : memref<3x!mqtopt.Qubit>
    %q2_0 = memref.load %qreg[%i2] : memref<3x!mqtopt.Qubit>

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

    memref.store %q02_1#0, %qreg[%i0] : memref<3x!mqtopt.Qubit>
    memref.store %q1_2, %qreg[%i1] : memref<3x!mqtopt.Qubit>
    memref.store %q02_1#1, %qreg[%i2] : memref<3x!mqtopt.Qubit>
    memref.dealloc %qreg : memref<3x!mqtopt.Qubit>

    return
  }
}


// -----
// This test checks if a single identity operation is removed correctly.

module {
  // CHECK-LABEL: func @testRemoveSingleIdentity()
  func.func @testRemoveSingleIdentity() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit

    // ========================== Check for operations that should not be canceled ==========================
    // CHECK: %[[Q0_1:.*]] = mqtopt.x() %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q0_3:.*]] = mqtopt.z() %[[Q0_1]] : !mqtopt.Qubit

    // ========================== Check for operations that should be canceled ==============================
    // CHECK-NOT: %[[ANY:.*]] = mqtopt.i() %[[ANY:.*]] : !mqtopt.Qubit

    // CHECK: mqtopt.deallocQubit %[[Q0_3]]

    %q0_0 = mqtopt.allocQubit

    %q0_1 = mqtopt.x() %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.i() %q0_1 : !mqtopt.Qubit
    %q0_3 = mqtopt.z() %q0_2 : !mqtopt.Qubit

    mqtopt.deallocQubit %q0_3
    return
  }
}


// -----
// This test checks if consecutive identity operations are removed correctly.

module {
  // CHECK-LABEL: func @testRemoveConsecutiveIdentities()
  func.func @testRemoveConsecutiveIdentities() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit

    // ========================== Check for operations that should not be canceled ==========================
    // CHECK: %[[Q0_1:.*]] = mqtopt.x() %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q0_7:.*]] = mqtopt.z() %[[Q0_1]] : !mqtopt.Qubit

    // ========================== Check for operations that should be canceled ==============================
    // CHECK-NOT: %[[ANY:.*]] = mqtopt.i() %[[ANY:.*]] : !mqtopt.Qubit

    // CHECK: mqtopt.deallocQubit %[[Q0_7]]

    %q0_0 = mqtopt.allocQubit

    %q0_1 = mqtopt.x() %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.i() %q0_1 : !mqtopt.Qubit
    %q0_3 = mqtopt.i() %q0_2 : !mqtopt.Qubit
    %q0_4 = mqtopt.i() %q0_3 : !mqtopt.Qubit
    %q0_5 = mqtopt.i() %q0_4 : !mqtopt.Qubit
    %q0_6 = mqtopt.i() %q0_5 : !mqtopt.Qubit
    %q0_7 = mqtopt.z() %q0_6 : !mqtopt.Qubit

    mqtopt.deallocQubit %q0_7
    return
  }
}


// -----
// This test checks if controlled identity operations are removed correctly.

module {
  // CHECK-LABEL: func @testRemoveControlledIdentities()
  func.func @testRemoveControlledIdentities() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit

    // ========================== Check for operations that should not be canceled ==========================
    // CHECK: %[[Q0_1:.*]] = mqtopt.x() %[[Q0_0]] : !mqtopt.Qubit

    // ========================== Check for operations that should be canceled ==============================
    // CHECK-NOT: %[[ANY:.*]] = mqtopt.i() %[[ANY:.*]] ctrl %[[ANY:.*]] : !mqtopt.Qubit

    // CHECK: mqtopt.deallocQubit %[[Q0_1]]
    // CHECK: mqtopt.deallocQubit %[[Q1_0]]

    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit

    %q0_1 = mqtopt.x() %q0_0 : !mqtopt.Qubit
    %q1_1, %q0_2 = mqtopt.i() %q1_0 ctrl %q0_1: !mqtopt.Qubit ctrl !mqtopt.Qubit

    mqtopt.deallocQubit %q0_2
    mqtopt.deallocQubit %q1_1
    return
  }
}


// -----
// This test checks if all identity operations are removed correctly.

module {
  // CHECK-LABEL: func @testRemoveAllIdentities()
  func.func @testRemoveAllIdentities() {
    // CHECK: %[[Q0_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q1_0:.*]] = mqtopt.allocQubit
    // CHECK: %[[Q2_0:.*]] = mqtopt.allocQubit

    // ========================== Check for operations that should not be canceled ==========================
    // CHECK: %[[Q0_1:.*]] = mqtopt.x() %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q1_4:.*]] = mqtopt.z() %[[Q1_0]] : !mqtopt.Qubit

    // ========================== Check for operations that should be canceled ==============================
    // CHECK-NOT: %[[ANY:.*]] = mqtopt.i() %[[ANY:.*]] : !mqtopt.Qubit

    // CHECK: mqtopt.deallocQubit %[[Q0_1]]
    // CHECK: mqtopt.deallocQubit %[[Q1_4]]
    // CHECK: mqtopt.deallocQubit %[[Q2_0]]

    %q0_0 = mqtopt.allocQubit
    %q1_0 = mqtopt.allocQubit
    %q2_0 = mqtopt.allocQubit

    %q0_1 = mqtopt.x() %q0_0 : !mqtopt.Qubit
    %q1_1, %q0_2 = mqtopt.i() %q1_0 ctrl %q0_1: !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_3 = mqtopt.i() %q0_2 : !mqtopt.Qubit

    %q1_2 = mqtopt.i() %q1_1 : !mqtopt.Qubit
    %q1_3 = mqtopt.i() %q1_2 : !mqtopt.Qubit
    %q1_4 = mqtopt.z() %q1_3 : !mqtopt.Qubit
    %q2_1, %q1_5, %q0_4 = mqtopt.i() %q2_0 ctrl %q1_4, %q0_3: !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit

    mqtopt.deallocQubit %q0_4
    mqtopt.deallocQubit %q1_5
    mqtopt.deallocQubit %q2_1
    return
  }
}
