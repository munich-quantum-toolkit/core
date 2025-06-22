// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s --mqtopt-to-mqtdyn | FileCheck %s

module {
// CHECK-LABEL: func @foo()
  func.func @foo() -> (i1, i1) {
    // CHECK: %[[reg_0:.*]] = "mqtdyn.allocQubitRegister"() <{size_attr = 5 : i64}>
    %r0 = "mqtopt.allocQubitRegister"() <{size_attr = 5 : i64}> : () -> !mqtopt.QubitRegister
    // CHECK: %[[q_0:.*]] = "mqtdyn.extractQubit"(%[[reg_0]]) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
    %r1, %q0 = "mqtopt.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[index_0:.*]] = arith.constant 1
    %i = arith.constant 1 : i64
    // CHECK: %[[q_1:.*]] = "mqtdyn.extractQubit"(%[[reg_0]], %[[index_0]])
    // CHECK: %[[q_2:.*]] = "mqtdyn.extractQubit"(%[[reg_0]]) <{index_attr = 2 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
    // CHECK: %[[q_3:.*]] = "mqtdyn.extractQubit"(%[[reg_0]]) <{index_attr = 3 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
    // CHECK: %[[q_4:.*]] = "mqtdyn.extractQubit"(%[[reg_0]]) <{index_attr = 4 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
    %r2, %q1 = "mqtopt.extractQubit"(%r1, %i) : (!mqtopt.QubitRegister, i64) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    %r3, %q2 = "mqtopt.extractQubit"(%r2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    %r4, %q3 = "mqtopt.extractQubit"(%r3) <{index_attr = 3 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    %r5, %q4 = "mqtopt.extractQubit"(%r4) <{index_attr = 4 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // CHECK: mqtdyn.x() %[[q_0]]
    %q5 = mqtopt.x() %q0 : !mqtopt.Qubit
    // CHECK: mqtdyn.x() %[[q_1]] ctrl %[[q_0]]
    %q6, %q7 = mqtopt.x() %q1 ctrl %q5 : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // CHECK: %[[c_0:.*]] = arith.constant 3.000000e-01
    %cst = arith.constant 3.000000e-01 : f64
    // CHECK:  mqtdyn.u(%[[c_0]], %[[c_0]] static [3.000000e-01] mask [false, true, false]) %[[q_1]]
    %q8 = mqtopt.u(%cst, %cst static [3.000000e-01] mask [false, true, false]) %q6 : !mqtopt.Qubit
    // CHECK:  [[b_0:.*]] = "mqtdyn.measure"(%[[q_0]])
    // CHECK:  [[b_1:.*]] = "mqtdyn.measure"(%[[q_1]])
    %q9, %c0 = "mqtopt.measure"(%q7) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    %q10, %c1 = "mqtopt.measure"(%q8) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    %r6 = "mqtopt.insertQubit"(%r5, %q9) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    %r7 = "mqtopt.insertQubit"(%r6, %q4)  <{index_attr = 4 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    %r8 = "mqtopt.insertQubit"(%r7, %q3)  <{index_attr = 3 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    %r9 = "mqtopt.insertQubit"(%r8, %q2)  <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    %r10 = "mqtopt.insertQubit"(%r9, %q10, %i) : (!mqtopt.QubitRegister, !mqtopt.Qubit, i64) -> !mqtopt.QubitRegister
    // CHECK: "mqtdyn.deallocQubitRegister"(%[[reg_0]]) : (!mqtdyn.QubitRegister) -> ()
    "mqtopt.deallocQubitRegister"(%r10) : (!mqtopt.QubitRegister) -> ()

    return %c0, %c1 : i1, i1
  }
}
