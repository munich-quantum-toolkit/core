// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s --mqtdyn-to-mqtopt | FileCheck %s

module {
    // CHECK-LABEL: func @foo()
    func.func @foo() -> (i1, i1) {
        // CHECK: %[[reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}>
        %r0 = "mqtdyn.allocQubitRegister" () {"size_attr" = 2 : i64} : () -> !mqtdyn.QubitRegister
        // CHECK: %[[reg_1:.*]], %[[q_0:.*]] = "mqtopt.extractQubit"(%[[reg_0]]) <{index_attr = 0 : i64}>
        %q0 = "mqtdyn.extractQubit" (%r0) {"index_attr" = 0 : i64} : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        // CHECK: %[[index_0:.*]] = arith.constant 0
        %i = arith.constant 0 : i64
        // CHECK: %[[reg_2:.*]], %[[q_1:.*]] = "mqtopt.extractQubit"(%[[reg_1]], %[[index_0]])
        // CHECK: %[[reg_3:.*]], %[[q_2:.*]] = "mqtopt.extractQubit"(%[[reg_2]], %[[index_0]])
        // CHECK: %[[reg_4:.*]], %[[q_3:.*]] = "mqtopt.extractQubit"(%[[reg_3]], %[[index_0]])
        // CHECK: %[[reg_5:.*]], %[[q_4:.*]] = "mqtopt.extractQubit"(%[[reg_4]], %[[index_0]])
        %q1 = "mqtdyn.extractQubit" (%r0, %i) : (!mqtdyn.QubitRegister, i64) -> !mqtdyn.Qubit
        %q2 = "mqtdyn.extractQubit" (%r0, %i) : (!mqtdyn.QubitRegister, i64) -> !mqtdyn.Qubit
        %q3 = "mqtdyn.extractQubit" (%r0, %i) : (!mqtdyn.QubitRegister, i64) -> !mqtdyn.Qubit
        %q4 = "mqtdyn.extractQubit" (%r0, %i) : (!mqtdyn.QubitRegister, i64) -> !mqtdyn.Qubit

        // CHECK: %[[q_5:.*]] = mqtopt.x() %[[q_0]] : !mqtopt.Qubit
        mqtdyn.x () %q0
        // CHECK: %[[q_6:.*]], %[[q_7:.*]] = mqtopt.x() %[[q_1]] ctrl %[[q_5]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
        mqtdyn.x () %q1 ctrl %q0

        // CHECK: %[[c_0:.*]] = arith.constant 3.000000e-01
        %cst = arith.constant 3.000000e-01 : f64
        // CHECK: %[[q_8:.*]] = mqtopt.u(%[[c_0]], %[[c_0]] static [3.000000e-01] mask [false, true, false]) %[[q_6]]
        mqtdyn.u(%cst, %cst static [3.000000e-01] mask [false, true, false]) %q1
        // CHECK: %[[q_9:.*]], [[b_0:.*]] = "mqtopt.measure"(%[[q_7]])
        // CHECK: %[[q_10:.*]], [[b_1:.*]] = "mqtopt.measure"(%[[q_8]])
        %c0 = "mqtdyn.measure" (%q0) : (!mqtdyn.Qubit) -> i1
        %c1 = "mqtdyn.measure" (%q1) : (!mqtdyn.Qubit) -> i1
        // CHECK: %[[reg_6:.*]] = "mqtopt.insertQubit"(%[[reg_5]], %[[q_9]]) <{index_attr = 0 : i64}>
        // CHECK: %[[reg_7:.*]] = "mqtopt.insertQubit"(%[[reg_6]], %[[q_4]], %[[index_0]])
        // CHECK: %[[reg_8:.*]] = "mqtopt.insertQubit"(%[[reg_7]], %[[q_3]], %[[index_0]])
        // CHECK: %[[reg_9:.*]] = "mqtopt.insertQubit"(%[[reg_8]], %[[q_2]], %[[index_0]])
        // CHECK: %[[reg_10:.*]] = "mqtopt.insertQubit"(%[[reg_9]], %[[q_10]], %[[index_0]])
        // CHECK: "mqtopt.deallocQubitRegister"(%[[reg_10]]) : (!mqtopt.QubitRegister) -> ()
        "mqtdyn.deallocQubitRegister" (%r0) : (!mqtdyn.QubitRegister) -> ()

        return %c0, %c1 : i1, i1
    }
}
