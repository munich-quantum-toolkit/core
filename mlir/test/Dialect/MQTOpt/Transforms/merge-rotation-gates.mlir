// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --merge-rotation-gates | FileCheck %s

module {
  func.func @testMergeSingleQubitGates() {
    // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

    // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
    %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // CHECK: %[[C0_F64:.*]] = 1.000000
    %c0_f64 = arith.constant 1.000000 : f64
    // CHECK: %[[C1_F64:.*]] = 2.000000
    %c1_f64 = arith.constant 2.000000 : f64

    // ========================== Check for operation that should be merged ==========================
    // CHECK: %[[Q0_1:.*]] = mqtopt.rx(%[[C1_F64]]) %[[Q0_0]] : !mqtopt.Qubit

    %q0_1 = mqtopt.rx(%c1_f64) %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.rx(%c1_f64) %q0_1 : !mqtopt.Qubit

    // CHECK: %[[Reg_3:.*]] = "mqtopt.insertQubit"(%[[Reg_2]], %[[Q0_1]])  <{index_attr = 0 : i64}>
    %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q1_0]])  <{index_attr = 1 : i64}>
    %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_0) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: "mqtopt.deallocQubitRegister"(%[[Reg_4]])
    "mqtopt.deallocQubitRegister"(%reg_4) : (!mqtopt.QubitRegister) -> ()
    return
  }
}
