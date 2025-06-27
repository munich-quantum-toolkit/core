// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --merge-rotation-gates | FileCheck %s

// -----
// This test checks if consecutive single-qubit gates are merged and canceled correctly.
// If a gate has no consecutive partner, it is left as is.
// If consecutive gates A and B satisfy angle_A == -angle_B, they are canceled.
// Otherwise, the gates are merged by adding their angles.

module {
  // CHECK-LABEL: func.func @testMergeSingleQubitGates
  func.func @testMergeSingleQubitGates() {
    // CHECK: %[[Res_2:.*]] = arith.constant 2.000000e+00 : f64
    // CHECK: %[[Res_3:.*]] = arith.constant 3.000000e+00 : f64
    // CHECK: %[[Res_1:.*]] = arith.constant 1.000000e+00 : f64

    // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

    // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
    %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // CHECK: %[[Q0_3:.*]] = mqtopt.rx(%[[Res_3:.*]]) %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q1_1:.*]] = mqtopt.ry(%[[Res_1:.*]]) %[[Q1_0]] : !mqtopt.Qubit
    // CHECK: %[[Q1_3:.*]] = mqtopt.rz(%[[Res_2:.*]]) %[[Q1_1]] : !mqtopt.Qubit
    // CHECK-NOT: %[[ANY:.*]] = mqtopt.rx(%[[ANY:.*]]) %[[ANY:.*]] : !mqtopt.Qubit

    %c_0 = arith.constant 1.000000e+00 : f64
    %c_1 = arith.constant -1.000000e+00 : f64
    %q0_1 = mqtopt.rx(%c_0) %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.rx(%c_0) %q0_1 : !mqtopt.Qubit
    %q0_3 = mqtopt.rx(%c_0) %q0_2 : !mqtopt.Qubit
    %q1_1 = mqtopt.ry(%c_0) %q1_0 : !mqtopt.Qubit
    %q1_2 = mqtopt.rz(%c_0) %q1_1 : !mqtopt.Qubit
    %q1_3 = mqtopt.rz(%c_0) %q1_2 : !mqtopt.Qubit
    %q1_4 = mqtopt.ry(%c_0) %q1_3 : !mqtopt.Qubit
    %q1_5 = mqtopt.ry(%c_1) %q1_4 : !mqtopt.Qubit

    // CHECK: %[[Reg_3:.*]] = "mqtopt.insertQubit"(%[[Reg_2]], %[[Q0_3]])  <{index_attr = 0 : i64}>
    %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_3) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q1_3]])  <{index_attr = 1 : i64}>
    %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_5) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

    // CHECK: "mqtopt.deallocQubitRegister"(%[[Reg_4]])
    "mqtopt.deallocQubitRegister"(%reg_4) : (!mqtopt.QubitRegister) -> ()

    return
  }
}

// -----
// This test checks if consecutive multi-qubit rotation gates are merged and canceled correctly.
// If a gate has no consecutive partner, it is left as is.
// If consecutive gates A and B satisfy angle_A == -angle_B, they are canceled.
// Otherwise, the gates are merged by adding their angles.

module {
  // CHECK-LABEL: func.func @testMergeMultiQubitGates
  func.func @testMergeMultiQubitGates() {
    // CHECK: %[[Res_2:.*]] = arith.constant 2.000000e+00 : f64
    // CHECK: %[[Res_1:.*]] = arith.constant 1.000000e+00 : f64

    // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister

    // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
    %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[Reg_3:.*]], %[[Q2_0:.*]] = "mqtopt.extractQubit"(%[[Reg_2]]) <{index_attr = 2 : i64}>
    %reg_3, %q2_0 = "mqtopt.extractQubit"(%reg_2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // CHECK: %[[Q01_1:.*]]:2 = mqtopt.rxx(%[[Res_1:.*]]) %[[Q0_0]], %[[Q1_0]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[Q12_2:.*]]:2 = mqtopt.rxx(%[[Res_2:.*]]) %[[Q01_1]]#1, %[[Q2_0]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[Q01_3:.*]]:2 = mqtopt.ryy(%[[Res_2:.*]]) %[[Q01_1]]#0, %[[Q12_2]]#0 : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK-NOT: %[[ANY:.*]]:2 = mqtopt.rxx(%[[ANY:.*]]) %[[ANY:.*]], %[[ANY:.*]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK-NOT: %[[ANY:.*]]:2 = mqtopt.ryy(%[[ANY:.*]]) %[[ANY:.*]], %[[ANY:.*]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK-NOT: %[[ANY:.*]]:2 = mqtopt.rzz(%[[ANY:.*]]) %[[ANY:.*]], %[[ANY:.*]] : !mqtopt.Qubit, !mqtopt.Qubit

    %c_0 = arith.constant 1.000000e+00 : f64
    %c_1 = arith.constant -1.000000e+00 : f64
    %q01_1:2 = mqtopt.rxx(%c_0) %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
    %q12_1:2 = mqtopt.rxx(%c_0) %q01_1#1, %q2_0 : !mqtopt.Qubit, !mqtopt.Qubit
    %q12_2:2 = mqtopt.rxx(%c_0) %q12_1#0, %q12_1#1 : !mqtopt.Qubit, !mqtopt.Qubit
    %q01_2:2 = mqtopt.ryy(%c_0) %q01_1#0, %q12_2#0 : !mqtopt.Qubit, !mqtopt.Qubit
    %q01_3:2 = mqtopt.ryy(%c_0) %q01_2#0, %q01_2#1 : !mqtopt.Qubit, !mqtopt.Qubit
    %q12_3:2 = mqtopt.rzz(%c_0) %q01_3#1, %q12_2#1 : !mqtopt.Qubit, !mqtopt.Qubit
    %q12_4:2 = mqtopt.rzz(%c_1) %q12_3#0, %q12_3#1 : !mqtopt.Qubit, !mqtopt.Qubit

    // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q01_3]]#0)  <{index_attr = 0 : i64}>
    %reg_4 = "mqtopt.insertQubit"(%reg_3, %q01_3#0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %[[Reg_5:.*]] = "mqtopt.insertQubit"(%[[Reg_4]], %[[Q01_3]]#1)  <{index_attr = 1 : i64}>
    %reg_5 = "mqtopt.insertQubit"(%reg_4, %q12_4#0) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %[[Reg_6:.*]] = "mqtopt.insertQubit"(%[[Reg_5]], %[[Q12_2]]#1)  <{index_attr = 2 : i64}>
    %reg_6 = "mqtopt.insertQubit"(%reg_5, %q12_4#1) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

    // CHECK: "mqtopt.deallocQubitRegister"(%[[Reg_6]])
    "mqtopt.deallocQubitRegister"(%reg_6) : (!mqtopt.QubitRegister) -> ()

    return
  }
}

// -----
// This test checks if consecutive xxminusyy and xxplusyy gates are canceled correctly.
// If a gate has no consecutive partner, it is left as is.
// If consecutive gates A and B satisfy theta_A == -theta_B, they are canceled.
// Currently, we do not support the merging of xxminusyy and xxplusyy gates.

module {
  // CHECK-LABEL: func.func @testCancelXxMinusPlusYyGates
  func.func @testCancelXxMinusPlusYyGates() {
    // CHECK: %[[Res_2:.*]] = arith.constant 2.000000e+00 : f64
    // CHECK: %[[Res_1:.*]] = arith.constant 1.000000e+00 : f64

    // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister

    // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
    %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[Reg_3:.*]], %[[Q2_0:.*]] = "mqtopt.extractQubit"(%[[Reg_2]]) <{index_attr = 2 : i64}>
    %reg_3, %q2_0 = "mqtopt.extractQubit"(%reg_2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // CHECK: %[[Q01_1:.*]]:2 = mqtopt.xxplusyy(%[[Res_1:.*]], %[[Res_2:.*]]) %[[Q0_0]], %[[Q1_0]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK-NOT: %[[ANY:.*]]:2 = mqtopt.xxminusyy(%[[ANY:.*]], %[[ANY:.*]]), %[[ANY:.*]] : !mqtopt.Qubit, !mqtopt.Qubit

    %c_0 = arith.constant -1.000000e+00 : f64
    %c_1 = arith.constant 1.000000e+00 : f64
    %c_2 = arith.constant 2.000000e+00 : f64
    %q01_1:2 = mqtopt.xxplusyy(%c_1, %c_2) %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
    %q12_1:2 = mqtopt.xxminusyy(%c_1, %c_2) %q01_1#1, %q2_0 : !mqtopt.Qubit, !mqtopt.Qubit
    %q12_2:2 = mqtopt.xxminusyy(%c_0, %c_2) %q12_1#0, %q12_1#1 : !mqtopt.Qubit, !mqtopt.Qubit

    // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q01_1]]#0)  <{index_attr = 0 : i64}>
    %reg_4 = "mqtopt.insertQubit"(%reg_3, %q01_1#0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %[[Reg_5:.*]] = "mqtopt.insertQubit"(%[[Reg_4]], %[[ANY:.*]])  <{index_attr = 1 : i64}>
    %reg_5 = "mqtopt.insertQubit"(%reg_4, %q12_2#0) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %[[Reg_6:.*]] = "mqtopt.insertQubit"(%[[Reg_5]], %[[ANY:.*]])  <{index_attr = 2 : i64}>
    %reg_6 = "mqtopt.insertQubit"(%reg_5, %q12_2#1) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

    // CHECK: "mqtopt.deallocQubitRegister"(%[[Reg_6]])
    "mqtopt.deallocQubitRegister"(%reg_6) : (!mqtopt.QubitRegister) -> ()

    return
  }
}

// -----
// This test checks if consecutive u and u2 gates are canceled correctly.
// If a gate has no consecutive partner, it is left as is.
// If consecutive gates A and B have compatible parameters, they are canceled.
// Currently, we do not support the merging of u and u2 gates.

module {
  // CHECK-LABEL: func.func @testCancelUGates
  func.func @testCancelUGates() {
    // CHECK: %[[Res_p2:.*]] = arith.constant 2.000000e+00 : f64
    // CHECK: %[[Res_p1:.*]] = arith.constant 1.000000e+00 : f64
    // CHECK: %[[Res_m1:.*]] = arith.constant -1.000000e+00 : f64

    // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

    // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
    %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // CHECK: %[[Q0_1:.*]] = mqtopt.u(%[[Res_m1:.*]], %[[Res_p1:.*]], %[[Res_p1:.*]]) %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]] = mqtopt.u(%[[Res_m1:.*]], %[[Res_m1:.*]], %[[Res_m1:.*]]) %[[Q0_1]] : !mqtopt.Qubit
    // CHECK: %[[Q0_3:.*]] = mqtopt.u2(%[[Res_p1:.*]], %[[Res_p2:.*]]) %[[Q0_2]] : !mqtopt.Qubit
    // CHECK-NOT: %[[ANY:.*]] = mqtopt.u(%[[ANY:.*]], %[[ANY:.*]], %[[ANY:.*]]) %[[ANY:.*]] : !mqtopt.Qubit

    %c_0 = arith.constant -1.000000e+00 : f64
    %c_1 = arith.constant 1.000000e+00 : f64
    %c_2 = arith.constant 2.000000e+00 : f64
    %q0_1 = mqtopt.u(%c_0, %c_1, %c_1) %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.u(%c_0, %c_0, %c_0) %q0_1 : !mqtopt.Qubit
    %q0_3 = mqtopt.u2(%c_1, %c_2) %q0_2 : !mqtopt.Qubit
    %q1_1 = mqtopt.u(%c_0, %c_1, %c_1) %q1_0 : !mqtopt.Qubit
    %q1_2 = mqtopt.u(%c_1, %c_0, %c_0) %q1_1 : !mqtopt.Qubit

    // CHECK: %[[Reg_3:.*]] = "mqtopt.insertQubit"(%[[Reg_2]], %[[Q0_3]])  <{index_attr = 0 : i64}>
    %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_3) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[ANY:.*]])  <{index_attr = 1 : i64}>
    %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_2) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

    // CHECK: "mqtopt.deallocQubitRegister"(%[[Reg_4]])
    "mqtopt.deallocQubitRegister"(%reg_4) : (!mqtopt.QubitRegister) -> ()

    return
  }
}
