// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --merge-rotation-gates | FileCheck %s

// -----
// This test checks that consecutive rx gates are merged correctly.

module {
  // CHECK-LABEL: func.func @testMergeRxGates
  func.func @testMergeRxGates() {
    // CHECK: %[[Res_3:.*]] = arith.constant 3.000000e+00 : f64

    // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

    // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // CHECK: %[[Q0_3:.*]] = mqtopt.rx(%[[Res_3:.*]]) %[[Q0_0]] : !mqtopt.Qubit
    // CHECK-NOT: %[[ANY:.*]] = mqtopt.rx(%[[ANY:.*]]) %[[ANY:.*]] : !mqtopt.Qubit

    %c_0 = arith.constant 1.000000e+00 : f64
    %q0_1 = mqtopt.rx(%c_0) %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.rx(%c_0) %q0_1 : !mqtopt.Qubit
    %q0_3 = mqtopt.rx(%c_0) %q0_2 : !mqtopt.Qubit

    // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[Reg_1]], %[[Q0_3]])  <{index_attr = 0 : i64}>
    %reg_2 = "mqtopt.insertQubit"(%reg_1, %q0_3) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

    // CHECK: "mqtopt.deallocQubitRegister"(%[[Reg_2]])
    "mqtopt.deallocQubitRegister"(%reg_2) : (!mqtopt.QubitRegister) -> ()

    return
  }
}

// -----
// This test checks that consecutive ry gates are merged correctly.

module {
  // CHECK-LABEL: func.func @testMergeRyGates
  func.func @testMergeRyGates() {
    // CHECK: %[[Res_0:.*]] = arith.constant 0.000000e+00 : f64

    // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

    // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // CHECK: %[[Q0_2:.*]] = mqtopt.ry(%[[Res_0:.*]]) %[[Q0_0]] : !mqtopt.Qubit
    // CHECK-NOT: %[[ANY:.*]] = mqtopt.ry(%[[ANY:.*]]) %[[ANY:.*]] : !mqtopt.Qubit

    %c_0 = arith.constant -1.000000e+00 : f64
    %c_1 = arith.constant 1.000000e+00 : f64
    %q0_1 = mqtopt.ry(%c_0) %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.ry(%c_1) %q0_1 : !mqtopt.Qubit

    // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[Reg_1]], %[[Q0_2]])  <{index_attr = 0 : i64}>
    %reg_2 = "mqtopt.insertQubit"(%reg_1, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

    // CHECK: "mqtopt.deallocQubitRegister"(%[[Reg_2]])
    "mqtopt.deallocQubitRegister"(%reg_2) : (!mqtopt.QubitRegister) -> ()

    return
  }
}

// -----
// This test checks that consecutive rz gates are merged correctly.

module {
  // CHECK-LABEL: func.func @testMergeRzGates
  func.func @testMergeRzGates() {
    // CHECK: %[[Res_3:.*]] = arith.constant 3.000000e+00 : f64

    // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

    // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // CHECK: %[[Q0_2:.*]] = mqtopt.rz(%[[Res_3:.*]]) %[[Q0_0]] : !mqtopt.Qubit
    // CHECK-NOT: %[[ANY:.*]] = mqtopt.rz(%[[ANY:.*]]) %[[ANY:.*]] : !mqtopt.Qubit

    %c_0 = arith.constant 1.000000e+00 : f64
    %c_1 = arith.constant 2.000000e+00 : f64
    %q0_1 = mqtopt.rz(%c_0) %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.rz(%c_1) %q0_1 : !mqtopt.Qubit

    // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[Reg_1]], %[[Q0_2]])  <{index_attr = 0 : i64}>
    %reg_2 = "mqtopt.insertQubit"(%reg_1, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

    // CHECK: "mqtopt.deallocQubitRegister"(%[[Reg_2]])
    "mqtopt.deallocQubitRegister"(%reg_2) : (!mqtopt.QubitRegister) -> ()

    return
  }
}

// -----
// This test checks that incompatible single-qubit gates are not merged.

module {
  // CHECK-LABEL: func.func @testDoNotMergeIncompatibleSingleQubitGates
  func.func @testDoNotMergeIncompatibleSingleQubitGates() {
    // CHECK: %[[Res_2:.*]] = arith.constant 2.000000e+00 : f64
    // CHECK: %[[Res_1:.*]] = arith.constant 1.000000e+00 : f64

    // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

    // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // CHECK: %[[Q0_1:.*]] = mqtopt.rx(%[[Res_1:.*]]) %[[Q0_0]] : !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]] = mqtopt.ry(%[[Res_1:.*]]) %[[Q0_1]] : !mqtopt.Qubit
    // CHECK: %[[Q0_3:.*]] = mqtopt.rz(%[[Res_2:.*]]) %[[Q0_2]] : !mqtopt.Qubit

    %c_0 = arith.constant 1.000000e+00 : f64
    %c_1 = arith.constant 2.000000e+00 : f64
    %q0_1 = mqtopt.rx(%c_0) %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.ry(%c_0) %q0_1 : !mqtopt.Qubit
    %q0_3 = mqtopt.rz(%c_1) %q0_2 : !mqtopt.Qubit

    // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[Reg_1]], %[[Q0_3]])  <{index_attr = 0 : i64}>
    %reg_2 = "mqtopt.insertQubit"(%reg_1, %q0_3) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

    // CHECK: "mqtopt.deallocQubitRegister"(%[[Reg_2]])
    "mqtopt.deallocQubitRegister"(%reg_2) : (!mqtopt.QubitRegister) -> ()

    return
  }
}

// -----
// This test checks that consecutive rxx gates are merged correctly.

module {
  // CHECK-LABEL: func.func @testMergeRxxGates
  func.func @testMergeRxxGates() {
    // CHECK: %[[Res_3:.*]] = arith.constant 3.000000e+00 : f64

    // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister

    // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
    %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // CHECK: %[[Q01_3:.*]]:2 = mqtopt.rxx(%[[Res_3:.*]]) %[[Q0_0]], %[[Q1_0]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK-NOT: %[[ANY:.*]]:2 = mqtopt.rxx(%[[ANY:.*]]) %[[ANY]], %[[ANY]] : !mqtopt.Qubit, !mqtopt.Qubit

    %c_0 = arith.constant 1.000000e+00 : f64
    %q01_1:2 = mqtopt.rxx(%c_0) %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
    %q01_2:2 = mqtopt.rxx(%c_0) %q01_1#0, %q01_1#1 : !mqtopt.Qubit, !mqtopt.Qubit
    %q01_3:2 = mqtopt.rxx(%c_0) %q01_2#0, %q01_2#1 : !mqtopt.Qubit, !mqtopt.Qubit

    // CHECK: %[[Reg_3:.*]] = "mqtopt.insertQubit"(%[[Reg_2]], %[[Q01_3]]#0)  <{index_attr = 0 : i64}>
    %reg_3 = "mqtopt.insertQubit"(%reg_2, %q01_3#0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q01_3]]#1)  <{index_attr = 1 : i64}>
    %reg_4 = "mqtopt.insertQubit"(%reg_3, %q01_3#1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

    // CHECK: "mqtopt.deallocQubitRegister"(%[[Reg_4]])
    "mqtopt.deallocQubitRegister"(%reg_4) : (!mqtopt.QubitRegister) -> ()

    return
  }
}

// -----
// This test checks that consecutive ryy gates are merged correctly.

module {
  // CHECK-LABEL: func.func @testMergeRyyGates
  func.func @testMergeRyyGates() {
    // CHECK: %[[Res_0:.*]] = arith.constant 0.000000e+00 : f64

    // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister

    // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
    %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // CHECK: %[[Q01_2:.*]]:2 = mqtopt.ryy(%[[Res_0:.*]]) %[[Q0_0]], %[[Q1_0]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK-NOT: %[[ANY:.*]]:2 = mqtopt.ryy(%[[ANY:.*]]) %[[ANY]], %[[ANY]] : !mqtopt.Qubit, !mqtopt.Qubit

    %c_0 = arith.constant -1.000000e+00 : f64
    %c_1 = arith.constant 1.000000e+00 : f64
    %q01_1:2 = mqtopt.ryy(%c_0) %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
    %q01_2:2 = mqtopt.ryy(%c_1) %q01_1#0, %q01_1#1 : !mqtopt.Qubit, !mqtopt.Qubit

    // CHECK: %[[Reg_3:.*]] = "mqtopt.insertQubit"(%[[Reg_2]], %[[Q01_2]]#0)  <{index_attr = 0 : i64}>
    %reg_3 = "mqtopt.insertQubit"(%reg_2, %q01_2#0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q01_2]]#1)  <{index_attr = 1 : i64}>
    %reg_4 = "mqtopt.insertQubit"(%reg_3, %q01_2#1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

    // CHECK: "mqtopt.deallocQubitRegister"(%[[Reg_4]])
    "mqtopt.deallocQubitRegister"(%reg_4) : (!mqtopt.QubitRegister) -> ()

    return
  }
}

// -----
// This test checks that consecutive rzz gates are merged correctly.

module {
  // CHECK-LABEL: func.func @testMergeRzzGates
  func.func @testMergeRzzGates() {
    // CHECK: %[[Res_3:.*]] = arith.constant 3.000000e+00 : f64

    // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister

    // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
    %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // CHECK: %[[Q01_2:.*]]:2 = mqtopt.rzz(%[[Res_3:.*]]) %[[Q0_0]], %[[Q1_0]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK-NOT: %[[ANY:.*]]:2 = mqtopt.rzz(%[[ANY:.*]]) %[[ANY]], %[[ANY]] : !mqtopt.Qubit, !mqtopt.Qubit

    %c_0 = arith.constant 1.000000e+00 : f64
    %c_1 = arith.constant 2.000000e+00 : f64
    %q01_1:2 = mqtopt.rzz(%c_0) %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
    %q01_2:2 = mqtopt.rzz(%c_1) %q01_1#0, %q01_1#1 : !mqtopt.Qubit, !mqtopt.Qubit

    // CHECK: %[[Reg_3:.*]] = "mqtopt.insertQubit"(%[[Reg_2]], %[[Q01_2]]#0)  <{index_attr = 0 : i64}>
    %reg_3 = "mqtopt.insertQubit"(%reg_2, %q01_2#0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q01_2]]#1)  <{index_attr = 1 : i64}>
    %reg_4 = "mqtopt.insertQubit"(%reg_3, %q01_2#1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

    // CHECK: "mqtopt.deallocQubitRegister"(%[[Reg_4]])
    "mqtopt.deallocQubitRegister"(%reg_4) : (!mqtopt.QubitRegister) -> ()

    return
  }
}

// -----
// This test checks that consecutive rzx gates are merged correctly.

module {
  // CHECK-LABEL: func.func @testMergeRzxGates
  func.func @testMergeRzxGates() {
    // CHECK: %[[Res_3:.*]] = arith.constant 3.000000e+00 : f64

    // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister

    // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
    %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // CHECK: %[[Q01_2:.*]]:2 = mqtopt.rzx(%[[Res_3:.*]]) %[[Q0_0]], %[[Q1_0]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK-NOT: %[[ANY:.*]]:2 = mqtopt.rzx(%[[ANY:.*]]) %[[ANY]], %[[ANY]] : !mqtopt.Qubit, !mqtopt.Qubit

    %c_0 = arith.constant 1.000000e+00 : f64
    %c_1 = arith.constant 2.000000e+00 : f64
    %q01_1:2 = mqtopt.rzx(%c_0) %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
    %q01_2:2 = mqtopt.rzx(%c_1) %q01_1#0, %q01_1#1 : !mqtopt.Qubit, !mqtopt.Qubit

    // CHECK: %[[Reg_3:.*]] = "mqtopt.insertQubit"(%[[Reg_2]], %[[Q01_2]]#0)  <{index_attr = 0 : i64}>
    %reg_3 = "mqtopt.insertQubit"(%reg_2, %q01_2#0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q01_2]]#1)  <{index_attr = 1 : i64}>
    %reg_4 = "mqtopt.insertQubit"(%reg_3, %q01_2#1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

    // CHECK: "mqtopt.deallocQubitRegister"(%[[Reg_4]])
    "mqtopt.deallocQubitRegister"(%reg_4) : (!mqtopt.QubitRegister) -> ()

    return
  }
}

// -----
// This test checks that incompatible multi-qubit gates are not merged.

module {
  // CHECK-LABEL: func.func @testDoNotMergeIncompatibleMultiQubitGates
  func.func @testDoNotMergeIncompatibleMultiQubitGates() {
    // CHECK: %[[Res_2:.*]] = arith.constant 2.000000e+00 : f64
    // CHECK: %[[Res_1:.*]] = arith.constant 1.000000e+00 : f64

    // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
    %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister

    // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
    %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
    %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // CHECK: %[[Q01_1:.*]]:2 = mqtopt.rxx(%[[Res_1:.*]]) %[[Q0_0]], %[[Q1_0]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[Q01_2:.*]]:2 = mqtopt.ryy(%[[Res_1:.*]]) %[[Q01_1]]#0, %[[Q01_1]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[Q01_3:.*]]:2 = mqtopt.rzz(%[[Res_2:.*]]) %[[Q01_2]]#0, %[[Q01_2]]#1 : !mqtopt.Qubit, !mqtopt.Qubit

    %c_0 = arith.constant 1.000000e+00 : f64
    %c_1 = arith.constant 2.000000e+00 : f64
    %q01_1:2 = mqtopt.rxx(%c_0) %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
    %q01_2:2 = mqtopt.ryy(%c_0) %q01_1#0, %q01_1#1 : !mqtopt.Qubit, !mqtopt.Qubit
    %q01_3:2 = mqtopt.rzz(%c_1) %q01_2#0, %q01_2#1 : !mqtopt.Qubit, !mqtopt.Qubit

    // CHECK: %[[Reg_3:.*]] = "mqtopt.insertQubit"(%[[Reg_2]], %[[Q01_3]]#0)  <{index_attr = 0 : i64}>
    %reg_3 = "mqtopt.insertQubit"(%reg_2, %q01_3#0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q01_3]]#1)  <{index_attr = 1 : i64}>
    %reg_4 = "mqtopt.insertQubit"(%reg_3, %q01_3#1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

    // CHECK: "mqtopt.deallocQubitRegister"(%[[Reg_4]])
    "mqtopt.deallocQubitRegister"(%reg_4) : (!mqtopt.QubitRegister) -> ()

    return
  }
}
