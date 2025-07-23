// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --mqt-core-round-trip | FileCheck %s

// -----
// This test checks if all unitary gates are detected correctly.
module {
    // CHECK-LABEL: func.func @testUnitaryGateDetection()
    func.func @testUnitaryGateDetection() -> () {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}>
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>

        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Q0_1:.*]] = mqtopt.i() %[[Q0_0]] : !mqtopt.Qubit
        // CHECK: %[[Q0_2:.*]] = mqtopt.h() %[[Q0_1]] : !mqtopt.Qubit
        // CHECK: %[[Q0_3:.*]] = mqtopt.x() %[[Q0_2]] : !mqtopt.Qubit
        // CHECK: %[[Q0_4:.*]] = mqtopt.y() %[[Q0_3]] : !mqtopt.Qubit
        // CHECK: %[[Q0_5:.*]] = mqtopt.z() %[[Q0_4]] : !mqtopt.Qubit
        // CHECK: %[[Q0_6:.*]] = mqtopt.s() %[[Q0_5]] : !mqtopt.Qubit
        // CHECK: %[[Q0_7:.*]] = mqtopt.sdg() %[[Q0_6]] : !mqtopt.Qubit
        // CHECK: %[[Q0_8:.*]] = mqtopt.t() %[[Q0_7]] : !mqtopt.Qubit
        // CHECK: %[[Q0_9:.*]] = mqtopt.tdg() %[[Q0_8]] : !mqtopt.Qubit
        // CHECK: %[[Q0_10:.*]] = mqtopt.v() %[[Q0_9]] : !mqtopt.Qubit
        // CHECK: %[[Q0_11:.*]] = mqtopt.vdg() %[[Q0_10]] : !mqtopt.Qubit
        // CHECK: %[[Q0_12:.*]] = mqtopt.sx() %[[Q0_11]] : !mqtopt.Qubit
        // CHECK: %[[Q0_13:.*]] = mqtopt.sxdg() %[[Q0_12]] : !mqtopt.Qubit

        %q0_1 = mqtopt.i() %q0_0 : !mqtopt.Qubit
        %q0_2 = mqtopt.h() %q0_1 : !mqtopt.Qubit
        %q0_3 = mqtopt.x() %q0_2 : !mqtopt.Qubit
        %q0_4 = mqtopt.y() %q0_3 : !mqtopt.Qubit
        %q0_5 = mqtopt.z() %q0_4 : !mqtopt.Qubit
        %q0_6 = mqtopt.s() %q0_5 : !mqtopt.Qubit
        %q0_7 = mqtopt.sdg() %q0_6 : !mqtopt.Qubit
        %q0_8 = mqtopt.t() %q0_7 : !mqtopt.Qubit
        %q0_9 = mqtopt.tdg() %q0_8 : !mqtopt.Qubit
        %q0_10 = mqtopt.v() %q0_9 : !mqtopt.Qubit
        %q0_11 = mqtopt.vdg() %q0_10 : !mqtopt.Qubit
        %q0_12 = mqtopt.sx() %q0_11 : !mqtopt.Qubit
        %q0_13 = mqtopt.sxdg() %q0_12 : !mqtopt.Qubit

        // CHECK: %[[Q0_14:.*]], %[[Q1_1:.*]] = mqtopt.swap() %[[Q0_13]], %[[Q1_0]] : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q0_15:.*]], %[[Q1_2:.*]] = mqtopt.iswap() %[[Q0_14]], %[[Q1_1]] : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q0_16:.*]], %[[Q1_3:.*]] = mqtopt.iswapdg() %[[Q0_15]], %[[Q1_2]] : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q0_17:.*]], %[[Q1_4:.*]] = mqtopt.peres() %[[Q0_16]], %[[Q1_3]] : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q0_18:.*]], %[[Q1_5:.*]] = mqtopt.peresdg() %[[Q0_17]], %[[Q1_4]] : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q0_19:.*]], %[[Q1_6:.*]] = mqtopt.dcx() %[[Q0_18]], %[[Q1_5]] : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q0_20:.*]], %[[Q1_7:.*]] = mqtopt.ecr() %[[Q0_19]], %[[Q1_6]] : !mqtopt.Qubit, !mqtopt.Qubit

        %q0_14, %q1_1 = mqtopt.swap() %q0_13, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
        %q0_15, %q1_2 = mqtopt.iswap() %q0_14, %q1_1 : !mqtopt.Qubit, !mqtopt.Qubit
        %q0_16, %q1_3 = mqtopt.iswapdg() %q0_15, %q1_2 : !mqtopt.Qubit, !mqtopt.Qubit
        %q0_17, %q1_4 = mqtopt.peres() %q0_16, %q1_3 : !mqtopt.Qubit, !mqtopt.Qubit
        %q0_18, %q1_5 = mqtopt.peresdg() %q0_17, %q1_4 : !mqtopt.Qubit, !mqtopt.Qubit
        %q0_19, %q1_6 = mqtopt.dcx() %q0_18, %q1_5 : !mqtopt.Qubit, !mqtopt.Qubit
        %q0_20, %q1_7 = mqtopt.ecr() %q0_19, %q1_6 : !mqtopt.Qubit, !mqtopt.Qubit

        // CHECK: %[[Reg_3:.*]] = "mqtopt.insertQubit"(%[[Reg_2]], %[[Q0_20]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q1_7]]) <{index_attr = 1 : i64}>

        %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_20) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_7) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return
    }
}

// -----
module {
    // CHECK-LABEL: func @bell()
    func.func @bell() -> (!mqtopt.QubitRegister, i1, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}>
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Q0_1:.*]] = mqtopt.x() %[[Q0_0]] : !mqtopt.Qubit
        %q0_1 = mqtopt.x() %q0_0 : !mqtopt.Qubit
        // CHECK: %[[Q1_1:.*]] = mqtopt.x() %[[Q1_0]] : !mqtopt.Qubit
        %q1_1 = mqtopt.x() %q1_0 : !mqtopt.Qubit

        // CHECK: %[[Q0_2:.*]], %[[Q1_2:.*]] = mqtopt.x() %[[Q0_1]] ctrl %[[Q1_1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
        %q0_2, %q1_2 = mqtopt.x() %q0_1 ctrl %q1_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit

        // CHECK: %[[Q1_3:.*]] = mqtopt.x() %[[Q1_2]] : !mqtopt.Qubit
        %q1_3 = mqtopt.x() %q1_2 : !mqtopt.Qubit

        // CHECK: %[[Q0_3:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q0_2]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %q0_3, %c0_0 = "mqtopt.measure"(%q0_2) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Q1_4:.*]], %[[C1_0:.*]] = "mqtopt.measure"(%[[Q1_3]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %q1_4, %c1_0 = "mqtopt.measure"(%q1_3) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

        // CHECK: %[[Reg_3:.*]] = "mqtopt.insertQubit"(%[[Reg_2]], %[[Q0_3]]) <{index_attr = 0 : i64}>
        %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_3) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q1_4]]) <{index_attr = 1 : i64}>
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_4) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        // CHECK: return %[[Reg_4]], %[[C0_0]], %[[C1_0]]
        return %reg_4, %c0_0, %c1_0 : !mqtopt.QubitRegister, i1, i1
    }
}
