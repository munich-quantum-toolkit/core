// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --mqt-core-round-trip | FileCheck %s

// -----
// This test checks that the QuantumComputation roundtrip works generally.

module {
    // CHECK-LABEL: func @circuit()
    func.func @circuit() -> (!mqtopt.QubitRegister, i1, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
        // CHECK: %[[Q0_1:.*]] = mqtopt.x() %[[Q0_0]] : !mqtopt.Qubit
        // CHECK: %[[Q1_1:.*]] = mqtopt.x() %[[Q1_0]] : !mqtopt.Qubit
        // CHECK: %[[Q0_2:.*]], %[[Q1_2:.*]] = mqtopt.x() %[[Q0_1]] ctrl %[[Q1_1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
        // CHECK: %[[Q1_3:.*]] = mqtopt.x() %[[Q1_2]] : !mqtopt.Qubit
        // CHECK: %[[Q0_3:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q0_2]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Q1_4:.*]], %[[C1_0:.*]] = "mqtopt.measure"(%[[Q1_3]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_3:.*]] = "mqtopt.insertQubit"(%[[Reg_2]], %[[Q0_3]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q1_4]]) <{index_attr = 1 : i64}>
        // CHECK: return %[[Reg_4]], %[[C0_0]], %[[C1_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1 = mqtopt.x() %q0_0 : !mqtopt.Qubit
        %q1_1 = mqtopt.x() %q1_0 : !mqtopt.Qubit

        %q0_2, %q1_2 = mqtopt.x() %q0_1 ctrl %q1_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit

        %q1_3 = mqtopt.x() %q1_2 : !mqtopt.Qubit

        %q0_3, %c0_0 = "mqtopt.measure"(%q0_2) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %q1_4, %c1_0 = "mqtopt.measure"(%q1_3) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

        %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_3) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_4) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_4, %c0_0, %c1_0 : !mqtopt.QubitRegister, i1, i1
    }
}

// -----
// This test checks that the QuantumComputation roundtrip works with negative controls.

module {
    // CHECK-LABEL: func @negativeControls()
    func.func @negativeControls() -> (!mqtopt.QubitRegister, i1, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
        // CHECK: %[[Q0_1:.*]] = mqtopt.x() %[[Q0_0]] : !mqtopt.Qubit
        // CHECK: %[[Q1_1:.*]] = mqtopt.h() %[[Q1_0]] : !mqtopt.Qubit
        // CHECK: %[[Q0_2:.*]], %[[Q1_2:.*]] = mqtopt.x() %[[Q0_1]] nctrl %[[Q1_1]] : !mqtopt.Qubit nctrl !mqtopt.Qubit
        // CHECK: %[[Q0_3:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q0_2]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Q1_3:.*]], %[[C1_0:.*]] = "mqtopt.measure"(%[[Q1_2]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_3:.*]] = "mqtopt.insertQubit"(%[[Reg_2]], %[[Q0_3]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q1_3]]) <{index_attr = 1 : i64}>
        // CHECK: return %[[Reg_4]], %[[C0_0]], %[[C1_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1 = mqtopt.x() %q0_0 : !mqtopt.Qubit
        %q1_1 = mqtopt.h() %q1_0 : !mqtopt.Qubit

        %q0_2, %q1_2 = mqtopt.x() %q0_1 nctrl %q1_1 : !mqtopt.Qubit nctrl !mqtopt.Qubit

        %q0_3, %c0_0 = "mqtopt.measure"(%q0_2) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %q1_3, %c1_0 = "mqtopt.measure"(%q1_2) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

        %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_3) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_3) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_4, %c0_0, %c1_0 : !mqtopt.QubitRegister, i1, i1
    }
}
