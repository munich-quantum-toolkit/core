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

// -----
// This test checks if single target unitary gates are parsed correctly by the QuantumComputation roundtrip pass.
module {
    // CHECK-LABEL: func.func @testSingleTargetUnitaryGates()
    func.func @testSingleTargetUnitaryGates() -> (!mqtopt.QubitRegister, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
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
        // CHECK: %[[Q0_14:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q0_13]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[Reg_1]], %[[Q0_14]]) <{index_attr = 0 : i64}>
        // CHECK: return %[[Reg_2]], %[[C0_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister

        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

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

        %q0_14, %c0_0 = "mqtopt.measure"(%q0_13) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q0_14) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_2, %c0_0 : !mqtopt.QubitRegister, i1
    }
}

// -----
// This test checks if two target unitary gates are parsed correctly parsed correctly by the QuantumComputation roundtrip pass.
module {
    // CHECK-LABEL: func.func @testTwoTargetUnitaryGates()
    func.func @testTwoTargetUnitaryGates() -> (!mqtopt.QubitRegister, i1, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
        // CHECK: %[[Q01_1:.*]]:2 = mqtopt.swap() %[[Q0_0]], %[[Q1_0]] : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q01_2:.*]]:2 = mqtopt.iswap() %[[Q01_1]]#0, %[[Q01_1]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q01_3:.*]]:2 = mqtopt.iswapdg() %[[Q01_2]]#0, %[[Q01_2]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q01_4:.*]]:2 = mqtopt.peres() %[[Q01_3]]#0, %[[Q01_3]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q01_5:.*]]:2 = mqtopt.peresdg() %[[Q01_4]]#0, %[[Q01_4]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q01_6:.*]]:2 = mqtopt.dcx() %[[Q01_5]]#0, %[[Q01_5]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q01_7:.*]]:2 = mqtopt.ecr() %[[Q01_6]]#0, %[[Q01_6]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q0_8:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q01_7]]#0) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Q1_8:.*]], %[[C1_0:.*]] = "mqtopt.measure"(%[[Q01_7]]#1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_3:.*]] = "mqtopt.insertQubit"(%[[Reg_2]], %[[Q0_8]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q1_8]]) <{index_attr = 1 : i64}>
        // CHECK: return %[[Reg_4]], %[[C0_0]], %[[C1_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1, %q1_1 = mqtopt.swap() %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
        %q0_2, %q1_2 = mqtopt.iswap() %q0_1, %q1_1 : !mqtopt.Qubit, !mqtopt.Qubit
        %q0_3, %q1_3 = mqtopt.iswapdg() %q0_2, %q1_2 : !mqtopt.Qubit, !mqtopt.Qubit
        %q0_4, %q1_4 = mqtopt.peres() %q0_3, %q1_3 : !mqtopt.Qubit, !mqtopt.Qubit
        %q0_5, %q1_5 = mqtopt.peresdg() %q0_4, %q1_4 : !mqtopt.Qubit, !mqtopt.Qubit
        %q0_6, %q1_6 = mqtopt.dcx() %q0_5, %q1_5 : !mqtopt.Qubit, !mqtopt.Qubit
        %q0_7, %q1_7 = mqtopt.ecr() %q0_6, %q1_6 : !mqtopt.Qubit, !mqtopt.Qubit

        %q0_8, %c0_0 = "mqtopt.measure"(%q0_7) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %q1_8, %c1_0 = "mqtopt.measure"(%q1_7) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

        %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_8) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_8) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_4, %c0_0, %c1_0 : !mqtopt.QubitRegister, i1, i1
    }
}

// -----
// This test checks if a u rotation gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testUGate()
    func.func @testUGate() -> (!mqtopt.QubitRegister, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>

        // CHECK: %[[Q0_1:.*]] = mqtopt.u( static [1.000000e-01, 2.000000e-01, 3.000000e-01]) %[[Q0_0]] : !mqtopt.Qubit

        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q0_1]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[Reg_1]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: return %[[Reg_2]], %[[C0_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1 = mqtopt.u( static [1.000000e-01, 2.000000e-01, 3.000000e-01]) %q0_0 : !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_2, %c0_0 : !mqtopt.QubitRegister, i1
    }
}

// -----
// This test checks if a u2 rotation gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testU2Gate()
    func.func @testU2Gate() -> (!mqtopt.QubitRegister, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>

        // CHECK: %[[Q0_1:.*]] = mqtopt.u2( static [1.000000e-01, 2.000000e-01]) %[[Q0_0]] : !mqtopt.Qubit

        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q0_1]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[Reg_1]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: return %[[Reg_2]], %[[C0_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1 = mqtopt.u2( static [1.000000e-01, 2.000000e-01]) %q0_0 : !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_2, %c0_0 : !mqtopt.QubitRegister, i1
    }
}

// -----
// This test checks if a p rotation gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testPGate()
    func.func @testPGate() -> (!mqtopt.QubitRegister, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>

        // CHECK: %[[Q0_1:.*]] = mqtopt.p( static [1.000000e-01]) %[[Q0_0]] : !mqtopt.Qubit

        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q0_1]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[Reg_1]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: return %[[Reg_2]], %[[C0_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1 = mqtopt.p( static [1.000000e-01]) %q0_0 : !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_2, %c0_0 : !mqtopt.QubitRegister, i1
    }
}

// -----
// This test checks if a rx rotation gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testRXGate()
    func.func @testRXGate() -> (!mqtopt.QubitRegister, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>

        // CHECK: %[[Q0_1:.*]] = mqtopt.rx( static [1.000000e-01]) %[[Q0_0]] : !mqtopt.Qubit

        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q0_1]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[Reg_1]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: return %[[Reg_2]], %[[C0_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1 = mqtopt.rx( static [1.000000e-01]) %q0_0 : !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_2, %c0_0 : !mqtopt.QubitRegister, i1
    }
}

// -----
// This test checks if a ry rotation gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testRYGate()
    func.func @testRYGate() -> (!mqtopt.QubitRegister, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>

        // CHECK: %[[Q0_1:.*]] = mqtopt.ry( static [1.000000e-01]) %[[Q0_0]] : !mqtopt.Qubit

        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q0_1]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[Reg_1]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: return %[[Reg_2]], %[[C0_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1 = mqtopt.ry( static [1.000000e-01]) %q0_0 : !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_2, %c0_0 : !mqtopt.QubitRegister, i1
    }
}

// -----
// This test checks if a rz rotation gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testRZGate()
    func.func @testRZGate() -> (!mqtopt.QubitRegister, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>

        // CHECK: %[[Q0_1:.*]] = mqtopt.rz( static [1.000000e-01]) %[[Q0_0]] : !mqtopt.Qubit

        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q0_1]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[Reg_1]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: return %[[Reg_2]], %[[C0_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1 = mqtopt.rz( static [1.000000e-01]) %q0_0 : !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_2, %c0_0 : !mqtopt.QubitRegister, i1
    }
}

// -----
// This test checks if a rxx rotation gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testRXXGate()
    func.func @testRXXGate() -> (!mqtopt.QubitRegister, i1, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>

        // CHECK: %[[Q01_1:.*]]:2 = mqtopt.rxx( static [1.000000e-01]) %[[Q0_0]], %[[Q1_0]] : !mqtopt.Qubit, !mqtopt.Qubit

        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q01_1]]#0) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Q1_2:.*]], %[[C1_0:.*]] = "mqtopt.measure"(%[[Q01_1]]#1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_3:.*]] = "mqtopt.insertQubit"(%[[Reg_2]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q1_2]]) <{index_attr = 1 : i64}>
        // CHECK: return %[[Reg_4]], %[[C0_0]], %[[C1_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1, %q1_1 = mqtopt.rxx( static [1.000000e-01]) %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %q1_2, %c1_0 = "mqtopt.measure"(%q1_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_2) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_4, %c0_0, %c1_0 : !mqtopt.QubitRegister, i1, i1
    }
}

// -----
// This test checks if a ryy rotation gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testRYYGate()
    func.func @testRYYGate() -> (!mqtopt.QubitRegister, i1, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>

        // CHECK: %[[Q01_1:.*]]:2 = mqtopt.ryy( static [1.000000e-01]) %[[Q0_0]], %[[Q1_0]] : !mqtopt.Qubit, !mqtopt.Qubit

        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q01_1]]#0) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Q1_2:.*]], %[[C1_0:.*]] = "mqtopt.measure"(%[[Q01_1]]#1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_3:.*]] = "mqtopt.insertQubit"(%[[Reg_2]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q1_2]]) <{index_attr = 1 : i64}>
        // CHECK: return %[[Reg_4]], %[[C0_0]], %[[C1_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1, %q1_1 = mqtopt.ryy( static [1.000000e-01]) %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %q1_2, %c1_0 = "mqtopt.measure"(%q1_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_2) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_4, %c0_0, %c1_0 : !mqtopt.QubitRegister, i1, i1
    }
}

// -----
// This test checks if a rzz rotation gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testRZZGate()
    func.func @testRZZGate() -> (!mqtopt.QubitRegister, i1, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>

        // CHECK: %[[Q01_1:.*]]:2 = mqtopt.rzz( static [1.000000e-01]) %[[Q0_0]], %[[Q1_0]] : !mqtopt.Qubit, !mqtopt.Qubit

        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q01_1]]#0) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Q1_2:.*]], %[[C1_0:.*]] = "mqtopt.measure"(%[[Q01_1]]#1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_3:.*]] = "mqtopt.insertQubit"(%[[Reg_2]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q1_2]]) <{index_attr = 1 : i64}>
        // CHECK: return %[[Reg_4]], %[[C0_0]], %[[C1_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1, %q1_1 = mqtopt.rzz( static [1.000000e-01]) %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %q1_2, %c1_0 = "mqtopt.measure"(%q1_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_2) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_4, %c0_0, %c1_0 : !mqtopt.QubitRegister, i1, i1
    }
}

// -----
// This test checks if a rzx rotation gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testRZXGate()
    func.func @testRZXGate() -> (!mqtopt.QubitRegister, i1, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>

        // CHECK: %[[Q01_1:.*]]:2 = mqtopt.rzx( static [1.000000e-01]) %[[Q0_0]], %[[Q1_0]] : !mqtopt.Qubit, !mqtopt.Qubit

        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q01_1]]#0) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Q1_2:.*]], %[[C1_0:.*]] = "mqtopt.measure"(%[[Q01_1]]#1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_3:.*]] = "mqtopt.insertQubit"(%[[Reg_2]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q1_2]]) <{index_attr = 1 : i64}>
        // CHECK: return %[[Reg_4]], %[[C0_0]], %[[C1_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1, %q1_1 = mqtopt.rzx( static [1.000000e-01]) %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %q1_2, %c1_0 = "mqtopt.measure"(%q1_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_2) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_4, %c0_0, %c1_0 : !mqtopt.QubitRegister, i1, i1
    }
}

// -----
// This test checks if a xxminusyy rotation gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testXXminusYYGate()
    func.func @testXXminusYYGate() -> (!mqtopt.QubitRegister, i1, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>

        // CHECK: %[[Q01_1:.*]]:2 = mqtopt.xxminusyy( static [1.000000e-01, 2.000000e-01]) %[[Q0_0]], %[[Q1_0]] : !mqtopt.Qubit, !mqtopt.Qubit

        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q01_1]]#0) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Q1_2:.*]], %[[C1_0:.*]] = "mqtopt.measure"(%[[Q01_1]]#1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_3:.*]] = "mqtopt.insertQubit"(%[[Reg_2]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q1_2]]) <{index_attr = 1 : i64}>
        // CHECK: return %[[Reg_4]], %[[C0_0]], %[[C1_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1, %q1_1 = mqtopt.xxminusyy( static [1.000000e-01, 2.000000e-01]) %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %q1_2, %c1_0 = "mqtopt.measure"(%q1_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_2) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_4, %c0_0, %c1_0 : !mqtopt.QubitRegister, i1, i1
    }
}

// -----
// This test checks if a xxplusyy rotation gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testXXplusYYGate()
    func.func @testXXplusYYGate() -> (!mqtopt.QubitRegister, i1, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>

        // CHECK: %[[Q01_1:.*]]:2 = mqtopt.xxplusyy( static [1.000000e-01, 2.000000e-01]) %[[Q0_0]], %[[Q1_0]] : !mqtopt.Qubit, !mqtopt.Qubit

        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q01_1]]#0) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Q1_2:.*]], %[[C1_0:.*]] = "mqtopt.measure"(%[[Q01_1]]#1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_3:.*]] = "mqtopt.insertQubit"(%[[Reg_2]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q1_2]]) <{index_attr = 1 : i64}>
        // CHECK: return %[[Reg_4]], %[[C0_0]], %[[C1_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1, %q1_1 = mqtopt.xxplusyy( static [1.000000e-01, 2.000000e-01]) %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %q1_2, %c1_0 = "mqtopt.measure"(%q1_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_2) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_4, %c0_0, %c1_0 : !mqtopt.QubitRegister, i1, i1
    }
}