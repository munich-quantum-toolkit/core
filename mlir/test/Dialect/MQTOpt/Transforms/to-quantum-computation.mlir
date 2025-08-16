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
// This test checks if the identity gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testIGate()
    func.func @testIGate() -> (!mqtopt.QubitRegister, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Q0_1:.*]] = mqtopt.i() %[[Q0_0]] : !mqtopt.Qubit
        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q0_1]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[Reg_1]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: return %[[Reg_2]], %[[C0_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1 = mqtopt.i() %q0_0 : !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_2, %c0_0 : !mqtopt.QubitRegister, i1
    }
}

// -----
// This test checks if the Hadamard gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testHGate()
    func.func @testHGate() -> (!mqtopt.QubitRegister, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Q0_1:.*]] = mqtopt.h() %[[Q0_0]] : !mqtopt.Qubit
        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q0_1]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[Reg_1]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: return %[[Reg_2]], %[[C0_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_2, %c0_0 : !mqtopt.QubitRegister, i1
    }
}

// -----
// This test checks if the x gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testXGate()
    func.func @testXGate() -> (!mqtopt.QubitRegister, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Q0_1:.*]] = mqtopt.x() %[[Q0_0]] : !mqtopt.Qubit
        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q0_1]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[Reg_1]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: return %[[Reg_2]], %[[C0_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1 = mqtopt.x() %q0_0 : !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_2, %c0_0 : !mqtopt.QubitRegister, i1
    }
}

// -----
// This test checks if the y gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testYGate()
    func.func @testYGate() -> (!mqtopt.QubitRegister, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Q0_1:.*]] = mqtopt.y() %[[Q0_0]] : !mqtopt.Qubit
        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q0_1]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[Reg_1]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: return %[[Reg_2]], %[[C0_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1 = mqtopt.y() %q0_0 : !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_2, %c0_0 : !mqtopt.QubitRegister, i1
    }
}

// -----
// This test checks if the z gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testZGate()
    func.func @testZGate() -> (!mqtopt.QubitRegister, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Q0_1:.*]] = mqtopt.z() %[[Q0_0]] : !mqtopt.Qubit
        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q0_1]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[Reg_1]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: return %[[Reg_2]], %[[C0_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1 = mqtopt.z() %q0_0 : !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_2, %c0_0 : !mqtopt.QubitRegister, i1
    }
}

// -----
// This test checks if the s gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testSGate()
    func.func @testSGate() -> (!mqtopt.QubitRegister, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Q0_1:.*]] = mqtopt.s() %[[Q0_0]] : !mqtopt.Qubit
        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q0_1]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[Reg_1]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: return %[[Reg_2]], %[[C0_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1 = mqtopt.s() %q0_0 : !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_2, %c0_0 : !mqtopt.QubitRegister, i1
    }
}

// -----
// This test checks if the sdg gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testSdgGate()
    func.func @testSdgGate() -> (!mqtopt.QubitRegister, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Q0_1:.*]] = mqtopt.sdg() %[[Q0_0]] : !mqtopt.Qubit
        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q0_1]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[Reg_1]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: return %[[Reg_2]], %[[C0_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1 = mqtopt.sdg() %q0_0 : !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_2, %c0_0 : !mqtopt.QubitRegister, i1
    }
}

// -----
// This test checks if the t gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testTGate()
    func.func @testTGate() -> (!mqtopt.QubitRegister, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Q0_1:.*]] = mqtopt.t() %[[Q0_0]] : !mqtopt.Qubit
        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q0_1]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[Reg_1]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: return %[[Reg_2]], %[[C0_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1 = mqtopt.t() %q0_0 : !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_2, %c0_0 : !mqtopt.QubitRegister, i1
    }
}

// -----
// This test checks if the tdg gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testTdgGate()
    func.func @testTdgGate() -> (!mqtopt.QubitRegister, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Q0_1:.*]] = mqtopt.tdg() %[[Q0_0]] : !mqtopt.Qubit
        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q0_1]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[Reg_1]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: return %[[Reg_2]], %[[C0_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1 = mqtopt.tdg() %q0_0 : !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_2, %c0_0 : !mqtopt.QubitRegister, i1
    }
}

// -----
// This test checks if the v gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testVGate()
    func.func @testVGate() -> (!mqtopt.QubitRegister, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Q0_1:.*]] = mqtopt.v() %[[Q0_0]] : !mqtopt.Qubit
        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q0_1]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[Reg_1]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: return %[[Reg_2]], %[[C0_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1 = mqtopt.v() %q0_0 : !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_2, %c0_0 : !mqtopt.QubitRegister, i1
    }
}

// -----
// This test checks if the vdg gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testVdgGate()
    func.func @testVdgGate() -> (!mqtopt.QubitRegister, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Q0_1:.*]] = mqtopt.vdg() %[[Q0_0]] : !mqtopt.Qubit
        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q0_1]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[Reg_1]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: return %[[Reg_2]], %[[C0_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1 = mqtopt.vdg() %q0_0 : !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_2, %c0_0 : !mqtopt.QubitRegister, i1
    }
}

// -----
// This test checks if the sx gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testSXGate()
    func.func @testSXGate() -> (!mqtopt.QubitRegister, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Q0_1:.*]] = mqtopt.sx() %[[Q0_0]] : !mqtopt.Qubit
        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q0_1]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[Reg_1]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: return %[[Reg_2]], %[[C0_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1 = mqtopt.sx() %q0_0 : !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_2, %c0_0 : !mqtopt.QubitRegister, i1
    }
}

// -----
// This test checks if the sxdg gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testSXdgGate()
    func.func @testSXdgGate() -> (!mqtopt.QubitRegister, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Q0_1:.*]] = mqtopt.sxdg() %[[Q0_0]] : !mqtopt.Qubit
        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q0_1]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[Reg_1]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: return %[[Reg_2]], %[[C0_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1 = mqtopt.sxdg() %q0_0 : !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_2, %c0_0 : !mqtopt.QubitRegister, i1
    }
}

// -----
// This test checks if the swap gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testSwapGate()
    func.func @testSwapGate() -> (!mqtopt.QubitRegister, i1, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
        // CHECK: %[[Q01_1:.*]]:2 = mqtopt.swap() %[[Q0_0]], %[[Q1_0]] : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q01_1]]#0) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Q1_2:.*]], %[[C1_0:.*]] = "mqtopt.measure"(%[[Q01_1]]#1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_3:.*]] = "mqtopt.insertQubit"(%[[Reg_2]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q1_2]]) <{index_attr = 1 : i64}>
        // CHECK: return %[[Reg_4]], %[[C0_0]], %[[C1_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1, %q1_1 = mqtopt.swap() %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %q1_2, %c1_0 = "mqtopt.measure"(%q1_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_2) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_4, %c0_0, %c1_0 : !mqtopt.QubitRegister, i1, i1
    }
}

// -----
// This test checks if the iswap gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testISwapGate()
    func.func @testISwapGate() -> (!mqtopt.QubitRegister, i1, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
        // CHECK: %[[Q01_1:.*]]:2 = mqtopt.iswap() %[[Q0_0]], %[[Q1_0]] : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q01_1]]#0) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Q1_2:.*]], %[[C1_0:.*]] = "mqtopt.measure"(%[[Q01_1]]#1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_3:.*]] = "mqtopt.insertQubit"(%[[Reg_2]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q1_2]]) <{index_attr = 1 : i64}>
        // CHECK: return %[[Reg_4]], %[[C0_0]], %[[C1_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1, %q1_1 = mqtopt.iswap() %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %q1_2, %c1_0 = "mqtopt.measure"(%q1_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_2) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_4, %c0_0, %c1_0 : !mqtopt.QubitRegister, i1, i1
    }
}

// -----
// This test checks if the iswapdg gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testISwapdgGate()
    func.func @testISwapdgGate() -> (!mqtopt.QubitRegister, i1, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
        // CHECK: %[[Q01_1:.*]]:2 = mqtopt.iswapdg() %[[Q0_0]], %[[Q1_0]] : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q01_1]]#0) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Q1_2:.*]], %[[C1_0:.*]] = "mqtopt.measure"(%[[Q01_1]]#1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_3:.*]] = "mqtopt.insertQubit"(%[[Reg_2]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q1_2]]) <{index_attr = 1 : i64}>
        // CHECK: return %[[Reg_4]], %[[C0_0]], %[[C1_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1, %q1_1 = mqtopt.iswapdg() %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %q1_2, %c1_0 = "mqtopt.measure"(%q1_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_2) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_4, %c0_0, %c1_0 : !mqtopt.QubitRegister, i1, i1
    }
}

// -----
// This test checks if the peres gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testPeresGate()
    func.func @testPeresGate() -> (!mqtopt.QubitRegister, i1, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
        // CHECK: %[[Q01_1:.*]]:2 = mqtopt.peres() %[[Q0_0]], %[[Q1_0]] : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q01_1]]#0) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Q1_2:.*]], %[[C1_0:.*]] = "mqtopt.measure"(%[[Q01_1]]#1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_3:.*]] = "mqtopt.insertQubit"(%[[Reg_2]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q1_2]]) <{index_attr = 1 : i64}>
        // CHECK: return %[[Reg_4]], %[[C0_0]], %[[C1_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1, %q1_1 = mqtopt.peres() %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %q1_2, %c1_0 = "mqtopt.measure"(%q1_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_2) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_4, %c0_0, %c1_0 : !mqtopt.QubitRegister, i1, i1
    }
}

// -----
// This test checks if the peresdg gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testPeresdgGate()
    func.func @testPeresdgGate() -> (!mqtopt.QubitRegister, i1, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
        // CHECK: %[[Q01_1:.*]]:2 = mqtopt.peresdg() %[[Q0_0]], %[[Q1_0]] : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q01_1]]#0) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Q1_2:.*]], %[[C1_0:.*]] = "mqtopt.measure"(%[[Q01_1]]#1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_3:.*]] = "mqtopt.insertQubit"(%[[Reg_2]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q1_2]]) <{index_attr = 1 : i64}>
        // CHECK: return %[[Reg_4]], %[[C0_0]], %[[C1_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1, %q1_1 = mqtopt.peresdg() %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %q1_2, %c1_0 = "mqtopt.measure"(%q1_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_2) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_4, %c0_0, %c1_0 : !mqtopt.QubitRegister, i1, i1
    }
}

// -----
// This test checks if the dcx gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testDCXGate()
    func.func @testDCXGate() -> (!mqtopt.QubitRegister, i1, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
        // CHECK: %[[Q01_1:.*]]:2 = mqtopt.dcx() %[[Q0_0]], %[[Q1_0]] : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q01_1]]#0) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Q1_2:.*]], %[[C1_0:.*]] = "mqtopt.measure"(%[[Q01_1]]#1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_3:.*]] = "mqtopt.insertQubit"(%[[Reg_2]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q1_2]]) <{index_attr = 1 : i64}>
        // CHECK: return %[[Reg_4]], %[[C0_0]], %[[C1_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1, %q1_1 = mqtopt.dcx() %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %q1_2, %c1_0 = "mqtopt.measure"(%q1_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_2) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        return %reg_4, %c0_0, %c1_0 : !mqtopt.QubitRegister, i1, i1
    }
}

// -----
// This test checks if the ecr gate is parsed correctly by the QuantumComputation roundtrip pass.

module {
    // CHECK-LABEL: func.func @testECRGate()
    func.func @testECRGate() -> (!mqtopt.QubitRegister, i1, i1) {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}>
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
        // CHECK: %[[Q01_1:.*]]:2 = mqtopt.ecr() %[[Q0_0]], %[[Q1_0]] : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q0_2:.*]], %[[C0_0:.*]] = "mqtopt.measure"(%[[Q01_1]]#0) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Q1_2:.*]], %[[C1_0:.*]] = "mqtopt.measure"(%[[Q01_1]]#1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_3:.*]] = "mqtopt.insertQubit"(%[[Reg_2]], %[[Q0_2]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q1_2]]) <{index_attr = 1 : i64}>
        // CHECK: return %[[Reg_4]], %[[C0_0]], %[[C1_0]]

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %q0_1, %q1_1 = mqtopt.ecr() %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit

        %q0_2, %c0_0 = "mqtopt.measure"(%q0_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %q1_2, %c1_0 = "mqtopt.measure"(%q1_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_2) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

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
