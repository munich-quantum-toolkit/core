// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// -----
// This test checks if the AllocOp is parsed and handled correctly using a static attribute.
module {
    // CHECK-LABEL: func.func @testAllocOpAttribute
    func.func @testAllocOpAttribute() {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}>

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        return
    }
}

// -----
// This test checks if the AllocOp is parsed and handled correctly using a dynamic operand.
module {
    // CHECK-LABEL: func.func @testAllocOpOperand
    func.func @testAllocOpOperand() {
        // CHECK: %[[Size:.*]] = arith.constant 2
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"(%[[Size]]) : (i64) -> !mqtopt.QubitRegister

        %size = arith.constant 2 : i64
        %reg_0 = "mqtopt.allocQubitRegister"(%size) : (i64) -> !mqtopt.QubitRegister
        return
    }
}

// -----
// This test checks if the DeallocOp is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testDeallocOp
    func.func @testDeallocOp() {
        // CHECK: "mqtopt.deallocQubitRegister"(%[[ANY:.*]])

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%reg_0) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if the ExtractOp is parsed and handled correctly using a static attribute.
module {
    // CHECK-LABEL: func.func @testExtractOpAttribute
    func.func @testExtractOpAttribute() {
        // CHECK: %[[Reg_1:.*]], %[[Q_0:.*]] = "mqtopt.extractQubit"(%[[ANY:.*]]) <{index_attr = 0 : i64}>

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        return
    }
}

// -----
// This test checks if the ExtractOp is parsed and handled correctly using a dynamic operand.
module {
    // CHECK-LABEL: func.func @testExtractOpOperand
    func.func @testExtractOpOperand() {
        // CHECK: %[[Index:.*]] = arith.constant 0
        // CHECK: %[[Reg_1:.*]], %[[Q_0:.*]] = "mqtopt.extractQubit"(%[[ANY:.*]], %[[Index]]) : (!mqtopt.QubitRegister, i64) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %index = arith.constant 0 : i64
        %reg_1, %q_0 = "mqtopt.extractQubit"(%reg_0, %index) : (!mqtopt.QubitRegister, i64) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        return
    }
}

// -----
// This test checks if the InsertOp is parsed and handled correctly using a static attribute.
module {
    // CHECK-LABEL: func.func @testInsertOpAttribute
    func.func @testInsertOpAttribute() {
        // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[ANY:.*]], %[[ANY:.*]])  <{index_attr = 0 : i64}>

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        return
    }
}

// -----
// This test checks if the InsertOp is parsed and handled correctly using a dynamic operand.
module {
    // CHECK-LABEL: func.func @testInsertOpOperand
    func.func @testInsertOpOperand() {
        // CHECK: %[[Index:.*]] = arith.constant 0
        // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[ANY:.*]], %[[ANY:.*]], %[[Index]])  : (!mqtopt.QubitRegister, !mqtopt.Qubit, i64) -> !mqtopt.QubitRegister

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %index = arith.constant 0 : i64
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q_0, %index) : (!mqtopt.QubitRegister, !mqtopt.Qubit, i64) -> !mqtopt.QubitRegister
        return
    }
}

// -----
// This test checks that all resources defined in the MQTOpt dialect are parsed and handled correctly using dynamic operands.
module {
    // CHECK-LABEL: func.func @testAllResourcesUsingOperands
    func.func @testAllResourcesUsingOperands() {
        // CHECK: %[[Size:.*]] = arith.constant 1 : i64
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"(%[[Size]]) : (i64) -> !mqtopt.QubitRegister
        // CHECK: %[[Index:.*]] = arith.constant 0 : i64
        // CHECK: %[[Reg_1:.*]], %[[Q_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]], %[[Index]]) : (!mqtopt.QubitRegister, i64) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[Reg_1]], %[[Q_0]], %[[Index]])  : (!mqtopt.QubitRegister, !mqtopt.Qubit, i64) -> !mqtopt.QubitRegister
        // CHECK: "mqtopt.deallocQubitRegister"(%[[Reg_2]])

        %size = arith.constant 1 : i64
        %reg_0 = "mqtopt.allocQubitRegister"(%size) : (i64) -> !mqtopt.QubitRegister
        %index = arith.constant 0 : i64
        %reg_1, %q_0 = "mqtopt.extractQubit"(%reg_0, %index) : (!mqtopt.QubitRegister, i64) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q_0, %index) : (!mqtopt.QubitRegister, !mqtopt.Qubit, i64) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%reg_2) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if the MeasureOp is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testMeasureOp
    func.func @testMeasureOp() {
        // CHECK: %[[Q_1:.*]], [[M0_0:.*]] = "mqtopt.measure"(%[[ANY:.*]])

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %q_1, %m0_0 = "mqtopt.measure"(%q_0) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q_1) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%reg_2) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if the MeasureOp applied to multiple qubits is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testMeasureOpOnMultipleInputs
    func.func @testMeasureOpOnMultipleInputs() {
        // CHECK: %[[Q01_1:.*]]:2, [[M01_0:.*]]:2 = "mqtopt.measure"(%[[ANY:.*]], %[[ANY:.*]])

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %q01_1:2, %m01_0:2 = "mqtopt.measure"(%q0_0, %q1_0) : (!mqtopt.Qubit, !mqtopt.Qubit) -> (!mqtopt.Qubit, !mqtopt.Qubit, i1, i1)
        %reg_3 = "mqtopt.insertQubit"(%reg_2, %q01_1#0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q01_1#1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%reg_4) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if global phase operations are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testGPhaseOp
    func.func @testGPhaseOp() {
        // CHECK: %[[C0_F64:.*]] = arith.constant 3.000000e-01
        // CHECK: %[[Q_1:.*]] = mqtopt.gphase(%[[C0_F64]]) ctrl %[[ANY:.*]] : !mqtopt.Qubit

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %c0_f64 = arith.constant 3.000000e-01 : f64
        %q_1 = mqtopt.gphase(%c0_f64) ctrl %q_0 : ctrl !mqtopt.Qubit
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q_1) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%reg_2) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if single qubit gates are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testSingleQubitOp
    func.func @testSingleQubitOp() {
        // CHECK: %[[Q_1:.*]] = mqtopt.i() %[[ANY:.*]] : !mqtopt.Qubit
        // CHECK: %[[Q_2:.*]] = mqtopt.h() %[[Q_1]] : !mqtopt.Qubit
        // CHECK: %[[Q_3:.*]] = mqtopt.x() %[[Q_2]] : !mqtopt.Qubit
        // CHECK: %[[Q_4:.*]] = mqtopt.y() %[[Q_3]] : !mqtopt.Qubit
        // CHECK: %[[Q_5:.*]] = mqtopt.z() %[[Q_4]] : !mqtopt.Qubit
        // CHECK: %[[Q_6:.*]] = mqtopt.s() %[[Q_5]] : !mqtopt.Qubit
        // CHECK: %[[Q_7:.*]] = mqtopt.sdg() %[[Q_6]] : !mqtopt.Qubit
        // CHECK: %[[Q_8:.*]] = mqtopt.t() %[[Q_7]] : !mqtopt.Qubit
        // CHECK: %[[Q_9:.*]] = mqtopt.tdg() %[[Q_8]] : !mqtopt.Qubit
        // CHECK: %[[Q_10:.*]] = mqtopt.v() %[[Q_9]] : !mqtopt.Qubit
        // CHECK: %[[Q_11:.*]] = mqtopt.vdg() %[[Q_10]] : !mqtopt.Qubit
        // CHECK: %[[Q_12:.*]] = mqtopt.sx() %[[Q_11]] : !mqtopt.Qubit
        // CHECK: %[[Q_13:.*]] = mqtopt.sxdg() %[[Q_12]] : !mqtopt.Qubit

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %q_1 = mqtopt.i() %q_0 : !mqtopt.Qubit
        %q_2 = mqtopt.h() %q_1 : !mqtopt.Qubit
        %q_3 = mqtopt.x() %q_2 : !mqtopt.Qubit
        %q_4 = mqtopt.y() %q_3 : !mqtopt.Qubit
        %q_5 = mqtopt.z() %q_4 : !mqtopt.Qubit
        %q_6 = mqtopt.s() %q_5 : !mqtopt.Qubit
        %q_7 = mqtopt.sdg() %q_6 : !mqtopt.Qubit
        %q_8 = mqtopt.t() %q_7 : !mqtopt.Qubit
        %q_9 = mqtopt.tdg() %q_8 : !mqtopt.Qubit
        %q_10 = mqtopt.v() %q_9 : !mqtopt.Qubit
        %q_11 = mqtopt.vdg() %q_10 : !mqtopt.Qubit
        %q_12 = mqtopt.sx() %q_11 : !mqtopt.Qubit
        %q_13 = mqtopt.sxdg() %q_12 : !mqtopt.Qubit
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q_13) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%reg_2) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if parameterized single qubit gates are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testSingleQubitRotationOp
    func.func @testSingleQubitRotationOp() {
        // CHECK: %[[C0_F64:.*]] = arith.constant 3.000000e-01
        // CHECK: %[[Q_1:.*]] = mqtopt.u(%[[C0_F64]], %[[C0_F64]], %[[C0_F64]]) %[[ANY:.*]] : !mqtopt.Qubit
        // CHECK: %[[Q_2:.*]] = mqtopt.u2(%[[C0_F64]], %[[C0_F64]]) %[[Q_1]] : !mqtopt.Qubit
        // CHECK: %[[Q_3:.*]] = mqtopt.p(%[[C0_F64]]) %[[Q_2]] : !mqtopt.Qubit
        // CHECK: %[[Q_4:.*]] = mqtopt.rx(%[[C0_F64]]) %[[Q_3]] : !mqtopt.Qubit
        // CHECK: %[[Q_5:.*]] = mqtopt.ry(%[[C0_F64]]) %[[Q_4]] : !mqtopt.Qubit
        // CHECK: %[[Q_6:.*]] = mqtopt.rz(%[[C0_F64]]) %[[Q_5]] : !mqtopt.Qubit

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %c0_f64 = arith.constant 3.000000e-01 : f64
        %q_1 = mqtopt.u(%c0_f64, %c0_f64, %c0_f64) %q_0 : !mqtopt.Qubit
        %q_2 = mqtopt.u2(%c0_f64, %c0_f64) %q_1 : !mqtopt.Qubit
        %q_3 = mqtopt.p(%c0_f64) %q_2 : !mqtopt.Qubit
        %q_4 = mqtopt.rx(%c0_f64) %q_3 : !mqtopt.Qubit
        %q_5 = mqtopt.ry(%c0_f64) %q_4 : !mqtopt.Qubit
        %q_6 = mqtopt.rz(%c0_f64) %q_5 : !mqtopt.Qubit
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q_6) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%reg_2) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if controlled parameterized single qubit gates are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testControlledSingleQubitRotationOp
    func.func @testControlledSingleQubitRotationOp() {
        // CHECK: %[[C0_F64:.*]] = arith.constant 3.000000e-01
        // CHECK: %[[Q0_1:.*]], %[[Q1_1:.*]] = mqtopt.u(%[[C0_F64]], %[[C0_F64]], %[[C0_F64]]) %[[ANY:.*]] ctrl %[[ANY:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
        // CHECK: %[[Q0_2:.*]], %[[Q1_2:.*]] = mqtopt.u2(%[[C0_F64]], %[[C0_F64]]) %[[Q0_1]] ctrl %[[Q1_1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
        // CHECK: %[[Q0_3:.*]], %[[Q1_3:.*]] = mqtopt.p(%[[C0_F64]]) %[[Q0_2]] ctrl %[[Q1_2]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
        // CHECK: %[[Q0_4:.*]], %[[Q1_4:.*]] = mqtopt.rx(%[[C0_F64]]) %[[Q0_3]] ctrl %[[Q1_3]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
        // CHECK: %[[Q0_5:.*]], %[[Q1_5:.*]] = mqtopt.ry(%[[C0_F64]]) %[[Q0_4]] ctrl %[[Q1_4]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
        // CHECK: %[[Q0_6:.*]], %[[Q1_6:.*]] = mqtopt.rz(%[[C0_F64]]) %[[Q0_5]] ctrl %[[Q1_5]] : !mqtopt.Qubit ctrl !mqtopt.Qubit

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %c0_f64 = arith.constant 3.000000e-01 : f64
        %q0_1, %q1_1 = mqtopt.u(%c0_f64, %c0_f64, %c0_f64) %q0_0 ctrl %q1_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit
        %q0_2, %q1_2 = mqtopt.u2(%c0_f64, %c0_f64) %q0_1 ctrl %q1_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
        %q0_3, %q1_3 = mqtopt.p(%c0_f64) %q0_2 ctrl %q1_2 : !mqtopt.Qubit ctrl !mqtopt.Qubit
        %q0_4, %q1_4 = mqtopt.rx(%c0_f64) %q0_3 ctrl %q1_3 : !mqtopt.Qubit ctrl !mqtopt.Qubit
        %q0_5, %q1_5 = mqtopt.ry(%c0_f64) %q0_4 ctrl %q1_4 : !mqtopt.Qubit ctrl !mqtopt.Qubit
        %q0_6, %q1_6 = mqtopt.rz(%c0_f64) %q0_5 ctrl %q1_5 : !mqtopt.Qubit ctrl !mqtopt.Qubit
        %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_6) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_0) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%reg_4) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if an CX gate is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testCXOp
    func.func @testCXOp() {
        // CHECK: %[[Q1_1:.*]], %[[Q0_1:.*]] = mqtopt.x() %[[ANY:.*]] ctrl %[[ANY:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %q1_1, %q0_1 = mqtopt.x() %q1_0 ctrl %q0_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit
        %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_1) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%reg_4) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a negative CX gate is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testNegativeCXOp
    func.func @testNegativeCXOp() {
        // CHECK: %[[Q1_1:.*]], %[[Q0_1:.*]] = mqtopt.x() %[[ANY:.*]] nctrl %[[ANY:.*]] : !mqtopt.Qubit nctrl !mqtopt.Qubit

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %q1_1, %q0_1 = mqtopt.x() %q1_0 nctrl %q0_0 : !mqtopt.Qubit nctrl !mqtopt.Qubit
        %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_1) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%reg_4) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if an MCX gate is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testMCXOp
    func.func @testMCXOp() {
        // CHECK: %[[Q1_1:.*]], %[[Q02_1:.*]]:2 = mqtopt.x() %[[ANY:.*]] ctrl %[[ANY:.*]], %[[ANY:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_3, %q2_0 = "mqtopt.extractQubit"(%reg_2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        //===------------------------------------------------------------------===//
        // q0_0: ──■── q02_1#0
        //       ┌─┴─┐
        // q1_0: ┤ X ├ q1_1
        //       └─┬─┘
        // q2_0: ──■── q02_1#1
        //===----------------------------------------------------------------===//

        %q1_1, %q02_1:2 = mqtopt.x() %q1_0 ctrl %q0_0, %q2_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q02_1#0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_5 = "mqtopt.insertQubit"(%reg_4, %q1_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_6 = "mqtopt.insertQubit"(%reg_5, %q02_1#1) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%reg_6) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if an MCX gate is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testMCXOpAlternativeFormat
    func.func @testMCXOpAlternativeFormat() {
        // CHECK: %[[Q1_1:.*]], %[[Q02_1:.*]]:2 = mqtopt.x() %[[ANY:.*]] ctrl %[[ANY:.*]], %[[ANY:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_3, %q2_0 = "mqtopt.extractQubit"(%reg_2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        //===------------------------------------------------------------------===//
        // q0_0: ──■── q102_1#0
        //       ┌─┴─┐
        // q1_0: ┤ X ├ q102_1#1
        //       └─┬─┘
        // q2_0: ──■── q102_1#2
        //===----------------------------------------------------------------===//

        %q102_1:3 = mqtopt.x() %q1_0 ctrl %q0_0, %q2_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit

        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q102_1#1) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_5 = "mqtopt.insertQubit"(%reg_4, %q102_1#0) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_6 = "mqtopt.insertQubit"(%reg_5, %q102_1#2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%reg_6) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a negative MCX gate is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testNegativeMCXOp
    func.func @testNegativeMCXOp() {
        // CHECK: %[[Q1_1:.*]], %[[Q02_1:.*]]:2 = mqtopt.x() %[[ANY:.*]] nctrl %[[ANY:.*]], %[[ANY:.*]] : !mqtopt.Qubit nctrl !mqtopt.Qubit, !mqtopt.Qubit

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_3, %q2_0 = "mqtopt.extractQubit"(%reg_2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        //===------------------------------------------------------------------===//
        // q0_0: ──○── q02_1#0
        //       ┌─┴─┐
        // q1_0: ┤ X ├ q1_1
        //       └─┬─┘
        // q2_0: ──○── q02_1#1
        //===----------------------------------------------------------------===//

        %q1_1, %q02_1:2 = mqtopt.x() %q1_0 nctrl %q0_0, %q2_0 : !mqtopt.Qubit nctrl !mqtopt.Qubit, !mqtopt.Qubit
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q02_1#0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_5 = "mqtopt.insertQubit"(%reg_4, %q1_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_6 = "mqtopt.insertQubit"(%reg_5, %q02_1#1) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%reg_6) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if an MCX gate is parsed and handled correctly using different types of controls.
module {
    // CHECK-LABEL: func.func @testMixedMCXOp
    func.func @testMixedMCXOp() {
        // CHECK: %[[Q1_1:.*]], %[[Q0_1:.*]], %[[Q2_1:.*]] = mqtopt.x() %[[ANY:.*]] ctrl %[[ANY:.*]] nctrl %[[ANY:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit nctrl !mqtopt.Qubit

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_3, %q2_0 = "mqtopt.extractQubit"(%reg_2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        //===------------------------------------------------------------------===//
        // q0_0: ──■── q0_1
        //       ┌─┴─┐
        // q1_0: ┤ X ├ q1_1
        //       └─┬─┘
        // q2_0: ──○── q2_1
        //===----------------------------------------------------------------===//

        %q1_1, %q0_1, %q2_1 = mqtopt.x() %q1_0 ctrl %q0_0 nctrl %q2_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit nctrl !mqtopt.Qubit
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q0_1) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_5 = "mqtopt.insertQubit"(%reg_4, %q1_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_6 = "mqtopt.insertQubit"(%reg_5, %q2_1) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%reg_6) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if two target gates are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testTwoTargetOp
    func.func @testTwoTargetOp() {
        // CHECK: %[[Q01_1:.*]]:2 = mqtopt.swap() %[[ANY:.*]], %[[ANY:.*]] : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q01_2:.*]]:2 = mqtopt.iswap() %[[Q01_1]]#0, %[[Q01_1]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q01_3:.*]]:2 = mqtopt.iswapdg() %[[Q01_2]]#0, %[[Q01_2]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q01_4:.*]]:2 = mqtopt.peres() %[[Q01_3]]#0, %[[Q01_3]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q01_5:.*]]:2 = mqtopt.peresdg() %[[Q01_4]]#0, %[[Q01_4]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q01_6:.*]]:2 = mqtopt.dcx() %[[Q01_5]]#0, %[[Q01_5]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q01_7:.*]]:2 = mqtopt.ecr() %[[Q01_6]]#0, %[[Q01_6]]#1 : !mqtopt.Qubit, !mqtopt.Qubit

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
        %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_7) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_7) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%reg_4) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a controlled SWAP gate is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testControlledSWAPOp
    func.func @testControlledSWAPOp() {
        // CHECK: %[[Q01_1:.*]]:2, %[[Q2_1:.*]] = mqtopt.swap() %[[ANY:.*]], %[[ANY:.*]] ctrl %[[ANY:.*]] : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_3, %q2_0 = "mqtopt.extractQubit"(%reg_2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %q0_1, %q1_1, %q2_1 = mqtopt.swap() %q0_0, %q1_0 ctrl %q2_0 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q0_1) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_5 = "mqtopt.insertQubit"(%reg_4, %q1_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_6 = "mqtopt.insertQubit"(%reg_5, %q2_1) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%reg_6) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a negative controlled SWAP gate is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testNegativeControlledSWAPOp
    func.func @testNegativeControlledSWAPOp() {
        // CHECK: %[[Q01_1:.*]]:2, %[[Q2_1:.*]] = mqtopt.swap() %[[ANY:.*]], %[[ANY:.*]] nctrl %[[ANY:.*]] : !mqtopt.Qubit, !mqtopt.Qubit nctrl !mqtopt.Qubit

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_3, %q2_0 = "mqtopt.extractQubit"(%reg_2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %q0_1, %q1_1, %q2_1 = mqtopt.swap() %q0_0, %q1_0 nctrl %q2_0 : !mqtopt.Qubit, !mqtopt.Qubit nctrl !mqtopt.Qubit
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q0_1) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_5 = "mqtopt.insertQubit"(%reg_4, %q1_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_6 = "mqtopt.insertQubit"(%reg_5, %q2_1) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%reg_6) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a mixed controlled SWAP gate is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testMixedControlledSWAPOp
    func.func @testMixedControlledSWAPOp() {
        // CHECK: %[[Q01_1:.*]]:2, %[[Q2_1:.*]], %[[Q3_1:.*]] = mqtopt.swap() %[[ANY:.*]], %[[ANY:.*]] ctrl %[[ANY:.*]] nctrl %[[ANY:.*]] : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit nctrl !mqtopt.Qubit

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 4 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_3, %q2_0 = "mqtopt.extractQubit"(%reg_2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_4, %q3_0 = "mqtopt.extractQubit"(%reg_3) <{index_attr = 3 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        //===------------------------------------------------------------------===//
        //       ┌──────┐
        // q0_0: ┤      ├ q0_1
        //       │ SWAP │
        // q1_0: ┤      ├ q1_1
        //       └───┬──┘
        // q2_0: ────■─── q2_1
        //           │
        // q3_0: ────■─── q3_1
        //===----------------------------------------------------------------===//

        %q0_1, %q1_1, %q2_1, %q3_1 = mqtopt.swap() %q0_0, %q1_0 ctrl %q2_0 nctrl %q3_0 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit nctrl !mqtopt.Qubit
        %reg_5 = "mqtopt.insertQubit"(%reg_4, %q0_1) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_6 = "mqtopt.insertQubit"(%reg_5, %q1_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_7 = "mqtopt.insertQubit"(%reg_6, %q2_1) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_8 = "mqtopt.insertQubit"(%reg_7, %q3_1) <{index_attr = 3 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%reg_8) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if parameterized multiple qubit gates are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testMultipleQubitRotationOp
    func.func @testMultipleQubitRotationOp() {
        // CHECK: %[[C0_F64:.*]] = arith.constant 3.000000e-01 : f64
        // CHECK: %[[Q01_1:.*]]:2 = mqtopt.rxx(%[[C0_F64]]) %[[ANY:.*]], %[[ANY:.*]] : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q01_2:.*]]:2 = mqtopt.ryy(%[[C0_F64]]) %[[Q01_1]]#0, %[[Q01_1]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q01_3:.*]]:2 = mqtopt.rzz(%[[C0_F64]]) %[[Q01_2]]#0, %[[Q01_2]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q01_4:.*]]:2 = mqtopt.rzx(%[[C0_F64]]) %[[Q01_3]]#0, %[[Q01_3]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q01_5:.*]]:2 = mqtopt.xxminusyy(%[[C0_F64]], %[[C0_F64]]) %[[Q01_4]]#0, %[[Q01_4]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q01_6:.*]]:2 = mqtopt.xxplusyy(%[[C0_F64]], %[[C0_F64]]) %[[Q01_5]]#0, %[[Q01_5]]#1 : !mqtopt.Qubit, !mqtopt.Qubit

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %c0_f64 = arith.constant 3.000000e-01 : f64
        %q01_1:2 = mqtopt.rxx(%c0_f64) %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
        %q01_2:2 = mqtopt.ryy(%c0_f64) %q01_1#0, %q01_1#1 : !mqtopt.Qubit, !mqtopt.Qubit
        %q01_3:2 = mqtopt.rzz(%c0_f64) %q01_2#0, %q01_2#1 : !mqtopt.Qubit, !mqtopt.Qubit
        %q01_4:2 = mqtopt.rzx(%c0_f64) %q01_3#0, %q01_3#1 : !mqtopt.Qubit, !mqtopt.Qubit
        %q01_5:2 = mqtopt.xxminusyy(%c0_f64, %c0_f64) %q01_4#0, %q01_4#1 : !mqtopt.Qubit, !mqtopt.Qubit
        %q01_6:2 = mqtopt.xxplusyy(%c0_f64, %c0_f64) %q01_5#0, %q01_5#1 : !mqtopt.Qubit, !mqtopt.Qubit
        %reg_3 = "mqtopt.insertQubit"(%reg_2, %q01_6#0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q01_6#1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%reg_4) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if parameterized multiple qubit gates are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testControlledMultipleQubitRotationOp
    func.func @testControlledMultipleQubitRotationOp() {
        // CHECK: %[[C0_F64:.*]] = arith.constant 3.000000e-01 : f64
        // CHECK: %[[Q01_1:.*]]:2, %[[Q2_1:.*]] = mqtopt.rxx(%[[C0_F64]]) %[[ANY:.*]], %[[ANY:.*]] ctrl %[[ANY:.*]] : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
        // CHECK: %[[Q01_2:.*]]:2, %[[Q2_2:.*]] = mqtopt.ryy(%[[C0_F64]]) %[[Q01_1]]#0, %[[Q01_1]]#1 ctrl %[[Q2_1]] : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
        // CHECK: %[[Q01_3:.*]]:2, %[[Q2_3:.*]] = mqtopt.rzz(%[[C0_F64]]) %[[Q01_2]]#0, %[[Q01_2]]#1 ctrl %[[Q2_2]] : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
        // CHECK: %[[Q01_4:.*]]:2, %[[Q2_4:.*]] = mqtopt.rzx(%[[C0_F64]]) %[[Q01_3]]#0, %[[Q01_3]]#1 ctrl %[[Q2_3]] : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
        // CHECK: %[[Q01_5:.*]]:2, %[[Q2_5:.*]] = mqtopt.xxminusyy(%[[C0_F64]], %[[C0_F64]]) %[[Q01_4]]#0, %[[Q01_4]]#1 ctrl %[[Q2_4]] : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
        // CHECK: %[[Q01_6:.*]]:2, %[[Q2_6:.*]] = mqtopt.xxplusyy(%[[C0_F64]], %[[C0_F64]]) %[[Q01_5]]#0, %[[Q01_5]]#1 ctrl %[[Q2_5]] : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_3, %q2_0 = "mqtopt.extractQubit"(%reg_2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %c0_f64 = arith.constant 3.000000e-01 : f64
        %q01_1:2, %q2_1 = mqtopt.rxx(%c0_f64) %q0_0, %q1_0 ctrl %q2_0 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
        %q01_2:2, %q2_2 = mqtopt.ryy(%c0_f64) %q01_1#0, %q01_1#1 ctrl %q2_1 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
        %q01_3:2, %q2_3 = mqtopt.rzz(%c0_f64) %q01_2#0, %q01_2#1 ctrl %q2_2 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
        %q01_4:2, %q2_4 = mqtopt.rzx(%c0_f64) %q01_3#0, %q01_3#1 ctrl %q2_3 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
        %q01_5:2, %q2_5 = mqtopt.xxminusyy(%c0_f64, %c0_f64) %q01_4#0, %q01_4#1 ctrl %q2_4 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
        %q01_6:2, %q2_6 = mqtopt.xxplusyy(%c0_f64, %c0_f64) %q01_5#0, %q01_5#1 ctrl %q2_5 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q01_6#0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_5 = "mqtopt.insertQubit"(%reg_4, %q01_6#1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_6 = "mqtopt.insertQubit"(%reg_5, %q2_6) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%reg_6) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test expects an error to be thrown when an alloc op defines size operands and attributes.
module {
    func.func @testAllocOperandAndAttribute() {
        %reg_size = arith.constant 3 : i64

        // expected-error-re@+1 {{'mqtopt.allocQubitRegister' op exactly one attribute ({{.*}}) or operand ({{.*}}) must be provided for 'size'}}
        %reg = "mqtopt.allocQubitRegister"(%reg_size) <{size_attr = 3 : i64}> : (i64) -> !mqtopt.QubitRegister
    }
}

// -----
// This test expects an error to be thrown when an alloc op does not define size operands nor attributes.
module {
    func.func @testAllocMissingSize() {
        // expected-error-re@+1 {{'mqtopt.allocQubitRegister' op exactly one attribute ({{.*}}) or operand ({{.*}}) must be provided for 'size'}}
        %reg = "mqtopt.allocQubitRegister"() : () -> !mqtopt.QubitRegister
    }
}

// -----
// This test expects an error to be thrown when an extract op defines index operands and attributes.
module {
    func.func @testExtractOperandAndAttribute() {
        %reg = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %idx = arith.constant 0 : i64

        // expected-error-re@+1 {{'mqtopt.extractQubit' op exactly one attribute ({{.*}}) or operand ({{.*}}) must be provided for 'index'}}
        %reg2, %q_0 = "mqtopt.extractQubit"(%reg, %idx) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, i64) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
  }
}

// -----
// This test expects an error to be thrown when an extract op does not define index operands nor attributes.
module {
    func.func @testExtractMissingIndex() {
        %reg = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %idx = arith.constant 0 : i64

        // expected-error-re@+1 {{'mqtopt.extractQubit' op exactly one attribute ({{.*}}) or operand ({{.*}}) must be provided for 'index'}}
        %reg2, %q_0 = "mqtopt.extractQubit"(%reg) : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
  }
}

// -----
// This test expects an error to be thrown when an insert op defines index operands and attributes.
module {
    func.func @testInsertOperandAndAttribute() {
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q_0  = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %idx = arith.constant 0 : i64

        // expected-error-re@+1 {{'mqtopt.insertQubit' op exactly one attribute ({{.*}}) or operand ({{.*}}) must be provided for 'index'}}
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q_0, %idx) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit, i64) -> !mqtopt.QubitRegister
    }
}

// -----
// This test expects an error to be thrown when an insert op does not define index operands nor attributes.
module {
    func.func @testInsertMissingIndex() {
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q_0  = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %idx = arith.constant 0 : i64

        // expected-error-re@+1 {{'mqtopt.insertQubit' op exactly one attribute ({{.*}}) or operand ({{.*}}) must be provided for 'index'}}
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q_0) : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    }
}

// -----
// This test expects an error to be thrown when parsing a controlled operation.
module {
    func.func @testCtrlOpMismatchInOutputs() {
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // expected-error@+1 {{operation defines 2 results but was provided 1 to bind}}
        %q1_1 = mqtopt.x() %q1_0 ctrl %q0_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    }
}

// -----
// This test expects an error to be thrown when parsing a controlled operation using an invalid format.
module {
    func.func @testCtrlOpInvalidFormat() {
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // expected-error@+1 {{number of positively-controlling input qubits (0) and positively-controlling output qubits (1) must be the same}}
        %q1_1, %q0_1 = mqtopt.x() %q1_0 ctrl : !mqtopt.Qubit ctrl !mqtopt.Qubit
    }
}

// -----
// This test expects an error to be thrown when parsing a controlled operation using an invalid format.
module {
    func.func @testNegCtrlOpInvalidFormat() {
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // expected-error@+1 {{number of negatively-controlling input qubits (0) and negatively-controlling output qubits (1) must be the same}}
        %q1_1, %q0_1 = mqtopt.x() %q1_0 nctrl : !mqtopt.Qubit nctrl !mqtopt.Qubit
    }
}

// -----
// This test expects an error to be thrown when parsing a parameterised operation.
module {
    func.func @testParamOpInvalidFormat() {
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %c0_f64 = arith.constant 3.000000e-01 : f64

        // expected-error@+1 {{operation expects exactly 3 parameters but got 2}}
        %q_1 = mqtopt.u(%c0_f64, %c0_f64) %q_0 : !mqtopt.Qubit
    }
}

// -----
// This test checks if a measurement op with a mismatch between in-qubits and out-bits throws an error as expected.
module {
    func.func @testMeasureMismatchInOutBits() {
        %reg = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %reg1, %q0 = "mqtopt.extractQubit"(%reg)  <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg2, %q1 = "mqtopt.extractQubit"(%reg1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // expected-error@+1 {{'mqtopt.measure' op number of input qubits (2) and output bits (0) must be the same}}
        %reg_out = "mqtopt.measure"(%q0, %q1) : (!mqtopt.Qubit, !mqtopt.Qubit) -> (!mqtopt.Qubit)
    }
}

// -----
// This test checks if a Bell state is parsed and handled correctly by using many instructions tested above.
module {
    // CHECK-LABEL: func.func @bellState()
    func.func @bellState() {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 1 : i64}>
        // CHECK: %[[Q0_1:.*]] = mqtopt.h() %[[Q0_0]] : !mqtopt.Qubit
        // CHECK: %[[Q1_1:.*]], %[[Q0_2:.*]] = mqtopt.x() %[[Q1_0:.*]] ctrl %[[Q0_1:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
        // CHECK: %[[Q0_3:.*]], [[M0_0:.*]] = "mqtopt.measure"(%[[Q0_0]])
        // CHECK: %[[Q1_2:.*]], %[[M1_0:.*]] = "mqtopt.measure"(%[[Q1_1]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[Reg_3:.*]] = "mqtopt.insertQubit"(%[[Reg_2]], %[[Q0_3]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q1_2]]) <{index_attr = 1 : i64}>
        // CHECK: "mqtopt.deallocQubitRegister"(%[[Reg_4]]) : (!mqtopt.QubitRegister) -> ()

        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
        %q1_1, %q0_2 = mqtopt.x() %q1_0 ctrl %q0_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
        %q0_3, %m0_0 = "mqtopt.measure"(%q0_0) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %q1_2, %m1_0 = "mqtopt.measure"(%q1_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_3) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_2) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%reg_4) : (!mqtopt.QubitRegister) -> ()
        return
    }
}
