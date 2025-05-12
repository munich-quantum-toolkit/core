// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file | FileCheck %s

// -----
// This test checks if the AllocOp is parsed and handled correctly using a static attribute
module {
    // CHECK-LABEL: func.func @testAllocOp
    func.func @testAllocOp() {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}>
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

        // ========================== Check that there are no further allocations ==============================
        // CHECK-NOT: "mqtopt.allocQubitRegister"

        return
    }
}

// -----
// This test checks if the AllocOp is parsed and handled correctly using a dynamic operand
module {
    // CHECK-LABEL: func.func @testAllocOp
    func.func @testAllocOp() {
        // CHECK: %[[Size:.*]] = arith.constant 2 : i64
        %size = arith.constant 2 : i64

        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"(%[[Size]]) : (i64) -> !mqtopt.QubitRegister
        %reg_0 = "mqtopt.allocQubitRegister"(%size) : (i64) -> !mqtopt.QubitRegister

        // ========================== Check that there are no further allocations ==============================
        // CHECK-NOT: "mqtopt.allocQubitRegister"

        return
    }
}

// -----
// This test checks if the DeallocOp is parsed and handled correctly
module {
    // CHECK-LABEL: func.func @testDeallocOp
    func.func @testDeallocOp() {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

        // CHECK: "mqtopt.deallocQubitRegister"(%[[Reg_0]])
        "mqtopt.deallocQubitRegister"(%reg_0) : (!mqtopt.QubitRegister) -> ()

        // ==========================  Check that there are no further deallocations ==============================
        // CHECK-NOT: "mqtopt.deallocQubitRegister"

        return
    }
}

// -----
// This test checks if the ExtractOp is parsed and handled correctly using a static attribute
module {
    // CHECK-LABEL: func.func @testExtractOp
    func.func @testExtractOp() {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

        // CHECK: %[[Reg_1:.*]], %[[Q_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        %reg_1, %q_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // ==========================  Check that there are no further extractions ==============================
        // CHECK-NOT: "mqtopt.extractQubit"([[ANY:.*]])

        return
    }
}

// -----
// This test checks if the ExtractOp is parsed and handled correctly using a dynamic operand
module {
    // CHECK-LABEL: func.func @testExtractOp
    func.func @testExtractOp() {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

        // CHECK: %[[Index:.*]] = arith.constant 0 : i64
        %index = arith.constant 0 : i64

        // CHECK: %[[Reg_1:.*]], %[[Q_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]], %[[Index]]) : (!mqtopt.QubitRegister, i64) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %reg_1, %q_0 = "mqtopt.extractQubit"(%reg_0, %index) : (!mqtopt.QubitRegister, i64) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // ==========================  Check that there are no further extractions ==============================
        // CHECK-NOT: "mqtopt.extractQubit"([[ANY:.*]])

        return
    }
}

// -----
// This test checks if the InsertOp is parsed and handled correctly using a static attribute
module {
    // CHECK-LABEL: func.func @testInsertOp
    func.func @testInsertOp() {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

        // CHECK: %[[Reg_1:.*]], %[[Q_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        %reg_1, %q_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[Reg_1]], %[[Q_0]])  <{index_attr = 0 : i64}>
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        // ==========================  Check that there are no further insertions ==============================
        // CHECK-NOT: "mqtopt.insertQubit"([[ANY:.*]], [[ANY:.*]])

        return
    }
}

// -----
// This test checks if the InsertOp is parsed and handled correctly using a dynamic operand
module {
    // CHECK-LABEL: func.func @testInsertOp
    func.func @testInsertOp() {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

        // CHECK: %[[Reg_1:.*]], %[[Q_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        %reg_1, %q_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Index:.*]] = arith.constant 0 : i64
        %index = arith.constant 0 : i64

        // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[Reg_1]], %[[Q_0]], %[[Index]])  : (!mqtopt.QubitRegister, !mqtopt.Qubit, i64) -> !mqtopt.QubitRegister
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q_0, %index) : (!mqtopt.QubitRegister, !mqtopt.Qubit, i64) -> !mqtopt.QubitRegister

        // ==========================  Check that there are no further insertions ==============================
        // CHECK-NOT: "mqtopt.insertQubit"([[ANY:.*]], [[ANY:.*]])

        return
    }
}

// -----
// This test checks if the MeasureOp is parsed and handled correctly
module {
    // CHECK-LABEL: func.func @testMeasureOp
    func.func @testMeasureOp() {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

        // CHECK: %[[Reg_1:.*]], %[[Q_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        %reg_1, %q_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Q_1:.*]], [[M0_0:.*]] = "mqtopt.measure"(%[[Q_0]])
        %q_1, %m0_0 = "mqtopt.measure"(%q_0) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

        // ==========================  Check that there are no further measurements ==============================
        // CHECK-NOT: "mqtopt.measure"([[ANY:.*]])

        return
    }
}

// -----
// This test checks if single qubit  gate is parsed and handled correctly
module {
    // CHECK-LABEL: func.func @testSingleQubitOp
    func.func @testSingleQubitOp() {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister

        // CHECK: %[[Reg_1:.*]], %[[Q_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        %reg_1, %q_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Q_1:.*]] = mqtopt.i() %[[Q_0]] : !mqtopt.Qubit
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

        // ==========================  Check that there are no further single qubit operations ==============================
        // CHECK-NOT: mqtopt.i() [[ANY:.*]]
        // CHECK-NOT: mqtopt.h() [[ANY:.*]]
        // CHECK-NOT: mqtopt.x() [[ANY:.*]]
        // CHECK-NOT: mqtopt.y() [[ANY:.*]]
        // CHECK-NOT: mqtopt.z() [[ANY:.*]]
        // CHECK-NOT: mqtopt.s() [[ANY:.*]]
        // CHECK-NOT: mqtopt.sdg() [[ANY:.*]]
        // CHECK-NOT: mqtopt.t() [[ANY:.*]]
        // CHECK-NOT: mqtopt.tdg() [[ANY:.*]]
        // CHECK-NOT: mqtopt.v() [[ANY:.*]]
        // CHECK-NOT: mqtopt.vdg() [[ANY:.*]]
        // CHECK-NOT: mqtopt.sx() [[ANY:.*]]
        // CHECK-NOT: mqtopt.sxdg() [[ANY:.*]]

        // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[Reg_1]], %[[Q_13]])  <{index_attr = 0 : i64}>
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q_13) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        // CHECK: "mqtopt.deallocQubitRegister"(%[[Reg_2]])
        "mqtopt.deallocQubitRegister"(%reg_2) : (!mqtopt.QubitRegister) -> ()

        return
    }
}

// -----
// This test checks if parameterized single qubit gates are parsed and handled correctly
module {
    // CHECK-LABEL: func.func @testSingleQubitRotationOp
    func.func @testSingleQubitRotationOp() {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister

        // CHECK: %[[Reg_1:.*]], %[[Q_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        %reg_1, %q_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[C0_F64:.*]] = arith.constant 3.000000e-01 : f64
        %c0_f64 = arith.constant 3.000000e-01 : f64

        // CHECK: %[[Q_1:.*]] = mqtopt.u(%[[C0_F64]], %[[C0_F64]], %[[C0_F64]]) %[[Q_0]] : !mqtopt.Qubit
        // CHECK: %[[Q_2:.*]] = mqtopt.u2(%[[C0_F64]], %[[C0_F64]]) %[[Q_1]] : !mqtopt.Qubit
        // CHECK: %[[Q_3:.*]] = mqtopt.p(%[[C0_F64]]) %[[Q_2]] : !mqtopt.Qubit
        // CHECK: %[[Q_4:.*]] = mqtopt.rx(%[[C0_F64]]) %[[Q_3]] : !mqtopt.Qubit
        // CHECK: %[[Q_5:.*]] = mqtopt.ry(%[[C0_F64]]) %[[Q_4]] : !mqtopt.Qubit
        // CHECK: %[[Q_6:.*]] = mqtopt.rz(%[[C0_F64]]) %[[Q_5]] : !mqtopt.Qubit

        %q_1 = mqtopt.u(%c0_f64, %c0_f64, %c0_f64) %q_0 : !mqtopt.Qubit
        %q_2 = mqtopt.u2(%c0_f64, %c0_f64) %q_1 : !mqtopt.Qubit
        %q_3 = mqtopt.p(%c0_f64) %q_2 : !mqtopt.Qubit
        %q_4 = mqtopt.rx(%c0_f64) %q_3 : !mqtopt.Qubit
        %q_5 = mqtopt.ry(%c0_f64) %q_4 : !mqtopt.Qubit
        %q_6 = mqtopt.rz(%c0_f64) %q_5 : !mqtopt.Qubit

        // ==========================  Check that there are no further single qubit rotation operations ==============================
        // CHECK-NOT: mqtopt.u(%[[ANY:.*]], %[[ANY:.*]], %[[ANY:.*]]) %[[ANY:.*]]
        // CHECK-NOT: mqtopt.u2(%[[ANY:.*]], %[[ANY:.*]]) %[[ANY:.*]]
        // CHECK-NOT: mqtopt.p(%[[ANY:.*]]) %[[ANY:.*]]
        // CHECK-NOT: mqtopt.rx(%[[ANY:.*]]) %[[ANY:.*]]
        // CHECK-NOT: mqtopt.ry(%[[ANY:.*]]) %[[ANY:.*]]
        // CHECK-NOT: mqtopt.rz(%[[ANY:.*]]) %[[ANY:.*]]

        // CHECK: %[[Reg_2:.*]] = "mqtopt.insertQubit"(%[[Reg_1]], %[[Q_6]])  <{index_attr = 0 : i64}>
        %reg_2 = "mqtopt.insertQubit"(%reg_1, %q_6) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        // CHECK: "mqtopt.deallocQubitRegister"(%[[Reg_2]])
        "mqtopt.deallocQubitRegister"(%reg_2) : (!mqtopt.QubitRegister) -> ()

        return
    }
}

// -----
// This test checks if an CX gate is parsed and handled correctly
module {
    // CHECK-LABEL: func.func @testCXOp
    func.func @testCXOp() {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Q1_1:.*]], %[[Q0_1:.*]] = mqtopt.x() %[[Q1_0]] ctrl %[[Q0_0]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
        %q1_1, %q0_1 = mqtopt.x() %q1_0 ctrl %q0_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit

        // ==========================  Check that there are no further CX operations ==============================
        // CHECK-NOT: %[[ANY:.*]], %[[ANY:.*]] = mqtopt.x() %[[ANY:.*]] ctrl %[[ANY:.*]]

        return
    }
}

// -----
// This test checks if a negative CX gate is parsed and handled correctly
module {
    // CHECK-LABEL: func.func @testNegativeCXOp
    func.func @testNegativeCXOp() {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Q1_1:.*]], %[[Q0_1:.*]] = mqtopt.x() %[[Q1_0]] nctrl %[[Q0_0]] : !mqtopt.Qubit nctrl !mqtopt.Qubit
        %q1_1, %q0_1 = mqtopt.x() %q1_0 nctrl %q0_0 : !mqtopt.Qubit nctrl !mqtopt.Qubit

        // ==========================  Check that there are no further negative CX operations ==============================
        // CHECK-NOT: %[[ANY:.*]], %[[ANY:.*]] = mqtopt.x() %[[ANY:.*]] nctrl %[[ANY:.*]]

        return
    }
}

// -----
// This test checks if an MCX gate is parsed and handled correctly
module {
    // CHECK-LABEL: func.func @testMCXOp
    func.func @testMCXOp() {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister

        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Reg_3:.*]], %[[Q2_0:.*]] = "mqtopt.extractQubit"(%[[Reg_2]]) <{index_attr = 2 : i64}>
        %reg_3, %q2_0 = "mqtopt.extractQubit"(%reg_2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        //===------------------------------------------------------------------===//
        // q0_0: ──■── q02_1#0
        //       ┌─┴─┐
        // q1_0: ┤ X ├ q1_1
        //       └─┬─┘
        // q2_0: ──■── q02_1#1
        //===----------------------------------------------------------------===//

        // CHECK: %[[Q1_1:.*]], %[[Q02_2:.*]]:2 = mqtopt.x() %[[Q1_0]] ctrl %[[Q0_0]], %[[Q2_0]] : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit
        %q1_1, %q02_1:2 = mqtopt.x() %q1_0 ctrl %q0_0, %q2_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit

        // ==========================  Check that there are no further MCX operations ==============================
        // CHECK-NOT: mqtopt.x() [[ANY:.*]] ctrl %[[ANY:.*]]:2

        return
    }
}

// -----
// This test checks if a negative MCX gate is parsed and handled correctly
module {
    // CHECK-LABEL: func.func @testNegativeMCXOp
    func.func @testNegativeMCXOp() {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister

        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Reg_3:.*]], %[[Q2_0:.*]] = "mqtopt.extractQubit"(%[[Reg_2]]) <{index_attr = 2 : i64}>
        %reg_3, %q2_0 = "mqtopt.extractQubit"(%reg_2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        //===------------------------------------------------------------------===//
        // q0_0: ──■── q02_1#0
        //       ┌─┴─┐
        // q1_0: ┤ X ├ q1_1
        //       └─┬─┘
        // q2_0: ──■── q02_1#1
        //===----------------------------------------------------------------===//

        // CHECK: %[[Q1_1:.*]], %[[Q02_2:.*]]:2 = mqtopt.x() %[[Q1_0]] nctrl %[[Q0_0]], %[[Q2_0]] : !mqtopt.Qubit nctrl !mqtopt.Qubit, !mqtopt.Qubit
        %q1_1, %q02_1:2 = mqtopt.x() %q1_0 nctrl %q0_0, %q2_0 : !mqtopt.Qubit nctrl !mqtopt.Qubit, !mqtopt.Qubit

        // ==========================  Check that there are no further MCX operations ==============================
        // CHECK-NOT: mqtopt.x() [[ANY:.*]] nctrl %[[ANY:.*]]:2

        return
    }
}

// -----
// This test checks if an MCX gate is parsed and handled correctly using different types of controls
module {
    // CHECK-LABEL: func.func @testMixedMCXOp
    func.func @testMixedMCXOp() {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister

        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Reg_3:.*]], %[[Q2_0:.*]] = "mqtopt.extractQubit"(%[[Reg_2]]) <{index_attr = 2 : i64}>
        %reg_3, %q2_0 = "mqtopt.extractQubit"(%reg_2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        //===------------------------------------------------------------------===//
        // q0_0: ──■── q0_1
        //       ┌─┴─┐
        // q1_0: ┤ X ├ q1_1
        //       └─┬─┘
        // q2_0: ──■── q2_1
        //===----------------------------------------------------------------===//

        // CHECK: %[[Q1_1:.*]], %[[Q0_1:.*]], %[[Q2_1:.*]] = mqtopt.x() %[[Q1_0]] ctrl %[[Q0_0]] nctrl %[[Q2_0]] : !mqtopt.Qubit ctrl !mqtopt.Qubit nctrl !mqtopt.Qubit
        %q1_1, %q0_1, %q2_1 = mqtopt.x() %q1_0 ctrl %q0_0 nctrl %q2_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit nctrl !mqtopt.Qubit

        // ==========================  Check that there are no further MCX operations ==============================
        // CHECK-NOT: mqtopt.x() [[ANY:.*]] ctrl %[[ANY:.*]] nctrl %[[ANY:.*]]

        return
    }
}


// -----
// This test checks if an SWAP gate is parsed and handled correctly
module {
    // CHECK-LABEL: func.func @testSWAPOp
    func.func @testSWAPOp() {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Q10_1:.*]]:2 = mqtopt.swap() %[[Q1_0]], %[[Q0_0]] : !mqtopt.Qubit, !mqtopt.Qubit
        %q1_1, %q0_1 = mqtopt.swap() %q1_0, %q0_0 : !mqtopt.Qubit, !mqtopt.Qubit

        // ==========================  Check that there are no further SWAP operations ==============================
        // CHECK-NOT: mqtopt.swap() [[ANY:.*]], [[ANY:.*]]

        return
    }
}

// -----
// This test checks if a controlled SWAP gate is parsed and handled correctly
module {
    // CHECK-LABEL: func.func @testControlledSWAPOp
    func.func @testControlledSWAPOp() {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister

        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Reg_3:.*]], %[[Q2_0:.*]] = "mqtopt.extractQubit"(%[[Reg_2]]) <{index_attr = 2 : i64}>
        %reg_3, %q2_0 = "mqtopt.extractQubit"(%reg_2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Q01_1:.*]]:2, %[[Q2_1:.*]] = mqtopt.swap() %[[Q0_0]], %[[Q1_0]] ctrl %[[Q2_0]] : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
        %q0_1, %q1_1, %q2_1 = mqtopt.swap() %q0_0, %q1_0 ctrl %q2_0 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit

        // ==========================  Check that there are no further SWAP operations ==============================
        // CHECK-NOT: mqtopt.swap() [[ANY:.*]], [[ANY:.*]] ctrl [[ANY:.*]]

        return
    }
}

// -----
// This test checks if a negative controlled SWAP gate is parsed and handled correctly
module {
    // CHECK-LABEL: func.func @testNegativeControlledSWAPOp
    func.func @testNegativeControlledSWAPOp() {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister

        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Reg_3:.*]], %[[Q2_0:.*]] = "mqtopt.extractQubit"(%[[Reg_2]]) <{index_attr = 2 : i64}>
        %reg_3, %q2_0 = "mqtopt.extractQubit"(%reg_2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Q01_1:.*]]:2, %[[Q2_1:.*]] = mqtopt.swap() %[[Q0_0]], %[[Q1_0]] nctrl %[[Q2_0]] : !mqtopt.Qubit, !mqtopt.Qubit nctrl !mqtopt.Qubit
        %q0_1, %q1_1, %q2_1 = mqtopt.swap() %q0_0, %q1_0 nctrl %q2_0 : !mqtopt.Qubit, !mqtopt.Qubit nctrl !mqtopt.Qubit

        // ==========================  Check that there are no further SWAP operations ==============================
        // CHECK-NOT: mqtopt.swap() [[ANY:.*]], [[ANY:.*]] nctrl [[ANY:.*]]

        return
    }
}

// -----
// This test checks if a mixed controlled SWAP gate is parsed and handled correctly
module {
    // CHECK-LABEL: func.func @testMixedControlledSWAPOp
    func.func @testNegativeControlledSWAPOp() {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 4 : i64}> : () -> !mqtopt.QubitRegister

        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Reg_3:.*]], %[[Q2_0:.*]] = "mqtopt.extractQubit"(%[[Reg_2]]) <{index_attr = 2 : i64}>
        %reg_3, %q2_0 = "mqtopt.extractQubit"(%reg_2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Reg_4:.*]], %[[Q3_0:.*]] = "mqtopt.extractQubit"(%[[Reg_3]]) <{index_attr = 3 : i64}>
        %reg_4, %q3_0 = "mqtopt.extractQubit"(%reg_3) <{index_attr = 3 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Q01_1:.*]]:2, %[[Q2_1:.*]], %[[Q3_1:.*]] = mqtopt.swap() %[[Q0_0]], %[[Q1_0]] ctrl %[[Q2_0]] nctrl %[[Q3_0]] : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit nctrl !mqtopt.Qubit
        %q0_1, %q1_1, %q2_1, %q3_1 = mqtopt.swap() %q0_0, %q1_0 ctrl %q2_0 nctrl %q3_0 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit nctrl !mqtopt.Qubit

        // ==========================  Check that there are no further SWAP operations ==============================
        // CHECK-NOT: mqtopt.swap() [[ANY:.*]], [[ANY:.*]] ctrl [[ANY:.*]] nctrl [[ANY:.*]]

        return
    }
}

// -----
// This test checks if an iSWAP gate is parsed and handled correctly
module {
    // CHECK-LABEL: func.func @testiSWAPOp
    func.func @testiSWAPOp() {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Q10_1:.*]]:2 = mqtopt.iswap() %[[Q1_0]], %[[Q0_0]] : !mqtopt.Qubit, !mqtopt.Qubit
        %q1_1, %q0_1 = mqtopt.iswap() %q1_0, %q0_0 : !mqtopt.Qubit, !mqtopt.Qubit

        // ==========================  Check that there are no further iSWAP operations ==============================
        // CHECK-NOT: mqtopt.iswap() [[ANY:.*]], [[ANY:.*]]

        return
    }
}

// -----
// This test checks if an Peres gate is parsed and handled correctly
module {
    // CHECK-LABEL: func.func @testPeresOp
    func.func @testPeresOp() {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Q10_1:.*]]:2 = mqtopt.peres() %[[Q1_0]], %[[Q0_0]] : !mqtopt.Qubit, !mqtopt.Qubit
        %q1_1, %q0_1 = mqtopt.peres() %q1_0, %q0_0 : !mqtopt.Qubit, !mqtopt.Qubit

        // ==========================  Check that there are no further Peres operations ==============================
        // CHECK-NOT: mqtopt.peres() [[ANY:.*]], [[ANY:.*]]

        return
    }
}

// -----
// This test checks if an Peresdg gate is parsed and handled correctly
module {
    // CHECK-LABEL: func.func @testPeresdgOp
    func.func @testPeresdgOp() {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Q10_1:.*]]:2 = mqtopt.peresdg() %[[Q1_0]], %[[Q0_0]] : !mqtopt.Qubit, !mqtopt.Qubit
        %q1_1, %q0_1 = mqtopt.peresdg() %q1_0, %q0_0 : !mqtopt.Qubit, !mqtopt.Qubit

        // ==========================  Check that there are no further Peresdg operations ==============================
        // CHECK-NOT: mqtopt.peresdg() [[ANY:.*]], [[ANY:.*]]

        return
    }
}

// -----
// This test checks if an DCX gate is parsed and handled correctly
module {
    // CHECK-LABEL: func.func @testDCXOp
    func.func @testDCXOp() {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Q10_1:.*]]:2 = mqtopt.dcx() %[[Q1_0]], %[[Q0_0]] : !mqtopt.Qubit, !mqtopt.Qubit
        %q1_1, %q0_1 = mqtopt.dcx() %q1_0, %q0_0 : !mqtopt.Qubit, !mqtopt.Qubit

        // ==========================  Check that there are no further DCX operations ==============================
        // CHECK-NOT: mqtopt.dcx() [[ANY:.*]], [[ANY:.*]]

        return
    }
}

// -----
// This test checks if an ECR gate is parsed and handled correctly
module {
    // CHECK-LABEL: func.func @testECROp
    func.func @testECROp() {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Q10_1:.*]]:2 = mqtopt.ecr() %[[Q1_0]], %[[Q0_0]] : !mqtopt.Qubit, !mqtopt.Qubit
        %q1_1, %q0_1 = mqtopt.ecr() %q1_0, %q0_0 : !mqtopt.Qubit, !mqtopt.Qubit

        // ==========================  Check that there are no further ECR operations ==============================
        // CHECK-NOT: mqtopt.ecr() [[ANY:.*]], [[ANY:.*]]

        return
    }
}

// -----
// This test checks if parameterized single qubit gates are parsed and handled correctly
module {
    // CHECK-LABEL: func.func @testMultipleQubitRotationOp
    func.func @testMultipleQubitRotationOp() {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_1]]) <{index_attr = 1 : i64}>
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[C0_F64:.*]] = arith.constant 3.000000e-01 : f64
        %c0_f64 = arith.constant 3.000000e-01 : f64

        // CHECK: %[[Q01_1:.*]]:2 = mqtopt.rxx(%[[C0_F64]]) %[[Q0_0]], %[[Q1_0]] : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q01_2:.*]]:2 = mqtopt.ryy(%[[C0_F64]]) %[[Q01_1]]#0, %[[Q01_1]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q01_3:.*]]:2 = mqtopt.rzz(%[[C0_F64]]) %[[Q01_2]]#0, %[[Q01_2]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q01_4:.*]]:2 = mqtopt.rzx(%[[C0_F64]]) %[[Q01_3]]#0, %[[Q01_3]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q01_5:.*]]:2 = mqtopt.xxminusyy(%[[C0_F64]], %[[C0_F64]]) %[[Q01_4]]#0, %[[Q01_4]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[Q01_6:.*]]:2 = mqtopt.xxplusyy(%[[C0_F64]], %[[C0_F64]]) %[[Q01_5]]#0, %[[Q01_5]]#1 : !mqtopt.Qubit, !mqtopt.Qubit

        %q01_1:2 = mqtopt.rxx(%c0_f64) %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
        %q01_2:2 = mqtopt.ryy(%c0_f64) %q01_1#0, %q01_1#1 : !mqtopt.Qubit, !mqtopt.Qubit
        %q01_3:2 = mqtopt.rzz(%c0_f64) %q01_2#0, %q01_2#1 : !mqtopt.Qubit, !mqtopt.Qubit
        %q01_4:2 = mqtopt.rzx(%c0_f64) %q01_3#0, %q01_3#1 : !mqtopt.Qubit, !mqtopt.Qubit
        %q01_5:2 = mqtopt.xxminusyy(%c0_f64, %c0_f64) %q01_4#0, %q01_4#1 : !mqtopt.Qubit, !mqtopt.Qubit
        %q01_6:2 = mqtopt.xxplusyy(%c0_f64, %c0_f64) %q01_5#0, %q01_5#1 : !mqtopt.Qubit, !mqtopt.Qubit

        // ==========================  Check that there are no further single qubit rotation operations ==============================
        // CHECK-NOT: mqtopt.rxx(%[[ANY:.*]]) %[[ANY:.*]], %[[ANY:.*]]
        // CHECK-NOT: mqtopt.ryy(%[[ANY:.*]]) %[[ANY:.*]], %[[ANY:.*]]
        // CHECK-NOT: mqtopt.rzz(%[[ANY:.*]]) %[[ANY:.*]], %[[ANY:.*]]
        // CHECK-NOT: mqtopt.rzx(%[[ANY:.*]]) %[[ANY:.*]], %[[ANY:.*]]
        // CHECK-NOT: mqtopt.xxminusyy(%[[ANY:.*]], %[[ANY:.*]], %[[ANY:.*]]) %[[ANY:.*]], %[[ANY:.*]]
        // CHECK-NOT: mqtopt.xxplusyy(%[[ANY:.*]], %[[ANY:.*]]) %[[ANY:.*]], %[[ANY:.*]]

        // CHECK: %[[Reg_3:.*]] = "mqtopt.insertQubit"(%[[Reg_2]], %[[Q01_6]]#0)  <{index_attr = 0 : i64}>
        %reg_3 = "mqtopt.insertQubit"(%reg_2, %q01_6#0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q01_6]]#1)  <{index_attr = 1 : i64}>
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q01_6#1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        // CHECK: "mqtopt.deallocQubitRegister"(%[[Reg_4]])
        "mqtopt.deallocQubitRegister"(%reg_4) : (!mqtopt.QubitRegister) -> ()

        return
    }
}

// -----
// This test checks if a Bell state is parsed and handled correctly by using many instructions tested above
module {
    // CHECK-LABEL: func.func @bellState()
    func.func @bellState() {
        // CHECK: %[[Reg_0:.*]] = "mqtopt.allocQubitRegister"
        %reg_0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister

        // CHECK: %[[Reg_1:.*]], %[[Q0_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 0 : i64}>
        %reg_1, %q0_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Reg_2:.*]], %[[Q1_0:.*]] = "mqtopt.extractQubit"(%[[Reg_0]]) <{index_attr = 1 : i64}>
        %reg_2, %q1_0 = "mqtopt.extractQubit"(%reg_0) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        // CHECK: %[[Q0_1:.*]] = mqtopt.x() %[[Q0_0]] : !mqtopt.Qubit
        %q0_1 = mqtopt.x() %q0_0 : !mqtopt.Qubit

        // CHECK: %[[Q1_1:.*]], %[[Q0_2:.*]] = mqtopt.x() %[[Q1_0:.*]] ctrl %[[Q0_1:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
        %q1_1, %q0_2 = mqtopt.x() %q1_0 ctrl %q0_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit

        // CHECK: %[[Q0_3:.*]], [[M0_0:.*]] = "mqtopt.measure"(%[[Q0_0]])
        %q0_3, %m0_0 = "mqtopt.measure"(%q0_0) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

        // CHECK: %[[Q1_2:.*]], %[[M1_0:.*]] = "mqtopt.measure"(%[[Q1_1]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %q1_2, %m1_0 = "mqtopt.measure"(%q1_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

        // CHECK: %[[Reg_3:.*]] = "mqtopt.insertQubit"(%[[Reg_2]], %[[Q0_3]]) <{index_attr = 0 : i64}>
        %reg_3 = "mqtopt.insertQubit"(%reg_2, %q0_3) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        // CHECK: %[[Reg_4:.*]] = "mqtopt.insertQubit"(%[[Reg_3]], %[[Q1_2]]) <{index_attr = 1 : i64}>
        %reg_4 = "mqtopt.insertQubit"(%reg_3, %q1_2) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

        // CHECK: "mqtopt.deallocQubitRegister"(%[[Reg_4]]) : (!mqtopt.QubitRegister) -> ()
        "mqtopt.deallocQubitRegister"(%reg_4) : (!mqtopt.QubitRegister) -> ()

        return
    }
}
