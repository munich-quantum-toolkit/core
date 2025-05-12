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
module {
    // CHECK-LABEL: func @bell_state()
    func.func @bell_state() -> (i1, i1) {
        %0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %out_qreg, %out_qubit = "mqtopt.extractQubit"(%0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %c2_i64 = arith.constant 2 : i64
        %1 = "mqtopt.allocQubitRegister"(%c2_i64) : (i64) -> !mqtopt.QubitRegister
        %c0_i64 = arith.constant 0 : i64
        %out_qreg_0, %out_qubit_1 = "mqtopt.extractQubit"(%1, %c0_i64) : (!mqtopt.QubitRegister, i64) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %2 = mqtopt.x() %out_qubit : !mqtopt.Qubit
        %3, %4 = mqtopt.x() %2 ctrl %out_qubit_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
        %cst = arith.constant 3.000000e-01 : f64
        %5, %6 = mqtopt.rz(%cst) %3 nctrl %4 : !mqtopt.Qubit nctrl !mqtopt.Qubit
        %7 = mqtopt.u(%cst, %cst static [3.000000e-01] mask [false, true, false]) %5 : !mqtopt.Qubit
        %out_qubits, %out_bits = "mqtopt.measure"(%7) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %out_qubits_2, %out_bits_3 = "mqtopt.measure"(%6) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %8 = "mqtopt.insertQubit"(%out_qreg, %out_qubits) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %9 = "mqtopt.insertQubit"(%out_qreg_0, %out_qubits_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%8) : (!mqtopt.QubitRegister) -> ()
        "mqtopt.deallocQubitRegister"(%9) : (!mqtopt.QubitRegister) -> ()
        return %out_bits, %out_bits_3 : i1, i1
    }
}
