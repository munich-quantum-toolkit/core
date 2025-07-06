// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --mqtopt-to-mqtdyn | FileCheck %s

// -----
// This test checks if the AllocOp is converted correctly using a static attribute.
module {
    // CHECK-LABEL: func.func @testConvertAllocOpAttribute()
    func.func @testConvertAllocOpAttribute() {
        // CHECK: %[[r_0:.*]] = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}>

        %r0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%r0) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if the AllocOp is converted correctly using a dynamic operand.
module {
    // CHECK-LABEL: func.func @testConvertAllocOpOperand()
    func.func @testConvertAllocOpOperand() {
        // CHECK: %[[size:.*]] = arith.constant 2
        // CHECK: %[[r_0:.*]] = "mqtdyn.allocQubitRegister"(%[[size]]) : (i64) -> !mqtdyn.QubitRegister

        %size = arith.constant 2 : i64
        %r0 = "mqtopt.allocQubitRegister" (%size) : (i64) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%r0) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if the DeallocOp is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertDeallocOp
    func.func @testConvertDeallocOp() {
        // CHECK: "mqtdyn.deallocQubitRegister"(%[[ANY:.*]])

        %r0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%r0) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if the ExtractOp is converted correctly and the insertOP is removed using a static attribute.
module {
    // CHECK-LABEL: func.func @testConvertExtractOpAttribute
    func.func @testConvertExtractOpAttribute() {
        // CHECK: %[[q_0:.*]] = "mqtdyn.extractQubit"(%[[ANY:.*]]) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        //CHECK-NOT: %[[ANY:.*]] = mqtopt.insertQubit (%[[ANY:.*]], %[[ANY:.*]]) <{index_attr = 0 : i64}>

        %r0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %r1, %q0 = "mqtopt.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %r2 = "mqtopt.insertQubit"(%r1, %q0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%r2) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if the ExtractOp is converted correctly and the insertOP is removed using a dynamic operand.
module {
    // CHECK-LABEL: func.func @testConvertExtractOpOperand
    func.func @testConvertExtractOpOperand() {
        // CHECK: %[[index:.*]] = arith.constant 0
        // CHECK: %[[q_0:.*]] = "mqtdyn.extractQubit"(%[[ANY:.*]], %[[index]]) : (!mqtdyn.QubitRegister, i64) -> !mqtdyn.Qubit
        //CHECK-NOT: %[[ANY:.*]] = mqtopt.insertQubit (%[[ANY:.*]], %[[ANY:.*]], %[[ANY:.*]])

        %index = arith.constant 0 : i64
        %r0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %r1, %q0 = "mqtopt.extractQubit"(%r0, %index) : (!mqtopt.QubitRegister, i64) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %r2 = "mqtopt.insertQubit"(%r1, %q0, %index)  : (!mqtopt.QubitRegister, !mqtopt.Qubit, i64) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%r2) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if the operands/results are correctly replaced for the def-use chain in the dyn dialect
module {
    // CHECK-LABEL: func.func @testConvertOperandChain
    func.func @testConvertOperandChain() {
        // CHECK: %[[r_0:.*]] = "mqtdyn.allocQubitRegister"() <{size_attr = 3 : i64}>
        // CHECK: %[[q_0:.*]] = "mqtdyn.extractQubit"(%[[r_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[q_1:.*]] = "mqtdyn.extractQubit"(%[[r_0]]) <{index_attr = 1 : i64}>
        // CHECK: %[[q_2:.*]] = "mqtdyn.extractQubit"(%[[r_0]]) <{index_attr = 2 : i64}>
        // CHECK: "mqtdyn.deallocQubitRegister"(%[[r_0:.*]])

        %r0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister
        %r1, %q0 = "mqtopt.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %r2, %q1 = "mqtopt.extractQubit"(%r1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %r3, %q2 = "mqtopt.extractQubit"(%r2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %r4 = "mqtopt.insertQubit"(%r3, %q0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %r5 = "mqtopt.insertQubit"(%r4, %q1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %r6 = "mqtopt.insertQubit"(%r5, %q2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%r6) : (!mqtopt.QubitRegister) -> ()
        return
    }
}


// -----
// This test checks if the MeasureOp is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertMeasureOp
    func.func @testConvertMeasureOp() {
        // CHECK:  [[m_0:.*]] = "mqtdyn.measure"(%[[ANY:.*]])

        %r0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %r1, %q0 = "mqtopt.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %q1, %m0 = "mqtopt.measure"(%q0) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %r2 = "mqtopt.insertQubit"(%r1, %q1) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%r2) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if the ExtractOp is converted correctly for multiple qubits
module {
    // CHECK-LABEL: func.func @testConvertMeasureOpOnMultipleInputs
    func.func @testConvertMeasureOpOnMultipleInputs() {
        // CHECK: [[m01_1:.*]]:2 = "mqtdyn.measure"(%[[ANY:.*]], %[[ANY:.*]])

        %r0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %r1, %q0 = "mqtopt.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %r2, %q1 = "mqtopt.extractQubit"(%r1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %q01:2, %m:2 = "mqtopt.measure"(%q0, %q1) : (!mqtopt.Qubit, !mqtopt.Qubit) -> (!mqtopt.Qubit, !mqtopt.Qubit, i1, i1)
        %r3 = "mqtopt.insertQubit"(%r2, %q01#0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %r4 = "mqtopt.insertQubit"(%r3, %q01#1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%r4) : (!mqtopt.QubitRegister) -> ()
        return
    }
}


// -----
// This test checks if single qubit gates are converted correctly
module {
    // CHECK-LABEL: func.func @testConvertSingleQubitOp
    func.func @testConvertSingleQubitOp() {
        // CHECK: mqtdyn.i() %[[q_0:.*]]
        // CHECK: mqtdyn.h() %[[q_0]]
        // CHECK: mqtdyn.x() %[[q_0]]
        // CHECK: mqtdyn.y() %[[q_0]]
        // CHECK: mqtdyn.z() %[[q_0]]
        // CHECK: mqtdyn.s() %[[q_0]]
        // CHECK: mqtdyn.sdg() %[[q_0]]
        // CHECK: mqtdyn.t() %[[q_0]]
        // CHECK: mqtdyn.tdg() %[[q_0]]
        // CHECK: mqtdyn.v() %[[q_0]]
        // CHECK: mqtdyn.vdg() %[[q_0]]
        // CHECK: mqtdyn.sx() %[[q_0]]
        // CHECK: mqtdyn.sxdg() %[[q_0]]

        %r0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %r1, %q0 = "mqtopt.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %q1 = mqtopt.i() %q0 : !mqtopt.Qubit
        %q2 = mqtopt.h() %q1 : !mqtopt.Qubit
        %q3 = mqtopt.x() %q2 : !mqtopt.Qubit
        %q4 = mqtopt.y() %q3 : !mqtopt.Qubit
        %q5 = mqtopt.z() %q4 : !mqtopt.Qubit
        %q6 = mqtopt.s() %q5 : !mqtopt.Qubit
        %q7 = mqtopt.sdg() %q6 : !mqtopt.Qubit
        %q8 = mqtopt.t() %q7 : !mqtopt.Qubit
        %q9 = mqtopt.tdg() %q8 : !mqtopt.Qubit
        %q10 = mqtopt.v() %q9 : !mqtopt.Qubit
        %q11 = mqtopt.vdg() %q10 : !mqtopt.Qubit
        %q12 = mqtopt.sx() %q11 : !mqtopt.Qubit
        %q13 = mqtopt.sxdg() %q12 : !mqtopt.Qubit
        %r2 = "mqtopt.insertQubit"(%r1, %q13) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%r2) : (!mqtopt.QubitRegister) -> ()
        return
    }
}


// -----
// This test checks if two target gates are converted correctly
module {
    // CHECK-LABEL: func.func @testConvertTwoTargetOp
    func.func @testConvertTwoTargetOp() {
        // CHECK: mqtdyn.swap() %[[q_0:.*]], %[[q_1:.*]]
        // CHECK: mqtdyn.iswap() %[[q_0]], %[[q_1]]
        // CHECK: mqtdyn.iswapdg() %[[q_0]], %[[q_1]]
        // CHECK: mqtdyn.peres() %[[q_0]], %[[q_1]]
        // CHECK: mqtdyn.peresdg() %[[q_0]], %[[q_1]]
        // CHECK: mqtdyn.dcx() %[[q_0]], %[[q_1]]
        // CHECK: mqtdyn.ecr() %[[q_0]], %[[q_1]]

        %r0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %r1, %q0_0 = "mqtopt.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %r2, %q1_0 = "mqtopt.extractQubit"(%r1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %q0_1, %q1_1 = mqtopt.swap() %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
        %q0_2, %q1_2 = mqtopt.iswap() %q0_1, %q1_1 : !mqtopt.Qubit, !mqtopt.Qubit
        %q0_3, %q1_3 = mqtopt.iswapdg() %q0_2, %q1_2 : !mqtopt.Qubit, !mqtopt.Qubit
        %q0_4, %q1_4 = mqtopt.peres() %q0_3, %q1_3 : !mqtopt.Qubit, !mqtopt.Qubit
        %q0_5, %q1_5 = mqtopt.peresdg() %q0_4, %q1_4 : !mqtopt.Qubit, !mqtopt.Qubit
        %q0_6, %q1_6 = mqtopt.dcx() %q0_5, %q1_5 : !mqtopt.Qubit, !mqtopt.Qubit
        %q0_7, %q1_7 = mqtopt.ecr() %q0_6, %q1_6 : !mqtopt.Qubit, !mqtopt.Qubit
        %r3 = "mqtopt.insertQubit"(%r2, %q0_7) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %r4 = "mqtopt.insertQubit"(%r3, %q1_7) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%r4) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if parameterized single qubit gates are converted correctly
module {
    // CHECK-LABEL: func.func @testSingleQubitRotationOp
    func.func @testSingleQubitRotationOp() {
        // CHECK: %[[c_0:.*]] = arith.constant 3.000000e-01
        // CHECK: mqtdyn.u(%[[c_0]], %[[c_0]], %[[c_0]]) %[[q_0:.*]]
        // CHECK: mqtdyn.u2(%[[c_0]], %[[c_0]]) %[[q_0]]
        // CHECK: mqtdyn.p(%[[c_0]]) %[[q_0]]
        // CHECK: mqtdyn.rx(%[[c_0]]) %[[q_0]]
        // CHECK: mqtdyn.ry(%[[c_0]]) %[[q_0]]
        // CHECK: mqtdyn.rz(%[[c_0]]) %[[q_0]]

        %r0 = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtopt.QubitRegister
        %r1, %q0 = "mqtopt.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %c0 = arith.constant 3.000000e-01 : f64
        %q1 = mqtopt.u(%c0, %c0, %c0) %q0 : !mqtopt.Qubit
        %q2 = mqtopt.u2(%c0, %c0) %q1 : !mqtopt.Qubit
        %q3 = mqtopt.p(%c0) %q2 : !mqtopt.Qubit
        %q4 = mqtopt.rx(%c0) %q3 : !mqtopt.Qubit
        %q5 = mqtopt.ry(%c0) %q4 : !mqtopt.Qubit
        %q6 = mqtopt.rz(%c0) %q5 : !mqtopt.Qubit
        %r2 = "mqtopt.insertQubit"(%r1, %q6) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%r2) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if parameterized multiple qubit gates are converted correctly
module {
    // CHECK-LABEL: func.func @testMultipleQubitRotationOp
    func.func @testMultipleQubitRotationOp() {
        // CHECK: %[[c_0:.*]] = arith.constant 3.000000e-01 : f64
        // CHECK: mqtdyn.rxx(%[[c_0]]) %[[q_0:.*]], %[[q_1:.*]]
        // CHECK: mqtdyn.ryy(%[[c_0]]) %[[q_0]], %[[q_1]]
        // CHECK: mqtdyn.rzz(%[[c_0]]) %[[q_0]], %[[q_1]]
        // CHECK: mqtdyn.rzx(%[[c_0]]) %[[q_0]], %[[q_1]]
        // CHECK: mqtdyn.xxminusyy(%[[c_0]], %[[c_0]]) %[[q_0]], %[[q_1]]
        // CHECK: mqtdyn.xxplusyy(%[[c_0]], %[[c_0]]) %[[q_0]], %[[q_1]]

        %r0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %r1, %q0 = "mqtopt.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %r2, %q1 = "mqtopt.extractQubit"(%r1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %c0 = arith.constant 3.000000e-01 : f64
        %q01_1:2 = mqtopt.rxx(%c0) %q0, %q1 : !mqtopt.Qubit, !mqtopt.Qubit
        %q01_2:2 = mqtopt.ryy(%c0) %q01_1#0, %q01_1#1 : !mqtopt.Qubit, !mqtopt.Qubit
        %q01_3:2 = mqtopt.rzz(%c0) %q01_2#0, %q01_2#1 : !mqtopt.Qubit, !mqtopt.Qubit
        %q01_4:2 = mqtopt.rzx(%c0) %q01_3#0, %q01_3#1 : !mqtopt.Qubit, !mqtopt.Qubit
        %q01_5:2 = mqtopt.xxminusyy(%c0, %c0) %q01_4#0, %q01_4#1 : !mqtopt.Qubit, !mqtopt.Qubit
        %q01_6:2 = mqtopt.xxplusyy(%c0, %c0) %q01_5#0, %q01_5#1 : !mqtopt.Qubit, !mqtopt.Qubit
        %r3 = "mqtopt.insertQubit"(%r2, %q01_6#0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %r4 = "mqtopt.insertQubit"(%r3, %q01_6#1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%r4) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if static params and paramask is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertStaticParams
    func.func @testConvertStaticParams() {
        // CHECK:  mqtdyn.u(%[[ANY:.*]], %[[ANY:.*]] static [3.000000e-01] mask [false, true, false]) %[[ANY:.*]]

        %r0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %r1, %q0 = "mqtopt.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %cst = arith.constant 3.000000e-01 : f64
        %q1 = mqtopt.u(%cst, %cst static [3.000000e-01] mask [false, true, false]) %q0 : !mqtopt.Qubit
        %r2 = "mqtopt.insertQubit"(%r1, %q1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%r2) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a controlled op is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertControlledOp
    func.func @testConvertControlledOp() {
        // CHECK: mqtdyn.x() %[[q_0:.*]] ctrl %[[q_1:.*]]

        %r0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %r1, %q0 = "mqtopt.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %r2, %q1 = "mqtopt.extractQubit"(%r1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %q1_1, %q0_1 = mqtopt.x() %q1 ctrl %q0 : !mqtopt.Qubit ctrl !mqtopt.Qubit
        %r3 = "mqtopt.insertQubit"(%r2, %q0_1) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %r4 = "mqtopt.insertQubit"(%r3, %q1_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%r4) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a negative controlled op is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertNegativeControlledOp
    func.func @testConvertNegativeControlledOp() {
        // CHECK: mqtdyn.x() %[[q_0:.*]] nctrl %[[q_1:.*]]

        %r0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %r1, %q0 = "mqtopt.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %r2, %q1 = "mqtopt.extractQubit"(%r1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %q1_1, %q0_1 = mqtopt.x() %q1 nctrl %q0 : !mqtopt.Qubit nctrl !mqtopt.Qubit
        %r3 = "mqtopt.insertQubit"(%r2, %q0_1) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %r4 = "mqtopt.insertQubit"(%r3, %q1_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%r4) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a Bell state is converted correctly.
module {
    // CHECK-LABEL: func.func @bellState()
    func.func @bellState() {
        // CHECK: %[[r_0:.*]] = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}>
        // CHECK: %[[q_0:.*]] = "mqtdyn.extractQubit"(%[[r_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[q_1:.*]] = "mqtdyn.extractQubit"(%[[r_0]]) <{index_attr = 1 : i64}>
        // CHECK: mqtdyn.h() %[[q_0]]
        // CHECK: mqtdyn.x() %[[q_1]] ctrl %[[q_0]]
        // CHECK: %[[m_0:.*]] = "mqtdyn.measure"(%[[q_0]]) : (!mqtdyn.Qubit) -> i1
        // CHECK: %[[m_1:.*]] = "mqtdyn.measure"(%[[q_1]]) : (!mqtdyn.Qubit) -> i1
        // CHECK: "mqtdyn.deallocQubitRegister"(%[[r_0]]) : (!mqtdyn.QubitRegister) -> ()

        %r0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %r1, %q0 = "mqtopt.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %r2, %q1 = "mqtopt.extractQubit"(%r1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %q0_1 = mqtopt.h() %q0 : !mqtopt.Qubit
        %q1_1, %q0_2 = mqtopt.x() %q1 ctrl %q0_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
        %q0_3, %m0 = "mqtopt.measure"(%q0_2) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %q1_2, %m1 = "mqtopt.measure"(%q1_1) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %r3 = "mqtopt.insertQubit"(%r2, %q0_3) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %r4 = "mqtopt.insertQubit"(%r3, %q1_2) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%r4) : (!mqtopt.QubitRegister) -> ()
        return
    }
}



// -----
// This test checks if a barrierOp is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertBarrierOp()
    func.func @testConvertBarrierOp() {
        // CHECK: mqtdyn.barrier() %[[ANY:.*]]

        %r0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %r1, %q0 = "mqtopt.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %q1 = mqtopt.barrier() %q0 : !mqtopt.Qubit
        %r2 = "mqtopt.insertQubit"(%r1, %q1) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%r2) : (!mqtopt.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a barrierOp with multiple inputs is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertBarrierOpMultipleInputs()
    func.func @testConvertBarrierOpMultipleInputs() {
        // CHECK: mqtdyn.barrier() %[[ANY:.*]], %[[ANY:.*]]

        %r0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %r1, %q0 = "mqtopt.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %r2, %q1 = "mqtopt.extractQubit"(%r1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %q01_1:2 = mqtopt.barrier() %q0, %q1 : !mqtopt.Qubit, !mqtopt.Qubit
        %r3 = "mqtopt.insertQubit"(%r2, %q01_1#0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %r4 = "mqtopt.insertQubit"(%r3, %q01_1#1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%r4) : (!mqtopt.QubitRegister) -> ()
        return
    }
}
