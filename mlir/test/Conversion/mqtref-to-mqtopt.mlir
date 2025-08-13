// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --mqtref-to-mqtopt | FileCheck %s

// -----
// This test checks if the AllocOp is converted correctly using a static attribute.
module {
    // CHECK-LABEL: func.func @testConvertAllocOpAttribute()
    func.func @testConvertAllocOpAttribute() {
        // CHECK: %[[r_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}>

        %r0 = "mqtref.allocQubitRegister" () {"size_attr" = 2 : i64} : () -> !mqtref.QubitRegister
        return
    }
}

// -----
// This test checks if the AllocOp is converted correctly using a dynamic operand.
module {
    // CHECK-LABEL: func.func @testConvertAllocOpOperand()
    func.func @testConvertAllocOpOperand() {
        // CHECK: %[[size:.*]] = arith.constant 2
        // CHECK: %[[r_0:.*]] = "mqtopt.allocQubitRegister"(%[[size]]) : (i64) -> !mqtopt.QubitRegister

        %size = arith.constant 2 : i64
        %r0 = "mqtref.allocQubitRegister" (%size) : (i64) -> !mqtref.QubitRegister
        return
    }
}

// -----
// This test checks if the DeallocOp is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertDeallocOp
    func.func @testConvertDeallocOp() {
        // CHECK: "mqtopt.deallocQubitRegister"(%[[ANY:.*]])

        %r0 = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
        "mqtref.deallocQubitRegister"(%r0) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if the ExtractOp is converted correctly using a static attribute.
module {
    // CHECK-LABEL: func.func @testConvertExtractOpAttribute
    func.func @testConvertExtractOpAttribute() {
        // CHECK: %[[r_0:.*]], %[[q_0:.*]] = "mqtopt.extractQubit"(%[[ANY:.*]]) <{index_attr = 0 : i64}>

        %r0 = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        return
    }
}

// -----
// This test checks if the ExtractOp is converted correctly using a dynamic operand.
module {
    // CHECK-LABEL: func.func @testConvertExtractOpOperand
    func.func @testConvertExtractOpOperand() {
        // CHECK: %[[index:.*]] = arith.constant 0
        // CHECK: %[[r_0:.*]], %[[q_0:.*]] = "mqtopt.extractQubit"(%[[ANY:.*]], %[[index]]) : (!mqtopt.QubitRegister, i64) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

        %index = arith.constant 0 : i64
        %r0 = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%r0, %index) : (!mqtref.QubitRegister, i64) -> !mqtref.Qubit
        return
    }
}

// -----
// This test checks if the operands/results are correctly replaced for the def-use chain in the opt dialect
module {
    // CHECK-LABEL: func.func @testConvertOperandChain
    func.func @testConvertOperandChain() {
        // CHECK: %[[r_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}>
        // CHECK: %[[r_1:.*]], %[[q_0:.*]] = "mqtopt.extractQubit"(%[[r_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[r_2:.*]], %[[q_1:.*]] = "mqtopt.extractQubit"(%[[r_1]]) <{index_attr = 1 : i64}>
        // CHECK: %[[r_3:.*]], %[[q_2:.*]] = "mqtopt.extractQubit"(%[[r_2]]) <{index_attr = 2 : i64}>

        %r0 = "mqtref.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q1 = "mqtref.extractQubit"(%r0) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q2 = "mqtref.extractQubit"(%r0) <{index_attr = 2 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        return
    }
}

// -----
// This test checks if the InsertOp is correctly inserted using a static attribute before deallocating a qubit register.
module {
    // CHECK-LABEL: func.func @testConvertInsertOpAttribute
    func.func @testConvertInsertOpAttribute() {
        // CHECK: %[[r_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}>
        // CHECK: %[[r_1:.*]], %[[q_0:.*]] = "mqtopt.extractQubit"(%[[r_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[r_2:.*]] = "mqtopt.insertQubit"(%[[r_1]], %[[q_0]])  <{index_attr = 0 : i64}>
        // CHECK: "mqtopt.deallocQubitRegister"(%[[r_2]])

        %r0 = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        "mqtref.deallocQubitRegister"(%r0) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if the InsertOp is correctly inserted using a dynamic operand before deallocating a qubit register.
module {
    // CHECK-LABEL: func.func @testConvertInsertOpOperand
    func.func @testConvertInsertOpOperand() {
        // CHECK: %[[r_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}>
        // CHECK: %[[index:.*]] = arith.constant 0
        // CHECK: %[[r_1:.*]], %[[q_0:.*]] = "mqtopt.extractQubit"(%[[ANY:.*]], %[[index]]) : (!mqtopt.QubitRegister, i64) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        // CHECK: %[[r_2:.*]] = "mqtopt.insertQubit"(%[[r_1]], %[[q_0]], %[[index]])  : (!mqtopt.QubitRegister, !mqtopt.Qubit, i64) -> !mqtopt.QubitRegister
        // CHECK: "mqtopt.deallocQubitRegister"(%[[r_2]])

        %r0 = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtref.QubitRegister
        %index = arith.constant 0 : i64
         %q0 = "mqtref.extractQubit"(%r0, %index) : (!mqtref.QubitRegister, i64) -> !mqtref.Qubit
        "mqtref.deallocQubitRegister"(%r0) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if the InsertOp is correctly inserted before deallocating multiple qubit registers.
module {
    // CHECK-LABEL: func.func @testConvertInsertOptMultipleRegisters
    func.func @testConvertInsertOptMultipleRegisters() {
        // CHECK: %[[r0_1:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}>
        // CHECK: %[[r1_1:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 1 : i64}>
        // CHECK: %[[r0_2:.*]], %[[q0_1:.*]] = "mqtopt.extractQubit"(%[[r0_1]]) <{index_attr = 0 : i64}>
        // CHECK: %[[r1_2:.*]], %[[q1_1:.*]] = "mqtopt.extractQubit"(%[[r1_1]]) <{index_attr = 0 : i64}>
        // CHECK: %[[r0_3:.*]] = "mqtopt.insertQubit"(%[[r0_2]], %[[q0_1]])  <{index_attr = 0 : i64}>
        // CHECK: "mqtopt.deallocQubitRegister"(%[[r0_3]])
        // CHECK: %[[r1_3:.*]] = "mqtopt.insertQubit"(%[[r1_2]], %[[q1_1]])  <{index_attr = 0 : i64}>
        // CHECK: "mqtopt.deallocQubitRegister"(%[[r1_3]])

        %r0 = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtref.QubitRegister
        %r1 = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q1 = "mqtref.extractQubit"(%r1) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        "mqtref.deallocQubitRegister"(%r0) : (!mqtref.QubitRegister) -> ()
        "mqtref.deallocQubitRegister"(%r1) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if the MeasureOp is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertMeasureOp
    func.func @testConvertMeasureOp() {
        // CHECK: %[[q_0:.*]], [[m_0:.*]] = "mqtopt.measure"(%[[ANY:.*]])

        %r0 = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %m0 = mqtref.measure %q0
        "mqtref.deallocQubitRegister"(%r0) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if the ResetOp is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertResetOp
    func.func @testConvertResetOp() {
        // CHECK: %[[q_0:.*]] = "mqtopt.reset"(%[[ANY:.*]])

        %r0 = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        "mqtref.reset"(%q0) : (!mqtref.Qubit) -> ()
        "mqtref.deallocQubitRegister"(%r0) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if single qubit gates are converted correctly
module {
    // CHECK-LABEL: func.func @testConvertSingleQubitOp
    func.func @testConvertSingleQubitOp() {
        // CHECK: %[[q_0:.*]] = mqtopt.i() %[[ANY:.*]] : !mqtopt.Qubit
        // CHECK: %[[q_1:.*]] = mqtopt.h() %[[q_0]] : !mqtopt.Qubit
        // CHECK: %[[q_2:.*]] = mqtopt.x() %[[q_1]] : !mqtopt.Qubit
        // CHECK: %[[q_3:.*]] = mqtopt.y() %[[q_2]] : !mqtopt.Qubit
        // CHECK: %[[q_4:.*]] = mqtopt.z() %[[q_3]] : !mqtopt.Qubit
        // CHECK: %[[q_5:.*]] = mqtopt.s() %[[q_4]] : !mqtopt.Qubit
        // CHECK: %[[q_6:.*]] = mqtopt.sdg() %[[q_5]] : !mqtopt.Qubit
        // CHECK: %[[q_7:.*]] = mqtopt.t() %[[q_6]] : !mqtopt.Qubit
        // CHECK: %[[q_8:.*]] = mqtopt.tdg() %[[q_7]] : !mqtopt.Qubit
        // CHECK: %[[q_9:.*]] = mqtopt.v() %[[q_8]] : !mqtopt.Qubit
        // CHECK: %[[q_10:.*]] = mqtopt.vdg() %[[q_9]] : !mqtopt.Qubit
        // CHECK: %[[q_11:.*]] = mqtopt.sx() %[[q_10]] : !mqtopt.Qubit
        // CHECK: %[[q_12:.*]] = mqtopt.sxdg() %[[q_11]] : !mqtopt.Qubit

        %r0 = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit

        mqtref.i() %q0
        mqtref.h() %q0
        mqtref.x() %q0
        mqtref.y() %q0
        mqtref.z() %q0
        mqtref.s() %q0
        mqtref.sdg() %q0
        mqtref.t() %q0
        mqtref.tdg() %q0
        mqtref.v() %q0
        mqtref.vdg() %q0
        mqtref.sx() %q0
        mqtref.sxdg() %q0

        "mqtref.deallocQubitRegister"(%r0) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if two target gates are converted correctly
module {
    // CHECK-LABEL: func.func @testConvertTwoTargetOp
    func.func @testConvertTwoTargetOp() {
        // CHECK: %[[q01_1:.*]]:2 = mqtopt.swap() %[[ANY:.*]], %[[ANY:.*]] : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[q01_2:.*]]:2 = mqtopt.iswap() %[[q01_1]]#0, %[[q01_1]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[q01_3:.*]]:2 = mqtopt.iswapdg() %[[q01_2]]#0, %[[q01_2]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[q01_4:.*]]:2 = mqtopt.peres() %[[q01_3]]#0, %[[q01_3]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[q01_5:.*]]:2 = mqtopt.peresdg() %[[q01_4]]#0, %[[q01_4]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[q01_6:.*]]:2 = mqtopt.dcx() %[[q01_5]]#0, %[[q01_5]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[q01_7:.*]]:2 = mqtopt.ecr() %[[q01_6]]#0, %[[q01_6]]#1 : !mqtopt.Qubit, !mqtopt.Qubit

        %r0 = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q1 = "mqtref.extractQubit"(%r0) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit

        mqtref.swap() %q0, %q1
        mqtref.iswap() %q0, %q1
        mqtref.iswapdg() %q0, %q1
        mqtref.peres() %q0, %q1
        mqtref.peresdg() %q0, %q1
        mqtref.dcx() %q0, %q1
        mqtref.ecr() %q0, %q1

        "mqtref.deallocQubitRegister"(%r0) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if parameterized single qubit gates are converted correctly
module {
    // CHECK-LABEL: func.func @testSingleQubitRotationOp
    func.func @testSingleQubitRotationOp() {
        // CHECK: %[[c_0:.*]] = arith.constant 3.000000e-01
        // CHECK: %[[q_0:.*]] = mqtopt.u(%[[c_0]], %[[c_0]], %[[c_0]]) %[[ANY:.*]] : !mqtopt.Qubit
        // CHECK: %[[q_1:.*]] = mqtopt.u2(%[[c_0]], %[[c_0]]) %[[q_0]] : !mqtopt.Qubit
        // CHECK: %[[q_2:.*]] = mqtopt.p(%[[c_0]]) %[[q_1]] : !mqtopt.Qubit
        // CHECK: %[[q_3:.*]] = mqtopt.rx(%[[c_0]]) %[[q_2]] : !mqtopt.Qubit
        // CHECK: %[[q_4:.*]] = mqtopt.ry(%[[c_0]]) %[[q_3]] : !mqtopt.Qubit
        // CHECK: %[[q_5:.*]] = mqtopt.rz(%[[c_0]]) %[[q_4]] : !mqtopt.Qubit

        %cst = arith.constant 3.000000e-01 : f64
        %r0 = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit

        mqtref.u(%cst, %cst, %cst) %q0
        mqtref.u2(%cst, %cst) %q0
        mqtref.p(%cst) %q0
        mqtref.rx(%cst) %q0
        mqtref.ry(%cst) %q0
        mqtref.rz(%cst) %q0

        "mqtref.deallocQubitRegister"(%r0) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if parameterized multiple qubit gates are converted correctly
module {
    // CHECK-LABEL: func.func @testMultipleQubitRotationOp
    func.func @testMultipleQubitRotationOp() {
        // CHECK: %[[c_0:.*]] = arith.constant 3.000000e-01 : f64
        // CHECK: %[[q01_1:.*]]:2 = mqtopt.rxx(%[[c_0]]) %[[ANY:.*]], %[[ANY:.*]] : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[q01_2:.*]]:2 = mqtopt.ryy(%[[c_0]]) %[[q01_1]]#0, %[[q01_1]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[q01_3:.*]]:2 = mqtopt.rzz(%[[c_0]]) %[[q01_2]]#0, %[[q01_2]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[q01_4:.*]]:2 = mqtopt.rzx(%[[c_0]]) %[[q01_3]]#0, %[[q01_3]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[q01_5:.*]]:2 = mqtopt.xxminusyy(%[[c_0]], %[[c_0]]) %[[q01_4]]#0, %[[q01_4]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
        // CHECK: %[[q01_6:.*]]:2 = mqtopt.xxplusyy(%[[c_0]], %[[c_0]]) %[[q01_5]]#0, %[[q01_5]]#1 : !mqtopt.Qubit, !mqtopt.Qubit

        %cst = arith.constant 3.000000e-01 : f64
        %r0 = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q1 = "mqtref.extractQubit"(%r0) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit

        mqtref.rxx(%cst) %q0, %q1
        mqtref.ryy(%cst) %q0, %q1
        mqtref.rzz(%cst) %q0, %q1
        mqtref.rzx(%cst) %q0, %q1
        mqtref.xxminusyy(%cst, %cst) %q0, %q1
        mqtref.xxplusyy(%cst, %cst) %q0, %q1

        "mqtref.deallocQubitRegister"(%r0) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if static params and paramask is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertStaticParams
    func.func @testConvertStaticParams() {
        // CHECK: %[[q_0:.*]] = mqtopt.u(%[[ANY:.*]], %[[ANY:.*]] static [3.000000e-01] mask [false, true, false]) %[[ANY:.*]]

        %r0 = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit

        %cst = arith.constant 3.000000e-01 : f64
        mqtref.u(%cst, %cst static [3.000000e-01] mask [false, true, false]) %q0

        "mqtref.deallocQubitRegister"(%r0) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a controlled op is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertControlledOp
    func.func @testConvertControlledOp() {
        // CHECK: %[[q0_1:.*]], %[[q1_1:.*]] = mqtopt.x() %[[ANY:.*]] ctrl %[[ANY:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit

        %r0 = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q1 = "mqtref.extractQubit"(%r0) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit

        mqtref.x() %q1 ctrl %q0

        "mqtref.deallocQubitRegister"(%r0) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a negative controlled op is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertNegativeControlledOp
    func.func @testConvertNegativeControlledOp() {
        // CHECK: %[[q0_1:.*]], %[[q1_1:.*]] = mqtopt.x() %[[ANY:.*]] nctrl %[[ANY:.*]] : !mqtopt.Qubit nctrl !mqtopt.Qubit

        %r0 = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q1 = "mqtref.extractQubit"(%r0) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit

        mqtref.x() %q1 nctrl %q0

        "mqtref.deallocQubitRegister"(%r0) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a Bell state is converted correctly.
module {
    // CHECK-LABEL: func.func @bellConvertState()
    func.func @bellConvertState() {
        // CHECK: %[[r_0:.*]] = "mqtopt.allocQubitRegister"
        // CHECK: %[[r_1:.*]], %[[q0_1:.*]] = "mqtopt.extractQubit"(%[[r_0]]) <{index_attr = 0 : i64}>
        // CHECK: %[[r_2:.*]], %[[q1_1:.*]] = "mqtopt.extractQubit"(%[[r_1]]) <{index_attr = 1 : i64}>
        // CHECK: %[[q0_2:.*]] = mqtopt.h() %[[q0_1]] : !mqtopt.Qubit
        // CHECK: %[[q1_2:.*]], %[[q0_3:.*]] = mqtopt.x() %[[q1_1:.*]] ctrl %[[q0_2:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
        // CHECK: %[[q0_4:.*]], [[m0_0:.*]] = "mqtopt.measure"(%[[q0_3]])
        // CHECK: %[[q1_3:.*]], %[[m1_0:.*]] = "mqtopt.measure"(%[[q1_2]]) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        // CHECK: %[[r_3:.*]] = "mqtopt.insertQubit"(%[[r_2]], %[[q0_4]]) <{index_attr = 0 : i64}>
        // CHECK: %[[r_4:.*]] = "mqtopt.insertQubit"(%[[r_3]], %[[q1_3]]) <{index_attr = 1 : i64}>
        // CHECK: "mqtopt.deallocQubitRegister"(%[[r_4]]) : (!mqtopt.QubitRegister) -> ()

        %r0 = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q1 = "mqtref.extractQubit"(%r0) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit

        mqtref.h() %q0
        mqtref.x() %q1 ctrl %q0
        %m0 = mqtref.measure %q0
        %m1 = mqtref.measure %q1

        "mqtref.deallocQubitRegister"(%r0) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a gphaseOp with no target no controlled qubit is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertGPhaseOp()
    func.func @testConvertGPhaseOp() {
        // CHECK: %[[c_0:.*]] = arith.constant 3.000000e-01 : f64
        // mqtopt.gphase(%[[c_0]])

        %cst = arith.constant 3.000000e-01 : f64
        mqtref.gphase(%cst)
        return
    }
}

// -----
// This test checks if a gphaseOp with a controlled qubit is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertGPhaseOpControlled()
    func.func @testConvertGPhaseOpControlled() {
        // CHECK: %[[c_0:.*]] = arith.constant 3.000000e-01 : f64
        // CHECK: %[[q_0:.*]] = mqtopt.gphase(%[[c_0]]) ctrl %[[ANY:.*]] : ctrl !mqtopt.Qubit

        %r0 = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %cst = arith.constant 3.000000e-01 : f64
        mqtref.gphase(%cst) ctrl %q0
        "mqtref.deallocQubitRegister"(%r0) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a gphaseOp with a positive controlled qubit and a negative controlled qubit is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertGPhaseOpPositiveNegativeControlled()
    func.func @testConvertGPhaseOpPositiveNegativeControlled() {
        // CHECK: %[[c_0:.*]] = arith.constant 3.000000e-01 : f64
        // CHECK: %[[q0_1:.*]], %[[q1_1:.*]] = mqtopt.gphase(%[[c_0]]) ctrl %[[ANY:.*]] nctrl %[[ANY:.*]] : ctrl !mqtopt.Qubit nctrl !mqtopt.Qubit

        %r0 = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q1 = "mqtref.extractQubit"(%r0) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %cst = arith.constant 3.000000e-01 : f64
        mqtref.gphase(%cst) ctrl %q0 nctrl %q1
        "mqtref.deallocQubitRegister"(%r0) : (!mqtref.QubitRegister) -> ()
        return
    }
}


// -----
// This test checks if a barrierOp is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertBarrierOp()
    func.func @testConvertBarrierOp() {
        // CHECK: %[[q_0:.*]] = mqtopt.barrier() %[[ANY:.*]] : !mqtopt.Qubit

        %r0 = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        mqtref.barrier() %q0
        "mqtref.deallocQubitRegister"(%r0) : (!mqtref.QubitRegister) -> ()
        return
    }
}
// -----
// This test checks if a barrierOp with multiple inputs is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertBarrierOpMultipleInputs()
    func.func @testConvertBarrierOpMultipleInputs() {
        // CHECK: %[[q01_1:.*]]:2 = mqtopt.barrier() %[[ANY:.*]], %[[ANY:.*]] : !mqtopt.Qubit, !mqtopt.Qubit

        %r0 = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q1 = "mqtref.extractQubit"(%r0) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        mqtref.barrier() %q0, %q1
        "mqtref.deallocQubitRegister"(%r0) : (!mqtref.QubitRegister) -> ()
        return
    }
}
