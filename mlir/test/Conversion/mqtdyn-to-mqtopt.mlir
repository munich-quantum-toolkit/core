// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --mqtdyn-to-mqtopt | FileCheck %s

// -----
// This test checks if the AllocOp is converted correctly using a static attribute.
module {
    // CHECK-LABEL: func.func @testConvertAllocOpAttribute()
    func.func @testConvertAllocOpAttribute() {
        // CHECK: %[[r_0:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}>

        %r0 = "mqtdyn.allocQubitRegister" () {"size_attr" = 2 : i64} : () -> !mqtdyn.QubitRegister
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
        %r0 = "mqtdyn.allocQubitRegister" (%size) : (i64) -> !mqtdyn.QubitRegister
        return
    }
}

// -----
// This test checks if the DeallocOp is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertDeallocOp
    func.func @testConvertDeallocOp() {
        // CHECK: "mqtopt.deallocQubitRegister"(%[[ANY:.*]])

        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if the ExtractOp is converted correctly using a static attribute.
module {
    // CHECK-LABEL: func.func @testConvertExtractOpAttribute
    func.func @testConvertExtractOpAttribute() {
        // CHECK: %[[r_0:.*]], %[[q_0:.*]] = "mqtopt.extractQubit"(%[[ANY:.*]]) <{index_attr = 0 : i64}>

        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
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
        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0, %index) : (!mqtdyn.QubitRegister, i64) -> !mqtdyn.Qubit
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

        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%r0) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q2 = "mqtdyn.extractQubit"(%r0) <{index_attr = 2 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
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

        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
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

        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
        %index = arith.constant 0 : i64
         %q0 = "mqtdyn.extractQubit"(%r0, %index) : (!mqtdyn.QubitRegister, i64) -> !mqtdyn.Qubit
        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
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

        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
        %r1 = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%r1) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
        "mqtdyn.deallocQubitRegister"(%r1) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if the MeasureOp is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertMeasureOp
    func.func @testConvertMeasureOp() {
        // CHECK: %[[q_0:.*]], [[m_0:.*]] = "mqtopt.measure"(%[[ANY:.*]])

        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %m0 = "mqtdyn.measure"(%q0) : (!mqtdyn.Qubit) -> i1
        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if the ExtractOp is converted correctly for multiple qubits
module {
    // CHECK-LABEL: func.func @testConvertMeasureOpOnMultipleInputs
    func.func @testConvertMeasureOpOnMultipleInputs() {
         // CHECK: %[[q01_1:.*]]:2, [[m01_1:.*]]:2 = "mqtopt.measure"(%[[ANY:.*]], %[[ANY:.*]])

        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%r0) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        %m:2 = "mqtdyn.measure"(%q0, %q1) : (!mqtdyn.Qubit, !mqtdyn.Qubit) -> (i1, i1)
        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
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

        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.i() %q0
        mqtdyn.h() %q0
        mqtdyn.x() %q0
        mqtdyn.y() %q0
        mqtdyn.z() %q0
        mqtdyn.s() %q0
        mqtdyn.sdg() %q0
        mqtdyn.t() %q0
        mqtdyn.tdg() %q0
        mqtdyn.v() %q0
        mqtdyn.vdg() %q0
        mqtdyn.sx() %q0
        mqtdyn.sxdg() %q0

        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
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

        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%r0) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.swap() %q0, %q1
        mqtdyn.iswap() %q0, %q1
        mqtdyn.iswapdg() %q0, %q1
        mqtdyn.peres() %q0, %q1
        mqtdyn.peresdg() %q0, %q1
        mqtdyn.dcx() %q0, %q1
        mqtdyn.ecr() %q0, %q1

        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
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

        %p0 = arith.constant 3.000000e-01 : f64
        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.u(%p0, %p0, %p0) %q0
        mqtdyn.u2(%p0, %p0) %q0
        mqtdyn.p(%p0) %q0
        mqtdyn.rx(%p0) %q0
        mqtdyn.ry(%p0) %q0
        mqtdyn.rz(%p0) %q0

        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
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

        %p0 = arith.constant 3.000000e-01 : f64
        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%r0) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.rxx(%p0) %q0, %q1
        mqtdyn.ryy(%p0) %q0, %q1
        mqtdyn.rzz(%p0) %q0, %q1
        mqtdyn.rzx(%p0) %q0, %q1
        mqtdyn.xxminusyy(%p0, %p0) %q0, %q1
        mqtdyn.xxplusyy(%p0, %p0) %q0, %q1

        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if static params and paramask is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertStaticParams
    func.func @testConvertStaticParams() {
        // CHECK: %[[q_0:.*]] = mqtopt.u(%[[ANY:.*]], %[[ANY:.*]] static [3.000000e-01] mask [false, true, false]) %[[ANY:.*]]

        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        %cst = arith.constant 3.000000e-01 : f64
        mqtdyn.u(%cst, %cst static [3.000000e-01] mask [false, true, false]) %q0

        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a controlled op is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertControlledOp
    func.func @testConvertControlledOp() {
        // CHECK: %[[q0_1:.*]], %[[q1_1:.*]] = mqtopt.x() %[[ANY:.*]] ctrl %[[ANY:.*]] : !mqtopt.Qubit ctrl !mqtopt.Qubit

        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%r0) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.x() %q1 ctrl %q0

        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a negative controlled op is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertNegativeControlledOp
    func.func @testConvertNegativeControlledOp() {
        // CHECK: %[[q0_1:.*]], %[[q1_1:.*]] = mqtopt.x() %[[ANY:.*]] nctrl %[[ANY:.*]] : !mqtopt.Qubit nctrl !mqtopt.Qubit

        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%r0) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.x() %q1 nctrl %q0

        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a Bell state is converted correctly.
module {
    // CHECK-LABEL: func.func @bellState()
    func.func @bellState() {
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

        %r0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%r0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%r0) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.h() %q0
        mqtdyn.x() %q1 ctrl %q0
        %m0 = "mqtdyn.measure"(%q0) : (!mqtdyn.Qubit) -> i1
        %m1 = "mqtdyn.measure"(%q1) : (!mqtdyn.Qubit) -> i1

        "mqtdyn.deallocQubitRegister"(%r0) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}
