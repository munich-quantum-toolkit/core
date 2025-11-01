// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --mqtopt-to-mqtref | FileCheck %s

// -----
// This test checks if a non-!mqtref.Qubit is not converted.
module {
    // CHECK-LABEL: func.func @testDoNotConvertMemRef()
    func.func @testDoNotConvertMemRef() {
        // CHECK: %[[I0:.*]] = arith.constant 0 : index
        // CHECK: %[[Memref:.*]] = memref.alloc() : memref<1xi1>
        // CHECK: %[[C:.*]] = memref.load %[[Memref]][%[[I0]]] : memref<1xi1
        // CHECK: memref.store %[[C]], %[[Memref]][%[[I0]]] : memref<1xi1
        // CHECK: memref.dealloc %[[Memref]] : memref<1xi1>

        %i0 = arith.constant 0 : index
        %memref = memref.alloc() : memref<1xi1>
        %0 = memref.load %memref[%i0] : memref<1xi1>
        memref.store %0, %memref[%i0] : memref<1xi1>
        memref.dealloc %memref : memref<1xi1>

        return
    }
}

// -----
// This test checks if the AllocOp is converted correctly using a static attribute.
module {
    // CHECK-LABEL: func.func @testConvertAllocOpStatic()
    func.func @testConvertAllocOpStatic() {
        // CHECK: %[[Qreg:.*]] = memref.alloc() : memref<2x!mqtref.Qubit>

        %qreg = memref.alloc() : memref<2x!mqtopt.Qubit>
        memref.dealloc %qreg : memref<2x!mqtopt.Qubit>

        return
    }
}

// -----
// This test checks if the AllocOp is converted correctly using a dynamic operand.
module {
    // CHECK-LABEL: func.func @testConvertAllocOpDynamic()
    func.func @testConvertAllocOpDynamic() {
        // CHECK: %[[I2:.*]] = arith.constant 2 : index
        // CHECK: %[[Qreg:.*]] = memref.alloc(%[[I2]]) : memref<?x!mqtref.Qubit>

        %i2 = arith.constant 2 : index
        %qreg = memref.alloc(%i2) : memref<?x!mqtopt.Qubit>
        memref.dealloc %qreg : memref<?x!mqtopt.Qubit>

        return
    }
}

// -----
// This test checks if the DeallocOp is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertDeallocOp
    func.func @testConvertDeallocOp() {
        // CHECK: memref.dealloc %[[ANY:.*]] : memref<2x!mqtref.Qubit>

        %qreg = memref.alloc() : memref<2x!mqtopt.Qubit>
        memref.dealloc %qreg : memref<2x!mqtopt.Qubit>

        return
    }
}

// -----
// This test checks if the LoadOp is converted correctly.
module {
    // CHECK-LABEL: func.func @testConvertLoadOp
    func.func @testConvertLoadOp() {
        // CHECK: %[[I0:.*]] = arith.constant 0 : index
        // CHECK: %[[Q0:.*]] = memref.load %[[ANY:.*]][%[[I0]]] : memref<1x!mqtref.Qubit>
        // CHECK-NOT: memref.store %[[Q0:.*]], %[[ANY:.*]][%[[I0]]] : memref<1x!mqtref.Qubit>

        %i0 = arith.constant 0 : index
        %qreg = memref.alloc() : memref<1x!mqtopt.Qubit>
        %q0 = memref.load %qreg[%i0] : memref<1x!mqtopt.Qubit>
        memref.store %q0, %qreg[%i0] : memref<1x!mqtopt.Qubit>
        memref.dealloc %qreg : memref<1x!mqtopt.Qubit>

        return
    }
}

// -----
// This test checks if the operands/results are correctly replaced for the def-use chain.
module {
    // CHECK-LABEL: func.func @testConvertOperandChain
    func.func @testConvertOperandChain() {
        // CHECK: %[[I2:.*]] = arith.constant 2 : index
        // CHECK: %[[I1:.*]] = arith.constant 1 : index
        // CHECK: %[[I0:.*]] = arith.constant 0 : index
        // CHECK: %[[Qreg:.*]] = memref.alloc() : memref<3x!mqtref.Qubit>
        // CHECK: %[[Q0:.*]] = memref.load %[[Qreg]][%[[I0]]] : memref<3x!mqtref.Qubit>
        // CHECK: %[[Q1:.*]] = memref.load %[[Qreg]][%[[I1]]] : memref<3x!mqtref.Qubit>
        // CHECK: %[[Q2:.*]] = memref.load %[[Qreg]][%[[I2]]] : memref<3x!mqtref.Qubit>
        // CHECK: memref.dealloc %[[Qreg]] : memref<3x!mqtref.Qubit>

        %i2 = arith.constant 2 : index
        %i1 = arith.constant 1 : index
        %i0 = arith.constant 0 : index
        %qreg = memref.alloc() : memref<3x!mqtopt.Qubit>
        %q0 = memref.load %qreg[%i0] : memref<3x!mqtopt.Qubit>
        %q1 = memref.load %qreg[%i1] : memref<3x!mqtopt.Qubit>
        %q2 = memref.load %qreg[%i2] : memref<3x!mqtopt.Qubit>
        memref.store %q0, %qreg[%i0] : memref<3x!mqtopt.Qubit>
        memref.store %q1, %qreg[%i1] : memref<3x!mqtopt.Qubit>
        memref.store %q2, %qreg[%i2] : memref<3x!mqtopt.Qubit>
        memref.dealloc %qreg : memref<3x!mqtopt.Qubit>

        return
    }
}

// -----
// This test checks if the MeasureOp is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertMeasureOp
    func.func @testConvertMeasureOp() {
        // CHECK: %[[m_0:.*]] = mqtref.measure %[[ANY:.*]]
        // CHECK: memref.store %[[m_0]], %[[ANY:.*]][%[[ANY:.*]]] : memref<1xi1>

        %i0 = arith.constant 0 : index
        %qreg = memref.alloc() : memref<1x!mqtopt.Qubit>
        %creg = memref.alloca() : memref<1xi1>
        %q0 = memref.load %qreg[%i0] : memref<1x!mqtopt.Qubit>

        %q1, %m0 = mqtopt.measure %q0
        memref.store %m0, %creg[%i0] : memref<1xi1>

        memref.store %q1, %qreg[%i0] : memref<1x!mqtopt.Qubit>
        memref.dealloc %qreg : memref<1x!mqtopt.Qubit>

        return
    }
}

// -----
// This test checks if the ResetOp is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertResetOp
    func.func @testConvertResetOp() {
        // CHECK: mqtref.reset %[[ANY:.*]]

        %i0 = arith.constant 0 : index
        %qreg = memref.alloc() : memref<1x!mqtopt.Qubit>
        %q0 = memref.load %qreg[%i0] : memref<1x!mqtopt.Qubit>

        %q1 = mqtopt.reset %q0

        memref.store %q1, %qreg[%i0] : memref<1x!mqtopt.Qubit>
        memref.dealloc %qreg : memref<1x!mqtopt.Qubit>

        return
    }
}

// -----
// This test checks if single-qubit gates are converted correctly
module {
    // CHECK-LABEL: func.func @testConvertSingleQubitOp
    func.func @testConvertSingleQubitOp() {
        // CHECK: mqtref.i() %[[q_0:.*]]
        // CHECK: mqtref.h() %[[q_0]]
        // CHECK: mqtref.x() %[[q_0]]
        // CHECK: mqtref.y() %[[q_0]]
        // CHECK: mqtref.z() %[[q_0]]
        // CHECK: mqtref.s() %[[q_0]]
        // CHECK: mqtref.sdg() %[[q_0]]
        // CHECK: mqtref.t() %[[q_0]]
        // CHECK: mqtref.tdg() %[[q_0]]
        // CHECK: mqtref.v() %[[q_0]]
        // CHECK: mqtref.vdg() %[[q_0]]
        // CHECK: mqtref.sx() %[[q_0]]
        // CHECK: mqtref.sxdg() %[[q_0]]

        %i0 = arith.constant 0 : index
        %qreg = memref.alloc() : memref<1x!mqtopt.Qubit>
        %q0 = memref.load %qreg[%i0] : memref<1x!mqtopt.Qubit>

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

        memref.store %q13, %qreg[%i0] : memref<1x!mqtopt.Qubit>
        memref.dealloc %qreg : memref<1x!mqtopt.Qubit>

        return
    }
}

// -----
// This test checks if two-qubit gates are converted correctly
module {
    // CHECK-LABEL: func.func @testConvertTwoQubitOp
    func.func @testConvertTwoQubitOp() {
        // CHECK: mqtref.swap() %[[q_0:.*]], %[[q_1:.*]]
        // CHECK: mqtref.iswap() %[[q_0]], %[[q_1]]
        // CHECK: mqtref.iswapdg() %[[q_0]], %[[q_1]]
        // CHECK: mqtref.peres() %[[q_0]], %[[q_1]]
        // CHECK: mqtref.peresdg() %[[q_0]], %[[q_1]]
        // CHECK: mqtref.dcx() %[[q_0]], %[[q_1]]
        // CHECK: mqtref.ecr() %[[q_0]], %[[q_1]]

        %i1 = arith.constant 1 : index
        %i0 = arith.constant 0 : index
        %qreg = memref.alloc() : memref<2x!mqtopt.Qubit>
        %q0_0 = memref.load %qreg[%i0] : memref<2x!mqtopt.Qubit>
        %q1_0 = memref.load %qreg[%i1] : memref<2x!mqtopt.Qubit>

        %q0_1, %q1_1 = mqtopt.swap() %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
        %q0_2, %q1_2 = mqtopt.iswap() %q0_1, %q1_1 : !mqtopt.Qubit, !mqtopt.Qubit
        %q0_3, %q1_3 = mqtopt.iswapdg() %q0_2, %q1_2 : !mqtopt.Qubit, !mqtopt.Qubit
        %q0_4, %q1_4 = mqtopt.peres() %q0_3, %q1_3 : !mqtopt.Qubit, !mqtopt.Qubit
        %q0_5, %q1_5 = mqtopt.peresdg() %q0_4, %q1_4 : !mqtopt.Qubit, !mqtopt.Qubit
        %q0_6, %q1_6 = mqtopt.dcx() %q0_5, %q1_5 : !mqtopt.Qubit, !mqtopt.Qubit
        %q0_7, %q1_7 = mqtopt.ecr() %q0_6, %q1_6 : !mqtopt.Qubit, !mqtopt.Qubit

        memref.store %q0_7, %qreg[%i0] : memref<2x!mqtopt.Qubit>
        memref.store %q1_7, %qreg[%i1] : memref<2x!mqtopt.Qubit>
        memref.dealloc %qreg : memref<2x!mqtopt.Qubit>

        return
    }
}

// -----
// This test checks if controlled gates are converted correctly
module {
    // CHECK-LABEL: func.func @testConvertControlledOp
    func.func @testConvertControlledOp() {
        // CHECK: %[[I2:.*]] = arith.constant 2 : index
        // CHECK: %[[I1:.*]] = arith.constant 1 : index
        // CHECK: %[[I0:.*]] = arith.constant 0 : index
        // CHECK: %[[Qreg:.*]] = memref.alloc() : memref<3x!mqtref.Qubit>
        // CHECK: %[[Q0:.*]] = memref.load %[[Qreg]][%[[I0]]] : memref<3x!mqtref.Qubit>
        // CHECK: %[[Q1:.*]] = memref.load %[[Qreg]][%[[I1]]] : memref<3x!mqtref.Qubit>
        // CHECK: %[[Q2:.*]] = memref.load %[[Qreg]][%[[I2]]] : memref<3x!mqtref.Qubit>
        // CHECK: mqtref.x() %[[Q1]] ctrl %[[Q2]] nctrl %[[Q0]]
        // CHECK: mqtref.swap() %[[Q1]], %[[Q0]] ctrl %[[Q2]]
        // CHECK: memref.dealloc %[[Qreg]] : memref<3x!mqtref.Qubit>

        %i2 = arith.constant 2 : index
        %i1 = arith.constant 1 : index
        %i0 = arith.constant 0 : index
        %qreg = memref.alloc() : memref<3x!mqtopt.Qubit>
        %q0_0 = memref.load %qreg[%i0] : memref<3x!mqtopt.Qubit>
        %q1_0 = memref.load %qreg[%i1] : memref<3x!mqtopt.Qubit>
        %q2_0 = memref.load %qreg[%i2] : memref<3x!mqtopt.Qubit>

        %q1_1, %q2_1, %q0_1 = mqtopt.x() %q1_0 ctrl %q2_0 nctrl %q0_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit nctrl !mqtopt.Qubit
        %q1_2, %q0_2, %q2_2 = mqtopt.swap() %q1_1, %q0_1 ctrl %q2_1 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit

        memref.store %q0_2, %qreg[%i0] : memref<3x!mqtopt.Qubit>
        memref.store %q1_2, %qreg[%i1] : memref<3x!mqtopt.Qubit>
        memref.store %q2_2, %qreg[%i2] : memref<3x!mqtopt.Qubit>
        memref.dealloc %qreg : memref<3x!mqtopt.Qubit>

        return
    }
}

// -----
// This test checks if parameterized single qubit gates are converted correctly
module {
    // CHECK-LABEL: func.func @testSingleQubitRotationOp
    func.func @testSingleQubitRotationOp() {
        // CHECK: %[[c_0:.*]] = arith.constant 3.000000e-01
        // CHECK: mqtref.u(%[[c_0]], %[[c_0]], %[[c_0]]) %[[q_0:.*]]
        // CHECK: mqtref.u2(%[[c_0]], %[[c_0]]) %[[q_0]]
        // CHECK: mqtref.p(%[[c_0]]) %[[q_0]]
        // CHECK: mqtref.rx(%[[c_0]]) %[[q_0]]
        // CHECK: mqtref.ry(%[[c_0]]) %[[q_0]]
        // CHECK: mqtref.rz(%[[c_0]]) %[[q_0]]
        // CHECK: mqtref.r(%[[c_0]], %[[c_0]]) %[[q_0]]

        %i0 = arith.constant 0 : index
        %qreg = memref.alloc() : memref<1x!mqtopt.Qubit>
        %q0 = memref.load %qreg[%i0] : memref<1x!mqtopt.Qubit>

        %cst = arith.constant 3.000000e-01 : f64
        %q1 = mqtopt.u(%cst, %cst, %cst) %q0 : !mqtopt.Qubit
        %q2 = mqtopt.u2(%cst, %cst) %q1 : !mqtopt.Qubit
        %q3 = mqtopt.p(%cst) %q2 : !mqtopt.Qubit
        %q4 = mqtopt.rx(%cst) %q3 : !mqtopt.Qubit
        %q5 = mqtopt.ry(%cst) %q4 : !mqtopt.Qubit
        %q6 = mqtopt.rz(%cst) %q5 : !mqtopt.Qubit
        %q7 = mqtopt.r(%cst, %cst) %q6 : !mqtopt.Qubit

        memref.store %q7, %qreg[%i0] : memref<1x!mqtopt.Qubit>
        memref.dealloc %qreg : memref<1x!mqtopt.Qubit>

        return
    }
}

// -----
// This test checks if parameterized multiple qubit gates are converted correctly
module {
    // CHECK-LABEL: func.func @testMultipleQubitRotationOp
    func.func @testMultipleQubitRotationOp() {
        // CHECK: %[[c_0:.*]] = arith.constant 3.000000e-01 : f64
        // CHECK: mqtref.rxx(%[[c_0]]) %[[q_0:.*]], %[[q_1:.*]]
        // CHECK: mqtref.ryy(%[[c_0]]) %[[q_0]], %[[q_1]]
        // CHECK: mqtref.rzz(%[[c_0]]) %[[q_0]], %[[q_1]]
        // CHECK: mqtref.rzx(%[[c_0]]) %[[q_0]], %[[q_1]]
        // CHECK: mqtref.xx_minus_yy(%[[c_0]], %[[c_0]]) %[[q_0]], %[[q_1]]
        // CHECK: mqtref.xx_plus_yy(%[[c_0]], %[[c_0]]) %[[q_0]], %[[q_1]]

        %i1 = arith.constant 1 : index
        %i0 = arith.constant 0 : index
        %qreg = memref.alloc() : memref<2x!mqtopt.Qubit>
        %q0_0 = memref.load %qreg[%i0] : memref<2x!mqtopt.Qubit>
        %q1_0 = memref.load %qreg[%i1] : memref<2x!mqtopt.Qubit>

        %cst = arith.constant 3.000000e-01 : f64
        %q01_1:2 = mqtopt.rxx(%cst) %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
        %q01_2:2 = mqtopt.ryy(%cst) %q01_1#0, %q01_1#1 : !mqtopt.Qubit, !mqtopt.Qubit
        %q01_3:2 = mqtopt.rzz(%cst) %q01_2#0, %q01_2#1 : !mqtopt.Qubit, !mqtopt.Qubit
        %q01_4:2 = mqtopt.rzx(%cst) %q01_3#0, %q01_3#1 : !mqtopt.Qubit, !mqtopt.Qubit
        %q01_5:2 = mqtopt.xx_minus_yy(%cst, %cst) %q01_4#0, %q01_4#1 : !mqtopt.Qubit, !mqtopt.Qubit
        %q01_6:2 = mqtopt.xx_plus_yy(%cst, %cst) %q01_5#0, %q01_5#1 : !mqtopt.Qubit, !mqtopt.Qubit

        memref.store %q01_6#0, %qreg[%i0] : memref<2x!mqtopt.Qubit>
        memref.store %q01_6#1, %qreg[%i1] : memref<2x!mqtopt.Qubit>
        memref.dealloc %qreg : memref<2x!mqtopt.Qubit>

        return
    }
}

// -----
// This test checks if static params and paramask is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertStaticParams
    func.func @testConvertStaticParams() {
        // CHECK:  mqtref.u(%[[ANY:.*]], %[[ANY:.*]] static [3.000000e-01] mask [false, true, false]) %[[ANY:.*]]

        %i0 = arith.constant 0 : index
        %qreg = memref.alloc() : memref<1x!mqtopt.Qubit>
        %q0 = memref.load %qreg[%i0] : memref<1x!mqtopt.Qubit>

        %cst = arith.constant 3.000000e-01 : f64
        %q1 = mqtopt.u(%cst, %cst static [3.000000e-01] mask [false, true, false]) %q0 : !mqtopt.Qubit

        memref.store %q1, %qreg[%i0] : memref<1x!mqtopt.Qubit>
        memref.dealloc %qreg : memref<1x!mqtopt.Qubit>

        return
    }
}

// -----
// This test checks if a controlled op is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertControlledOp
    func.func @testConvertControlledOp() {
        // CHECK: mqtref.x() %[[q_0:.*]] ctrl %[[q_1:.*]]

        %i1 = arith.constant 1 : index
        %i0 = arith.constant 0 : index
        %qreg = memref.alloc() : memref<2x!mqtopt.Qubit>
        %q0_0 = memref.load %qreg[%i0] : memref<2x!mqtopt.Qubit>
        %q1_0 = memref.load %qreg[%i1] : memref<2x!mqtopt.Qubit>

        %q1_1, %q0_1 = mqtopt.x() %q1_0 ctrl %q0_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit

        memref.store %q0_1, %qreg[%i0] : memref<2x!mqtopt.Qubit>
        memref.store %q1_1, %qreg[%i1] : memref<2x!mqtopt.Qubit>
        memref.dealloc %qreg : memref<2x!mqtopt.Qubit>

        return
    }
}

// -----
// This test checks if a negative controlled op is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertNegativeControlledOp
    func.func @testConvertNegativeControlledOp() {
        // CHECK: mqtref.x() %[[q_0:.*]] nctrl %[[q_1:.*]]

        %i1 = arith.constant 1 : index
        %i0 = arith.constant 0 : index
        %qreg = memref.alloc() : memref<2x!mqtopt.Qubit>
        %q0_0 = memref.load %qreg[%i0] : memref<2x!mqtopt.Qubit>
        %q1_0 = memref.load %qreg[%i1] : memref<2x!mqtopt.Qubit>

        %q1_1, %q0_1 = mqtopt.x() %q1_0 nctrl %q0_0 : !mqtopt.Qubit nctrl !mqtopt.Qubit

        memref.store %q0_1, %qreg[%i0] : memref<2x!mqtopt.Qubit>
        memref.store %q1_1, %qreg[%i1] : memref<2x!mqtopt.Qubit>
        memref.dealloc %qreg : memref<2x!mqtopt.Qubit>

        return
    }
}

// -----
// This test checks if a Bell state is converted correctly.
module {
    // CHECK-LABEL: func.func @bellState()
    func.func @bellState() {
        // CHECK: %[[I1:.*]] = arith.constant 1 : index
        // CHECK: %[[I0:.*]] = arith.constant 0 : index
        // CHECK: %[[Qreg:.*]] = memref.alloc() : memref<2x!mqtref.Qubit>
        // CHECK: %[[Q0:.*]] = memref.load %[[Qreg]][%[[I0]]] : memref<2x!mqtref.Qubit>
        // CHECK: %[[Q1:.*]] = memref.load %[[Qreg]][%[[I1]]] : memref<2x!mqtref.Qubit>
        // CHECK: mqtref.h() %[[Q0]]
        // CHECK: mqtref.x() %[[Q1]] ctrl %[[Q0]]
        // CHECK: %[[M0:.*]] = mqtref.measure %[[Q0]]
        // CHECK: %[[M1:.*]] = mqtref.measure %[[Q1]]
        // CHECK: memref.dealloc %[[Qreg]] : memref<2x!mqtref.Qubit>

        %i1 = arith.constant 1 : index
        %i0 = arith.constant 0 : index
        %qreg = memref.alloc() : memref<2x!mqtopt.Qubit>
        %q0_0 = memref.load %qreg[%i0] : memref<2x!mqtopt.Qubit>
        %q1_0 = memref.load %qreg[%i1] : memref<2x!mqtopt.Qubit>

        %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
        %q1_1, %q0_2 = mqtopt.x() %q1_0 ctrl %q0_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
        %q0_3, %m0 = mqtopt.measure %q0_2
        %q1_2, %m1 = mqtopt.measure %q1_1

        memref.store %q0_3, %qreg[%i0] : memref<2x!mqtopt.Qubit>
        memref.store %q1_2, %qreg[%i1] : memref<2x!mqtopt.Qubit>
        memref.dealloc %qreg : memref<2x!mqtopt.Qubit>

        return
    }
}

// -----
// This test checks if a gphaseOp with no target no controlled qubit is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertGPhaseOp()
    func.func @testConvertGPhaseOp() {
        // CHECK: %[[c_0:.*]] = arith.constant 3.000000e-01 : f64
        // mqtref.gphase(%[[c_0]])

        %cst = arith.constant 3.000000e-01 : f64
        mqtopt.gphase(%cst)
        return
    }
}

// -----
// This test checks if a gphaseOp with a controlled qubit is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertGPhaseOpControlled()
    func.func @testConvertGPhaseOpControlled() {
        // CHECK: %[[c_0:.*]] = arith.constant 3.000000e-01 : f64
        // CHECK: mqtref.gphase(%[[c_0]]) ctrl %[[ANY:.*]]

        %i0 = arith.constant 0 : index
        %qreg = memref.alloc() : memref<1x!mqtopt.Qubit>
        %q0 = memref.load %qreg[%i0] : memref<1x!mqtopt.Qubit>

        %cst = arith.constant 3.000000e-01 : f64
        %q1 = mqtopt.gphase(%cst) ctrl %q0 : ctrl !mqtopt.Qubit

        memref.store %q1, %qreg[%i0] : memref<1x!mqtopt.Qubit>
        memref.dealloc %qreg : memref<1x!mqtopt.Qubit>

        return
    }
}

// -----
// This test checks if a gphaseOp with a positive controlled qubit and a negative controlled qubit is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertGPhaseOpPositiveNegativeControlled()
    func.func @testConvertGPhaseOpPositiveNegativeControlled() {
        // CHECK: %[[c_0:.*]] = arith.constant 3.000000e-01 : f64
        // CHECK: mqtref.gphase(%[[c_0]]) ctrl %[[ANY:.*]] nctrl %[[ANY:.*]]

        %i1 = arith.constant 1 : index
        %i0 = arith.constant 0 : index
        %qreg = memref.alloc() : memref<2x!mqtopt.Qubit>
        %q0_0 = memref.load %qreg[%i0] : memref<2x!mqtopt.Qubit>
        %q1_0 = memref.load %qreg[%i1] : memref<2x!mqtopt.Qubit>

        %cst = arith.constant 3.000000e-01 : f64
        %q0_1, %q1_1 = mqtopt.gphase(%cst) ctrl %q0_0 nctrl %q1_0 : ctrl !mqtopt.Qubit nctrl !mqtopt.Qubit

        memref.store %q0_1, %qreg[%i0] : memref<2x!mqtopt.Qubit>
        memref.store %q1_1, %qreg[%i1] : memref<2x!mqtopt.Qubit>
        memref.dealloc %qreg : memref<2x!mqtopt.Qubit>

        return
    }
}

// -----
// This test checks if a barrierOp is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertBarrierOp()
    func.func @testConvertBarrierOp() {
        // CHECK: mqtref.barrier() %[[ANY:.*]]

        %i0 = arith.constant 0 : index
        %qreg = memref.alloc() : memref<1x!mqtopt.Qubit>
        %q0 = memref.load %qreg[%i0] : memref<1x!mqtopt.Qubit>

        %q1 = mqtopt.barrier() %q0 : !mqtopt.Qubit

        memref.store %q1, %qreg[%i0] : memref<1x!mqtopt.Qubit>
        memref.dealloc %qreg : memref<1x!mqtopt.Qubit>

        return
    }
}

// -----
// This test checks if a barrierOp with multiple inputs is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertBarrierOpMultipleInputs()
    func.func @testConvertBarrierOpMultipleInputs() {
        // CHECK: mqtref.barrier() %[[ANY:.*]], %[[ANY:.*]]

        %i1 = arith.constant 1 : index
        %i0 = arith.constant 0 : index
        %qreg = memref.alloc() : memref<2x!mqtopt.Qubit>
        %q0_0 = memref.load %qreg[%i0] : memref<2x!mqtopt.Qubit>
        %q1_0 = memref.load %qreg[%i1] : memref<2x!mqtopt.Qubit>

        %q01_1:2 = mqtopt.barrier() %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit

        memref.store %q01_1#0, %qreg[%i0] : memref<2x!mqtopt.Qubit>
        memref.store %q01_1#1, %qreg[%i1] : memref<2x!mqtopt.Qubit>
        memref.dealloc %qreg : memref<2x!mqtopt.Qubit>

        return
    }
}

// -----
// This test checks if the QubitOp is converted correctly
module {
    // CHECK-LABEL: func.func @testConvertQubitOp
    func.func @testConvertQubitOp() {
        // CHECK: %[[Q0:.*]] = mqtref.qubit 0
        // CHECK-NOT: %[[ANY:.*]] = mqtopt.qubit 0

        %q0 = mqtopt.qubit 0
        return
    }
}

// -----
// This test checks if a Bell state is converted correctly for static qubits.
module {
    // CHECK-LABEL: func.func @bellStateStatic()
    func.func @bellStateStatic() {
        // CHECK: %[[q_0:.*]] = mqtref.qubit 0
        // CHECK: %[[q_1:.*]] = mqtref.qubit 1
        // CHECK: mqtref.h() %[[q_0]]
        // CHECK: mqtref.x() %[[q_1]] ctrl %[[q_0]]
        // CHECK: %[[m_0:.*]] = mqtref.measure %[[q_0]]
        // CHECK: %[[m_1:.*]] = mqtref.measure %[[q_1]]

        %q0 = mqtopt.qubit 0
        %q1 = mqtopt.qubit 1

        %q0_1 = mqtopt.h() %q0 : !mqtopt.Qubit
        %q1_1, %q0_2 = mqtopt.x() %q1 ctrl %q0_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
        %q0_3, %m0 = mqtopt.measure %q0_2
        %q1_2, %m1 = mqtopt.measure %q1_1

        return
    }
}

// -----
// This test checks if single-qubit allocation and deallocation are converted correctly
module {
    // CHECK-LABEL: func.func @testConvertAllocDeallocQubit()
    func.func @testConvertAllocDeallocQubit() {
        // CHECK: %[[q_0:.*]] = mqtref.allocQubit
        // CHECK: mqtref.deallocQubit %[[q_0]]
        %q0 = mqtopt.allocQubit
        mqtopt.deallocQubit %q0
        return
    }
}
