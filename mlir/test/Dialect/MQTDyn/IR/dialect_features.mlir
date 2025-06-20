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
        // CHECK: %[[QREQ_0:.*]] = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}>
        // CHECK-NOT: "mqtdyn.allocQubitRegister"

        %qreq_0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        return
    }
}

// -----
// This test checks if the AllocOp is parsed and handled correctly using a dynamic operand.
module {
    // CHECK-LABEL: func.func @testAllocOpOperand
    func.func @testAllocOpOperand() {
        // CHECK: %[[SIZE:.*]] = arith.constant 4
        // CHECK: %[[QREQ_0:.*]] = "mqtdyn.allocQubitRegister"(%[[SIZE]]) : (i64) -> !mqtdyn.QubitRegister
        // CHECK-NOT: "mqtdyn.allocQubitRegister"

        %size = arith.constant 4 : i64
        %qreg_0 = "mqtdyn.allocQubitRegister"(%size) : (i64) -> !mqtdyn.QubitRegister
        return
    }
}

// -----
// This test checks if the DeallocOp is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testDeallocOp
    func.func @testDeallocOp() {
        // CHECK: "mqtdyn.deallocQubitRegister"(%[[ANY:.*]])
        // CHECK-NOT: "mqtdyn.deallocQubitRegister"

        %qreg_0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        "mqtdyn.deallocQubitRegister"(%qreg_0) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if the ExtractOp is parsed and handled correctly using a static attribute.
module {
    // CHECK-LABEL: func.func @testExtractOpAttribute
    func.func @testExtractOpAttribute() {
        // CHECK: %[[Q_0:.*]] = "mqtdyn.extractQubit"(%[[ANY:.*]]) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        // CHECK-NOT: "mqtdyn.extractQubit"([[ANY:.*]])

        %qreg_0 = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
        %q_0 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        return
    }
}

// -----
// This test checks if the ExtractOp is parsed and handled correctly using a dynamic operand.
module {
    // CHECK-LABEL: func.func @testExtractOpOperand
    func.func @testExtractOpOperand() {
        // CHECK: %[[INDEX:.*]] = arith.constant 0
        // CHECK: %[[Q_0:.*]] = "mqtdyn.extractQubit"(%[[ANY:.*]], %[[INDEX]]) : (!mqtdyn.QubitRegister, i64) -> !mqtdyn.Qubit
        // CHECK-NOT: "mqtdyn.extractQubit"([[ANY:.*]])

        %index = arith.constant 0 : i64
        %qreg_0 = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
        %q_0 = "mqtdyn.extractQubit"(%qreg_0, %index) : (!mqtdyn.QubitRegister, i64) -> !mqtdyn.Qubit
        return
    }
}

// -----
// This test checks that all resources defined in the MQTDyn dialect are parsed and handled correctly using dynamic operands.
module {
    // CHECK-LABEL: func.func @testAllResourcesUsingOperands
    func.func @testAllResourcesUsingOperands() {
        // CHECK: %[[SIZE:.*]] = arith.constant 1 : i64
        // CHECK: %[[INDEX:.*]] = arith.constant 0 : i64
        // CHECK: %[[QREQ_0:.*]] = "mqtdyn.allocQubitRegister"(%[[SIZE]]) : (i64) -> !mqtdyn.QubitRegister
        // CHECK: %[[Q_0:.*]] = "mqtdyn.extractQubit"(%[[QREQ_0]], %[[INDEX]]) : (!mqtdyn.QubitRegister, i64) -> !mqtdyn.Qubit
        // CHECK: "mqtdyn.deallocQubitRegister"(%[[QREQ_0]])
        // CHECK-NOT: "mqtdyn.allocQubitRegister"
        // CHECK-NOT: "mqtdyn.extractQubit"
        // CHECK-NOT: "mqtdyn.deallocQubitRegister"

        %size = arith.constant 1 : i64
        %index = arith.constant 0 : i64
        %qreg_0 = "mqtdyn.allocQubitRegister"(%size) : (i64) -> !mqtdyn.QubitRegister
        %q_0 = "mqtdyn.extractQubit"(%qreg_0, %index) : (!mqtdyn.QubitRegister, i64) -> !mqtdyn.Qubit
        "mqtdyn.deallocQubitRegister"(%qreg_0) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if the MeasureOp applied to a single qubit is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testMeasureOp
    func.func @testMeasureOp() {
        // CHECK: [[M0_0:.*]] = "mqtdyn.measure"(%[[ANY:.*]])
        // CHECK-NOT: "mqtdyn.measure"([[ANY:.*]])

        %qreg_0 = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
        %q_0 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %m0_0 = "mqtdyn.measure"(%q_0) : (!mqtdyn.Qubit) -> i1
        "mqtdyn.deallocQubitRegister"(%qreg_0) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if the MeasureOp applied to multiple qubits is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testMeasureOpOnMultipleInputs
    func.func @testMeasureOpOnMultipleInputs() {
        // CHECK: [[M01_0:.*]]:2 = "mqtdyn.measure"(%[[ANY:.*]], %[[ANY:.*]])
        // CHECK-NOT: "mqtdyn.measure"([[ANY:.*]])

        %qreg_0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q_0 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q_1 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        %m01_0:2 = "mqtdyn.measure"(%q_0, %q_1) : (!mqtdyn.Qubit, !mqtdyn.Qubit) -> (i1, i1)

        "mqtdyn.deallocQubitRegister"(%qreg_0) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if single qubit gates are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testSingleQubitOp
    func.func @testSingleQubitOp() {
        // CHECK: mqtdyn.i() %[[Q_0:.*]]
        // CHECK: mqtdyn.h() %[[Q_0]]
        // CHECK: mqtdyn.x() %[[Q_0]]
        // CHECK: mqtdyn.y() %[[Q_0]]
        // CHECK: mqtdyn.z() %[[Q_0]]
        // CHECK: mqtdyn.s() %[[Q_0]]
        // CHECK: mqtdyn.sdg() %[[Q_0]]
        // CHECK: mqtdyn.t() %[[Q_0]]
        // CHECK: mqtdyn.tdg() %[[Q_0]]
        // CHECK: mqtdyn.v() %[[Q_0]]
        // CHECK: mqtdyn.vdg() %[[Q_0]]
        // CHECK: mqtdyn.sx() %[[Q_0]]
        // CHECK: mqtdyn.sxdg() %[[Q_0]]
        // CHECK-NOT: mqtdyn.[[ANY:.*]]() [[ANY:.*]]

        %qreg_0 = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
        %q_0 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.i() %q_0
        mqtdyn.h() %q_0
        mqtdyn.x() %q_0
        mqtdyn.y() %q_0
        mqtdyn.z() %q_0
        mqtdyn.s() %q_0
        mqtdyn.sdg() %q_0
        mqtdyn.t() %q_0
        mqtdyn.tdg() %q_0
        mqtdyn.v() %q_0
        mqtdyn.vdg() %q_0
        mqtdyn.sx() %q_0
        mqtdyn.sxdg() %q_0

        "mqtdyn.deallocQubitRegister"(%qreg_0) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if parameterized single qubit gates are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testSingleQubitRotationOp
    func.func @testSingleQubitRotationOp() {
        // CHECK: %[[P_0:.*]] = arith.constant 3.000000e-01
        // CHECK: mqtdyn.u(%[[P_0]], %[[P_0]], %[[P_0]]) %[[Q_0:.*]]
        // CHECK: mqtdyn.u2(%[[P_0]], %[[P_0]]) %[[Q_0]]
        // CHECK: mqtdyn.p(%[[P_0]]) %[[Q_0]]
        // CHECK: mqtdyn.rx(%[[P_0]]) %[[Q_0]]
        // CHECK: mqtdyn.ry(%[[P_0]]) %[[Q_0]]
        // CHECK: mqtdyn.rz(%[[P_0]]) %[[Q_0]]
        // CHECK-NOT: mqtdyn.[[ANY:.*]]([[ANY:.*]]) %[[ANY:.*]]

        %p_0 = arith.constant 3.000000e-01 : f64
        %qreg_0 = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
        %q_0 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.u(%p_0, %p_0, %p_0) %q_0
        mqtdyn.u2(%p_0, %p_0) %q_0
        mqtdyn.p(%p_0) %q_0
        mqtdyn.rx(%p_0) %q_0
        mqtdyn.ry(%p_0) %q_0
        mqtdyn.rz(%p_0) %q_0

        "mqtdyn.deallocQubitRegister"(%qreg_0) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if controlled parameterized single qubit gates are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testControlledSingleQubitRotationOp
    func.func @testControlledSingleQubitRotationOp() {
        // CHECK: %[[P_0:.*]] = arith.constant 3.000000e-01
        // CHECK: mqtdyn.u(%[[P_0]], %[[P_0]], %[[P_0]]) %[[Q_0:.*]] ctrl %[[Q_1:.*]]
        // CHECK: mqtdyn.u2(%[[P_0]], %[[P_0]]) %[[Q_0]] ctrl %[[Q_1]]
        // CHECK: mqtdyn.p(%[[P_0]]) %[[Q_0]] ctrl %[[Q_1]]
        // CHECK: mqtdyn.rx(%[[P_0]]) %[[Q_0]] ctrl %[[Q_1]]
        // CHECK: mqtdyn.ry(%[[P_0]]) %[[Q_0]] ctrl %[[Q_1]]
        // CHECK: mqtdyn.rz(%[[P_0]]) %[[Q_0]] ctrl %[[Q_1]]
        // CHECK-NOT: mqtdyn.[[ANY:.*]]([[ANY:.*]]) %[[ANY:.*]] ctrl %[[ANY:.*]]

        %p_0 = arith.constant 3.000000e-01 : f64
        %qreg_0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q_0 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q_1 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.u(%p_0, %p_0, %p_0) %q_0 ctrl %q_1
        mqtdyn.u2(%p_0, %p_0) %q_0 ctrl %q_1
        mqtdyn.p(%p_0) %q_0 ctrl %q_1
        mqtdyn.rx(%p_0) %q_0 ctrl %q_1
        mqtdyn.ry(%p_0) %q_0 ctrl %q_1
        mqtdyn.rz(%p_0) %q_0 ctrl %q_1

        "mqtdyn.deallocQubitRegister"(%qreg_0) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if an CX gate is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testCXOp
    func.func @testCXOp() {
        // CHECK: mqtdyn.x() %[[Q_0:.*]] ctrl %[[Q_1:.*]]
        // CHECK-NOT: %[[ANY:.*]], %[[ANY:.*]] = mqtdyn.x() %[[ANY:.*]] ctrl %[[ANY:.*]]

        %qreg_0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q_0 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q_1 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.x() %q_0 ctrl %q_1

        "mqtdyn.deallocQubitRegister"(%qreg_0) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a negative CX gate is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testNegativeCXOp
    func.func @testNegativeCXOp() {
        // CHECK: mqtdyn.x() %[[Q_1:.*]] nctrl %[[Q_0:.*]]
        // CHECK-NOT: %[[ANY:.*]], %[[ANY:.*]] = mqtdyn.x() %[[ANY:.*]] nctrl %[[ANY:.*]]

        %qreg_0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q_0 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q_1 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.x() %q_1 nctrl %q_0

        "mqtdyn.deallocQubitRegister"(%qreg_0) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if an MCX gate is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testMCXOp
    func.func @testMCXOp() {
        // CHECK: mqtdyn.x() %[[Q_1:.*]] ctrl %[[Q_0:.*]], %[[Q_2:.*]]
        // CHECK-NOT: mqtdyn.x() [[ANY:.*]] ctrl %[[ANY:.*]]:2

        %qreg_0 = "mqtdyn.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtdyn.QubitRegister
        %q_0 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q_1 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q_2 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 2 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        //===------------------------------------------------------------------===//
        // q_0: ──■── q_0
        //      ┌─┴─┐
        // q_1: ┤ X ├ q_1
        //      └─┬─┘
        // q_2: ──■── q_2
        //===----------------------------------------------------------------===//

        mqtdyn.x() %q_1 ctrl %q_0, %q_2

        "mqtdyn.deallocQubitRegister"(%qreg_0) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a negative MCX gate is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testNegativeMCXOp
    func.func @testNegativeMCXOp() {
        // CHECK: mqtdyn.x() %[[Q_1:.*]] nctrl %[[Q_0:.*]], %[[Q_2:.*]]
        // CHECK-NOT: mqtdyn.x() [[ANY:.*]] nctrl %[[ANY:.*]]:2

        %qreg_0 = "mqtdyn.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtdyn.QubitRegister
        %q_0 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q_1 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q_2 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 2 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        //===------------------------------------------------------------------===//
        // q_0: ──○── q_0
        //      ┌─┴─┐
        // q_1: ┤ X ├ q_1
        //      └─┬─┘
        // q_2: ──○── q_2
        //===----------------------------------------------------------------===//

        mqtdyn.x() %q_1 nctrl %q_0, %q_2

        "mqtdyn.deallocQubitRegister"(%qreg_0) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if an MCX gate is parsed and handled correctly using different types of controls.
module {
    // CHECK-LABEL: func.func @testMixedMCXOp
    func.func @testMixedMCXOp() {
        // CHECK: mqtdyn.x() %[[Q_1:.*]] ctrl %[[Q_0:.*]] nctrl %[[Q_2:.*]]
        // CHECK-NOT: mqtdyn.x() [[ANY:.*]] ctrl %[[ANY:.*]] nctrl %[[ANY:.*]]

        %qreg_0 = "mqtdyn.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtdyn.QubitRegister
        %q_0 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q_1 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q_2 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 2 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        //===------------------------------------------------------------------===//
        // q_0: ──■── q_0
        //      ┌─┴─┐
        // q_1: ┤ X ├ q_1
        //      └─┬─┘
        // q_2: ──○-─ q_2
        //===----------------------------------------------------------------===//

        mqtdyn.x() %q_1 ctrl %q_0 nctrl %q_2

       "mqtdyn.deallocQubitRegister"(%qreg_0) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if two target gates are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testTwoTargetOp
    func.func @testTwoTargetOp() {
        // CHECK: mqtdyn.swap() %[[Q_0:.*]], %[[Q_1:.*]]
        // CHECK: mqtdyn.iswap() %[[Q_0]], %[[Q_1]]
        // CHECK: mqtdyn.iswapdg() %[[Q_0]], %[[Q_1]]
        // CHECK: mqtdyn.peres() %[[Q_0]], %[[Q_1]]
        // CHECK: mqtdyn.peresdg() %[[Q_0]], %[[Q_1]]
        // CHECK: mqtdyn.dcx() %[[Q_0]], %[[Q_1]]
        // CHECK: mqtdyn.ecr() %[[Q_0]], %[[Q_1]]
        // CHECK-NOT: mqtdyn.[[ANY:.*]]() %[[ANY:.*]], %[[ANY:.*]]

        %qreg_0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q_0 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q_1 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.swap() %q_0, %q_1
        mqtdyn.iswap() %q_0, %q_1
        mqtdyn.iswapdg() %q_0, %q_1
        mqtdyn.peres() %q_0, %q_1
        mqtdyn.peresdg() %q_0, %q_1
        mqtdyn.dcx() %q_0, %q_1
        mqtdyn.ecr() %q_0, %q_1

        "mqtdyn.deallocQubitRegister"(%qreg_0) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a controlled SWAP gate is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testControlledSWAPOp
    func.func @testControlledSWAPOp() {
        // CHECK: mqtdyn.swap() %[[Q_0:.*]], %[[Q_1:.*]] ctrl %[[Q_2:.*]]
        // CHECK-NOT: mqtdyn.swap() [[ANY:.*]], [[ANY:.*]] ctrl [[ANY:.*]]

        %qreg_0 = "mqtdyn.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtdyn.QubitRegister
        %q_0 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q_1 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q_2 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 2 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.swap() %q_0, %q_1 ctrl %q_2


        "mqtdyn.deallocQubitRegister"(%qreg_0) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a negative controlled SWAP gate is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testNegativeControlledSWAPOp
    func.func @testNegativeControlledSWAPOp() {
        // CHECK: mqtdyn.swap() %[[Q_0:.*]], %[[Q_1:.*]] nctrl %[[Q_2:.*]]
        // CHECK-NOT: mqtdyn.swap() [[ANY:.*]], [[ANY:.*]] nctrl [[ANY:.*]]

        %qreg_0 = "mqtdyn.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtdyn.QubitRegister
        %q_0 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q_1 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q_2 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 2 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.swap() %q_0, %q_1 nctrl %q_2


        "mqtdyn.deallocQubitRegister"(%qreg_0) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a mixed controlled SWAP gate is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testMixedControlledSWAPOp
    func.func @testMixedControlledSWAPOp() {
        // CHECK: mqtdyn.swap() %[[Q_0:.*]], %[[Q_1:.*]] ctrl %[[Q_2:.*]] nctrl %[[Q_3:.*]]
        // CHECK-NOT: mqtdyn.swap() [[ANY:.*]], [[ANY:.*]] ctrl [[ANY:.*]] nctrl [[ANY:.*]]

        %qreg_0 = "mqtdyn.allocQubitRegister"() <{size_attr = 4 : i64}> : () -> !mqtdyn.QubitRegister
        %q_0 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q_1 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q_2 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 2 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q_3 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 3 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        //===------------------------------------------------------------------===//
        //       ┌──────┐
        // q_0:  ┤      ├ q_0
        //       │ SWAP │
        // q_1:  ┤      ├ q_1
        //       └───┬──┘
        // q_2:  ────■─── q_2
        //           │
        // q_3:  ────■─── q_3
        //===----------------------------------------------------------------===//

        mqtdyn.swap() %q_0, %q_1 ctrl %q_2 nctrl %q_3

        "mqtdyn.deallocQubitRegister"(%qreg_0) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}


// -----
// This test checks if parameterized multiple qubit gates are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testMultipleQubitRotationOp
    func.func @testMultipleQubitRotationOp() {
        // CHECK: %[[P_0:.*]] = arith.constant 3.000000e-01 : f64
        // CHECK: mqtdyn.rxx(%[[P_0]]) %[[Q_0:.*]], %[[Q_1:.*]]
        // CHECK: mqtdyn.ryy(%[[P_0]]) %[[Q_0]], %[[Q_1]]
        // CHECK: mqtdyn.rzz(%[[P_0]]) %[[Q_0]], %[[Q_1]]
        // CHECK: mqtdyn.rzx(%[[P_0]]) %[[Q_0]], %[[Q_1]]
        // CHECK: mqtdyn.xxminusyy(%[[P_0]], %[[P_0]]) %[[Q_0]], %[[Q_1]]
        // CHECK: mqtdyn.xxplusyy(%[[P_0]], %[[P_0]]) %[[Q_0]], %[[Q_1]]
        // CHECK-NOT: mqtdyn.[[ANY:.*]]([[ANY:.*]]) %[[ANY:.*]], %[[ANY:.*]]

        %p_0 = arith.constant 3.000000e-01 : f64
        %qreg_0 = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q_0 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q_1 = "mqtdyn.extractQubit"(%qreg_0) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.rxx(%p_0) %q_0, %q_1
        mqtdyn.ryy(%p_0) %q_0, %q_1
        mqtdyn.rzz(%p_0) %q_0, %q_1
        mqtdyn.rzx(%p_0) %q_0, %q_1
        mqtdyn.xxminusyy(%p_0, %p_0) %q_0, %q_1
        mqtdyn.xxplusyy(%p_0, %p_0) %q_0, %q_1

        "mqtdyn.deallocQubitRegister"(%qreg_0) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}
