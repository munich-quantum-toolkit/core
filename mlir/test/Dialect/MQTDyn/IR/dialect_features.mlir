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
        // CHECK: %[[QREG:.*]] = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}>
        // CHECK-NOT: "mqtdyn.allocQubitRegister"

        %qreg = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        return
    }
}

// -----
// This test checks if the AllocOp is parsed and handled correctly using a dynamic operand.
module {
    // CHECK-LABEL: func.func @testAllocOpOperand
    func.func @testAllocOpOperand() {
        // CHECK: %[[SIZE:.*]] = arith.constant 4
        // CHECK: %[[QREG:.*]] = "mqtdyn.allocQubitRegister"(%[[SIZE]]) : (i64) -> !mqtdyn.QubitRegister
        // CHECK-NOT: "mqtdyn.allocQubitRegister"

        %size = arith.constant 4 : i64
        %qreg = "mqtdyn.allocQubitRegister"(%size) : (i64) -> !mqtdyn.QubitRegister
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

        %qreg = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        "mqtdyn.deallocQubitRegister"(%qreg) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if the ExtractOp is parsed and handled correctly using a static attribute.
module {
    // CHECK-LABEL: func.func @testExtractOpAttribute
    func.func @testExtractOpAttribute() {
        // CHECK: %[[Q0:.*]] = "mqtdyn.extractQubit"(%[[ANY:.*]]) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        // CHECK-NOT: "mqtdyn.extractQubit"([[ANY:.*]])

        %qreg = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        return
    }
}

// -----
// This test checks if the ExtractOp is parsed and handled correctly using a dynamic operand.
module {
    // CHECK-LABEL: func.func @testExtractOpOperand
    func.func @testExtractOpOperand() {
        // CHECK: %[[INDEX:.*]] = arith.constant 0
        // CHECK: %[[Q0:.*]] = "mqtdyn.extractQubit"(%[[ANY:.*]], %[[INDEX]]) : (!mqtdyn.QubitRegister, i64) -> !mqtdyn.Qubit
        // CHECK-NOT: "mqtdyn.extractQubit"([[ANY:.*]])

        %index = arith.constant 0 : i64
        %qreg = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%qreg, %index) : (!mqtdyn.QubitRegister, i64) -> !mqtdyn.Qubit
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
        // CHECK: %[[QREG:.*]] = "mqtdyn.allocQubitRegister"(%[[SIZE]]) : (i64) -> !mqtdyn.QubitRegister
        // CHECK: %[[Q0:.*]] = "mqtdyn.extractQubit"(%[[QREG]], %[[INDEX]]) : (!mqtdyn.QubitRegister, i64) -> !mqtdyn.Qubit
        // CHECK: "mqtdyn.deallocQubitRegister"(%[[QREG]])
        // CHECK-NOT: "mqtdyn.allocQubitRegister"
        // CHECK-NOT: "mqtdyn.extractQubit"
        // CHECK-NOT: "mqtdyn.deallocQubitRegister"

        %size = arith.constant 1 : i64
        %index = arith.constant 0 : i64
        %qreg = "mqtdyn.allocQubitRegister"(%size) : (i64) -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%qreg, %index) : (!mqtdyn.QubitRegister, i64) -> !mqtdyn.Qubit
        "mqtdyn.deallocQubitRegister"(%qreg) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if the MeasureOp applied to a single qubit is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testMeasureOp
    func.func @testMeasureOp() {
        // CHECK: [[M0:.*]] = "mqtdyn.measure"(%[[ANY:.*]])
        // CHECK-NOT: "mqtdyn.measure"([[ANY:.*]])

        %qreg = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        %m0 = "mqtdyn.measure"(%q0) : (!mqtdyn.Qubit) -> i1

        "mqtdyn.deallocQubitRegister"(%qreg) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if the MeasureOp applied to multiple qubits is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testMeasureOpOnMultipleInputs
    func.func @testMeasureOpOnMultipleInputs() {
        // CHECK: [[M:.*]]:2 = "mqtdyn.measure"(%[[ANY:.*]], %[[ANY:.*]])
        // CHECK-NOT: "mqtdyn.measure"([[ANY:.*]])

        %qreg = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        %m:2 = "mqtdyn.measure"(%q0, %q1) : (!mqtdyn.Qubit, !mqtdyn.Qubit) -> (i1, i1)

        "mqtdyn.deallocQubitRegister"(%qreg) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if single qubit gates are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testSingleQubitOp
    func.func @testSingleQubitOp() {
        // CHECK: mqtdyn.i() %[[Q0:.*]]
        // CHECK: mqtdyn.h() %[[Q0]]
        // CHECK: mqtdyn.x() %[[Q0]]
        // CHECK: mqtdyn.y() %[[Q0]]
        // CHECK: mqtdyn.z() %[[Q0]]
        // CHECK: mqtdyn.s() %[[Q0]]
        // CHECK: mqtdyn.sdg() %[[Q0]]
        // CHECK: mqtdyn.t() %[[Q0]]
        // CHECK: mqtdyn.tdg() %[[Q0]]
        // CHECK: mqtdyn.v() %[[Q0]]
        // CHECK: mqtdyn.vdg() %[[Q0]]
        // CHECK: mqtdyn.sx() %[[Q0]]
        // CHECK: mqtdyn.sxdg() %[[Q0]]
        // CHECK-NOT: mqtdyn.[[ANY:.*]]() [[ANY:.*]]

        %qreg = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

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

        "mqtdyn.deallocQubitRegister"(%qreg) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if parameterized single qubit gates are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testSingleQubitRotationOp
    func.func @testSingleQubitRotationOp() {
        // CHECK: %[[P0:.*]] = arith.constant 3.000000e-01
        // CHECK: mqtdyn.u(%[[P0]], %[[P0]], %[[P0]]) %[[Q0:.*]]
        // CHECK: mqtdyn.u2(%[[P0]], %[[P0]]) %[[Q0]]
        // CHECK: mqtdyn.p(%[[P0]]) %[[Q0]]
        // CHECK: mqtdyn.rx(%[[P0]]) %[[Q0]]
        // CHECK: mqtdyn.ry(%[[P0]]) %[[Q0]]
        // CHECK: mqtdyn.rz(%[[P0]]) %[[Q0]]
        // CHECK-NOT: mqtdyn.[[ANY:.*]]([[ANY:.*]]) %[[ANY:.*]]

        %p0 = arith.constant 3.000000e-01 : f64
        %qreg = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.u(%p0, %p0, %p0) %q0
        mqtdyn.u2(%p0, %p0) %q0
        mqtdyn.p(%p0) %q0
        mqtdyn.rx(%p0) %q0
        mqtdyn.ry(%p0) %q0
        mqtdyn.rz(%p0) %q0

        "mqtdyn.deallocQubitRegister"(%qreg) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if controlled parameterized single qubit gates are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testControlledSingleQubitRotationOp
    func.func @testControlledSingleQubitRotationOp() {
        // CHECK: %[[P0:.*]] = arith.constant 3.000000e-01
        // CHECK: mqtdyn.u(%[[P0]], %[[P0]], %[[P0]]) %[[Q0:.*]] ctrl %[[Q1:.*]]
        // CHECK: mqtdyn.u2(%[[P0]], %[[P0]]) %[[Q0]] ctrl %[[Q1]]
        // CHECK: mqtdyn.p(%[[P0]]) %[[Q0]] ctrl %[[Q1]]
        // CHECK: mqtdyn.rx(%[[P0]]) %[[Q0]] ctrl %[[Q1]]
        // CHECK: mqtdyn.ry(%[[P0]]) %[[Q0]] ctrl %[[Q1]]
        // CHECK: mqtdyn.rz(%[[P0]]) %[[Q0]] ctrl %[[Q1]]
        // CHECK-NOT: mqtdyn.[[ANY:.*]]([[ANY:.*]]) %[[ANY:.*]] ctrl %[[ANY:.*]]

        %p0 = arith.constant 3.000000e-01 : f64
        %qreg = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.u(%p0, %p0, %p0) %q0 ctrl %q1
        mqtdyn.u2(%p0, %p0) %q0 ctrl %q1
        mqtdyn.p(%p0) %q0 ctrl %q1
        mqtdyn.rx(%p0) %q0 ctrl %q1
        mqtdyn.ry(%p0) %q0 ctrl %q1
        mqtdyn.rz(%p0) %q0 ctrl %q1

        "mqtdyn.deallocQubitRegister"(%qreg) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if an CX gate is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testCXOp
    func.func @testCXOp() {
        // CHECK: mqtdyn.x() %[[Q0:.*]] ctrl %[[Q1:.*]]
        // CHECK-NOT: %[[ANY:.*]], %[[ANY:.*]] = mqtdyn.x() %[[ANY:.*]] ctrl %[[ANY:.*]]

        %qreg = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.x() %q0 ctrl %q1

        "mqtdyn.deallocQubitRegister"(%qreg) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a negative CX gate is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testNegativeCXOp
    func.func @testNegativeCXOp() {
        // CHECK: mqtdyn.x() %[[Q1:.*]] nctrl %[[Q0:.*]]
        // CHECK-NOT: %[[ANY:.*]], %[[ANY:.*]] = mqtdyn.x() %[[ANY:.*]] nctrl %[[ANY:.*]]

        %qreg = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.x() %q1 nctrl %q0

        "mqtdyn.deallocQubitRegister"(%qreg) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if an MCX gate is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testMCXOp
    func.func @testMCXOp() {
        // CHECK: mqtdyn.x() %[[Q1:.*]] ctrl %[[Q0:.*]], %[[Q2:.*]]
        // CHECK-NOT: mqtdyn.x() [[ANY:.*]] ctrl %[[ANY:.*]]:2

        %qreg = "mqtdyn.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q2 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 2 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        //===------------------------------------------------------------------===//
        // q0: ──■── q0
        //     ┌─┴─┐
        // q1: ┤ X ├ q1
        //     └─┬─┘
        // q2: ──■── q2
        //===----------------------------------------------------------------===//

        mqtdyn.x() %q1 ctrl %q0, %q2

        "mqtdyn.deallocQubitRegister"(%qreg) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a negative MCX gate is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testNegativeMCXOp
    func.func @testNegativeMCXOp() {
        // CHECK: mqtdyn.x() %[[Q1:.*]] nctrl %[[Q0:.*]], %[[Q2:.*]]
        // CHECK-NOT: mqtdyn.x() [[ANY:.*]] nctrl %[[ANY:.*]]:2

        %qreg = "mqtdyn.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q2 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 2 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        //===------------------------------------------------------------------===//
        // q0: ──○── q0
        //     ┌─┴─┐
        // q1: ┤ X ├ q1
        //     └─┬─┘
        // q2: ──○── q2
        //===----------------------------------------------------------------===//

        mqtdyn.x() %q1 nctrl %q0, %q2

        "mqtdyn.deallocQubitRegister"(%qreg) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if an MCX gate is parsed and handled correctly using different types of controls.
module {
    // CHECK-LABEL: func.func @testMixedMCXOp
    func.func @testMixedMCXOp() {
        // CHECK: mqtdyn.x() %[[Q1:.*]] ctrl %[[Q0:.*]] nctrl %[[Q2:.*]]
        // CHECK-NOT: mqtdyn.x() [[ANY:.*]] ctrl %[[ANY:.*]] nctrl %[[ANY:.*]]

        %qreg = "mqtdyn.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q2 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 2 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        //===------------------------------------------------------------------===//
        // q0: ──■── q0
        //     ┌─┴─┐
        // q1: ┤ X ├ q1
        //     └─┬─┘
        // q2: ──○-─ q2
        //===----------------------------------------------------------------===//

        mqtdyn.x() %q1 ctrl %q0 nctrl %q2

       "mqtdyn.deallocQubitRegister"(%qreg) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if two target gates are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testTwoTargetOp
    func.func @testTwoTargetOp() {
        // CHECK: mqtdyn.swap() %[[Q0:.*]], %[[Q1:.*]]
        // CHECK: mqtdyn.iswap() %[[Q0]], %[[Q1]]
        // CHECK: mqtdyn.iswapdg() %[[Q0]], %[[Q1]]
        // CHECK: mqtdyn.peres() %[[Q0]], %[[Q1]]
        // CHECK: mqtdyn.peresdg() %[[Q0]], %[[Q1]]
        // CHECK: mqtdyn.dcx() %[[Q0]], %[[Q1]]
        // CHECK: mqtdyn.ecr() %[[Q0]], %[[Q1]]
        // CHECK-NOT: mqtdyn.[[ANY:.*]]() %[[ANY:.*]], %[[ANY:.*]]

        %qreg = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.swap() %q0, %q1
        mqtdyn.iswap() %q0, %q1
        mqtdyn.iswapdg() %q0, %q1
        mqtdyn.peres() %q0, %q1
        mqtdyn.peresdg() %q0, %q1
        mqtdyn.dcx() %q0, %q1
        mqtdyn.ecr() %q0, %q1

        "mqtdyn.deallocQubitRegister"(%qreg) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a controlled SWAP gate is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testControlledSWAPOp
    func.func @testControlledSWAPOp() {
        // CHECK: mqtdyn.swap() %[[Q0:.*]], %[[Q1:.*]] ctrl %[[Q2:.*]]
        // CHECK-NOT: mqtdyn.swap() [[ANY:.*]], [[ANY:.*]] ctrl [[ANY:.*]]

        %qreg = "mqtdyn.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q2 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 2 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.swap() %q0, %q1 ctrl %q2


        "mqtdyn.deallocQubitRegister"(%qreg) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a negative controlled SWAP gate is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testNegativeControlledSWAPOp
    func.func @testNegativeControlledSWAPOp() {
        // CHECK: mqtdyn.swap() %[[Q0:.*]], %[[Q1:.*]] nctrl %[[Q2:.*]]
        // CHECK-NOT: mqtdyn.swap() [[ANY:.*]], [[ANY:.*]] nctrl [[ANY:.*]]

        %qreg = "mqtdyn.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q2 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 2 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.swap() %q0, %q1 nctrl %q2


        "mqtdyn.deallocQubitRegister"(%qreg) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a mixed controlled SWAP gate is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testMixedControlledSWAPOp
    func.func @testMixedControlledSWAPOp() {
        // CHECK: mqtdyn.swap() %[[Q0:.*]], %[[Q1:.*]] ctrl %[[Q2:.*]] nctrl %[[Q3:.*]]
        // CHECK-NOT: mqtdyn.swap() [[ANY:.*]], [[ANY:.*]] ctrl [[ANY:.*]] nctrl [[ANY:.*]]

        %qreg = "mqtdyn.allocQubitRegister"() <{size_attr = 4 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q2 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 2 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q3 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 3 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        //===------------------------------------------------------------------===//
        //      ┌──────┐
        // q0:  ┤      ├ q0
        //      │ SWAP │
        // q1:  ┤      ├ q1
        //      └───┬──┘
        // q2:  ────■─── q2
        //          │
        // q3:  ────■─── q3
        //===----------------------------------------------------------------===//

        mqtdyn.swap() %q0, %q1 ctrl %q2 nctrl %q3

        "mqtdyn.deallocQubitRegister"(%qreg) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}


// -----
// This test checks if parameterized multiple qubit gates are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testMultipleQubitRotationOp
    func.func @testMultipleQubitRotationOp() {
        // CHECK: %[[P0:.*]] = arith.constant 3.000000e-01 : f64
        // CHECK: mqtdyn.rxx(%[[P0]]) %[[Q0:.*]], %[[Q1:.*]]
        // CHECK: mqtdyn.ryy(%[[P0]]) %[[Q0]], %[[Q1]]
        // CHECK: mqtdyn.rzz(%[[P0]]) %[[Q0]], %[[Q1]]
        // CHECK: mqtdyn.rzx(%[[P0]]) %[[Q0]], %[[Q1]]
        // CHECK: mqtdyn.xxminusyy(%[[P0]], %[[P0]]) %[[Q0]], %[[Q1]]
        // CHECK: mqtdyn.xxplusyy(%[[P0]], %[[P0]]) %[[Q0]], %[[Q1]]
        // CHECK-NOT: mqtdyn.[[ANY:.*]]([[ANY:.*]]) %[[ANY:.*]], %[[ANY:.*]]

        %p0 = arith.constant 3.000000e-01 : f64
        %qreg = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.rxx(%p0) %q0, %q1
        mqtdyn.ryy(%p0) %q0, %q1
        mqtdyn.rzz(%p0) %q0, %q1
        mqtdyn.rzx(%p0) %q0, %q1
        mqtdyn.xxminusyy(%p0, %p0) %q0, %q1
        mqtdyn.xxplusyy(%p0, %p0) %q0, %q1

        "mqtdyn.deallocQubitRegister"(%qreg) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if parameterized multiple qubit gates are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testControlledMultipleQubitRotationOp
    func.func @testControlledMultipleQubitRotationOp() {
        // CHECK: %[[P0:.*]] = arith.constant 3.000000e-01 : f64
        // CHECK: mqtdyn.rxx(%[[P0]]) %[[Q0:.*]], %[[Q1:.*]] ctrl %[[Q2:.*]]
        // CHECK: mqtdyn.ryy(%[[P0]]) %[[Q0]], %[[Q1]] ctrl %[[Q2]]
        // CHECK: mqtdyn.rzz(%[[P0]]) %[[Q0]], %[[Q1]] ctrl %[[Q2]]
        // CHECK: mqtdyn.rzx(%[[P0]]) %[[Q0]], %[[Q1]] ctrl %[[Q2]]
        // CHECK: mqtdyn.xxminusyy(%[[P0]], %[[P0]]) %[[Q0]], %[[Q1]] ctrl %[[Q2]]
        // CHECK: mqtdyn.xxplusyy(%[[P0]], %[[P0]]) %[[Q0]], %[[Q1]] ctrl %[[Q2]]
        // CHECK-NOT: mqtdyn.[[ANY:.*]]([[ANY:.*]]) %[[ANY:.*]], %[[ANY:.*]] ctrl %[[ANY:.*]]

        %p0 = arith.constant 3.000000e-01 : f64
        %qreg = "mqtdyn.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q2 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 2 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.rxx(%p0) %q0, %q1 ctrl %q2
        mqtdyn.ryy(%p0) %q0, %q1 ctrl %q2
        mqtdyn.rzz(%p0) %q0, %q1 ctrl %q2
        mqtdyn.rzx(%p0) %q0, %q1 ctrl %q2
        mqtdyn.xxminusyy(%p0, %p0) %q0, %q1 ctrl %q2
        mqtdyn.xxplusyy(%p0, %p0) %q0, %q1 ctrl %q2

        "mqtdyn.deallocQubitRegister"(%qreg) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}

// -----
// This test expects an error to be thrown when an alloc op does not define size operands nor attributes.
module {
    func.func @testAllocMissingSize() {
        // expected-error-re@+1 {{'mqtdyn.allocQubitRegister' op exactly one attribute ({{.*}}) or operand ({{.*}}) must be provided for 'size'}}
        %qreg = "mqtdyn.allocQubitRegister"() : () -> !mqtdyn.QubitRegister
    }
}

// -----
// This test expects an error to be thrown when an alloc op defines size operands and attributes.
module {
    func.func @testAllocOperandAndAttribute() {
        %size = arith.constant 3 : i64

        // expected-error-re@+1 {{'mqtdyn.allocQubitRegister' op exactly one attribute ({{.*}}) or operand ({{.*}}) must be provided for 'size'}}
        %qreg = "mqtdyn.allocQubitRegister"(%size) <{size_attr = 3 : i64}> : (i64) -> !mqtdyn.QubitRegister
    }
}

// -----
// This test expects an error to be thrown when an extract op defines index operands and attributes.
module {
    func.func @testExtractOperandAndAttribute() {
        %index = arith.constant 0 : i64
        %qreg = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister

        // expected-error-re@+1 {{'mqtdyn.extractQubit' op exactly one attribute ({{.*}}) or operand ({{.*}}) must be provided for 'index'}}
        %q = "mqtdyn.extractQubit"(%qreg, %index) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister, i64) -> !mqtdyn.Qubit
  }
}

// -----
// This test expects an error to be thrown when an extract op does not define index operands nor attributes.
module {
    func.func @testExtractMissingIndex() {
        %qreg = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
        %index = arith.constant 0 : i64

        // expected-error-re@+1 {{'mqtdyn.extractQubit' op exactly one attribute ({{.*}}) or operand ({{.*}}) must be provided for 'index'}}
        %q = "mqtdyn.extractQubit"(%qreg) : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
  }
}

// -----
// This test expects an error to be thrown when parsing a parameterised operation.
module {
    func.func @testParamOpInvalidFormat() {
        %p0 = arith.constant 3.000000e-01 : f64
        %qreg = "mqtdyn.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        // expected-error@+1 {{operation expects exactly 3 parameters but got 2}}
        mqtdyn.u(%p0, %p0) %q0
    }
}

// -----
// This test checks if a measurement op with a mismatch between in-qubits and out-bits throws an error as expected.
module {
    func.func @testMeasureMismatchInOutBits() {
        %qreg = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%qreg)  <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        // expected-error@+1 {{'mqtdyn.measure' op number of input qubits (2) and output bits (0) must be the same}}
        "mqtdyn.measure"(%q0, %q1) : (!mqtdyn.Qubit, !mqtdyn.Qubit) -> ()
    }
}

// -----
// This test checks if a Bell state is parsed and handled correctly by using many instructions tested above.
module {
    // CHECK-LABEL: func.func @bellState()
    func.func @bellState() {
        // CHECK: %[[QREG:.*]] = "mqtdyn.allocQubitRegister"
        // CHECK: %[[Q0:.*]] = "mqtdyn.extractQubit"(%[[QREG]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Q1:.*]] = "mqtdyn.extractQubit"(%[[QREG]]) <{index_attr = 1 : i64}>
        // CHECK: mqtdyn.h() %[[Q0]]
        // CHECK: mqtdyn.x() %[[Q1]] ctrl %[[Q0]]
        // CHECK: %[[M0:.*]] = "mqtdyn.measure"(%[[Q0]]) : (!mqtdyn.Qubit) -> i1
        // CHECK: %[[M1:.*]] = "mqtdyn.measure"(%[[Q1]]) : (!mqtdyn.Qubit) -> i1
        // CHECK: "mqtdyn.deallocQubitRegister"(%[[QREG]]) : (!mqtdyn.QubitRegister) -> ()

        %qreg = "mqtdyn.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit
        %q1 = "mqtdyn.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        mqtdyn.h() %q0
        mqtdyn.x() %q1 ctrl %q0
        %m0 = "mqtdyn.measure"(%q0) : (!mqtdyn.Qubit) -> i1
        %m1 = "mqtdyn.measure"(%q1) : (!mqtdyn.Qubit) -> i1

        "mqtdyn.deallocQubitRegister"(%qreg) : (!mqtdyn.QubitRegister) -> ()
        return
    }
}
