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
        // CHECK: %[[QREG:.*]] = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}>

        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
        return
    }
}

// -----
// This test checks if the AllocOp is parsed and handled correctly using a dynamic operand.
module {
    // CHECK-LABEL: func.func @testAllocOpOperand
    func.func @testAllocOpOperand() {
        // CHECK: %[[SIZE:.*]] = arith.constant 4
        // CHECK: %[[QREG:.*]] = "mqtref.allocQubitRegister"(%[[SIZE]]) : (i64) -> !mqtref.QubitRegister

        %size = arith.constant 4 : i64
        %qreg = "mqtref.allocQubitRegister"(%size) : (i64) -> !mqtref.QubitRegister
        return
    }
}

// -----
// This test checks if the AllocQubitOp is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testAllocQubitOp
    func.func @testAllocQubitOp() {
        // CHECK: %[[Q0:.*]] = "mqtref.allocQubit"() : () -> !mqtref.Qubit

        %q0 = "mqtref.allocQubit"() : () -> !mqtref.Qubit
        return
    }
}

// -----
// This test checks if the DeallocQubitOp is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testDeallocQubitOp
    func.func @testDeallocQubitOp() {
        // CHECK: %[[Q0:.*]] = "mqtref.allocQubit"() : () -> !mqtref.Qubit
        // CHECK: "mqtref.deallocQubit"(%[[Q0]]) : (!mqtref.Qubit) -> ()

        %q0 = "mqtref.allocQubit"() : () -> !mqtref.Qubit
        "mqtref.deallocQubit"(%q0) : (!mqtref.Qubit) -> ()
        return
    }
}

// -----
// This test checks if the DeallocOp is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testDeallocOp
    func.func @testDeallocOp() {
        // CHECK: "mqtref.deallocQubitRegister"(%[[ANY:.*]])

        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
        "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if the ExtractOp is parsed and handled correctly using a static attribute.
module {
    // CHECK-LABEL: func.func @testExtractOpAttribute
    func.func @testExtractOpAttribute() {
        // CHECK: %[[Q0:.*]] = "mqtref.extractQubit"(%[[ANY:.*]]) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit

        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        return
    }
}

// -----
// This test checks if the ExtractOp is parsed and handled correctly using a dynamic operand.
module {
    // CHECK-LABEL: func.func @testExtractOpOperand
    func.func @testExtractOpOperand() {
        // CHECK: %[[INDEX:.*]] = arith.constant 0
        // CHECK: %[[Q0:.*]] = "mqtref.extractQubit"(%[[ANY:.*]], %[[INDEX]]) : (!mqtref.QubitRegister, i64) -> !mqtref.Qubit

        %index = arith.constant 0 : i64
        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%qreg, %index) : (!mqtref.QubitRegister, i64) -> !mqtref.Qubit
        return
    }
}

// -----
// This test checks that all resources defined in the MQTRef dialect are parsed and handled correctly using dynamic operands.
module {
    // CHECK-LABEL: func.func @testAllResourcesUsingOperands
    func.func @testAllResourcesUsingOperands() {
        // CHECK: %[[SIZE:.*]] = arith.constant 1 : i64
        // CHECK: %[[INDEX:.*]] = arith.constant 0 : i64
        // CHECK: %[[QREG:.*]] = "mqtref.allocQubitRegister"(%[[SIZE]]) : (i64) -> !mqtref.QubitRegister
        // CHECK: %[[Q0:.*]] = "mqtref.extractQubit"(%[[QREG]], %[[INDEX]]) : (!mqtref.QubitRegister, i64) -> !mqtref.Qubit
        // CHECK: "mqtref.deallocQubitRegister"(%[[QREG]])

        %size = arith.constant 1 : i64
        %index = arith.constant 0 : i64
        %qreg = "mqtref.allocQubitRegister"(%size) : (i64) -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%qreg, %index) : (!mqtref.QubitRegister, i64) -> !mqtref.Qubit
        "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if the MeasureOp is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testMeasureOp
    func.func @testMeasureOp() {
        // CHECK: [[M0:.*]] = mqtref.measure %[[ANY:.*]]

        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit

        %m0 = mqtref.measure %q0

        "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if the MeasureOp on a static qubit is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testMeasureOpStatic
    func.func @testMeasureOpStatic() {
        // CHECK: [[M0:.*]] = mqtref.measure %[[ANY:.*]]

        %q0 = mqtref.qubit 0
        %m0 = mqtref.measure %q0
        return
    }
}

// -----
// This test checks if the MeasureOp applied to a static qubit is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testMeasureOpOnStaticInput
    func.func @testMeasureOpOnStaticInput() {
        // CHECK: %[[M0:.*]] = mqtref.measure %[[ANY:.*]]

        %q0 = mqtref.qubit 0
        %m0 = mqtref.measure %q0
        return
    }
}

// -----
// This test checks if the ResetOp on a dynamic qubit is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testResetOp
    func.func @testResetOp() {
        // CHECK: "mqtref.reset"(%[[ANY:.*]])

        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit

        "mqtref.reset"(%q0) : (!mqtref.Qubit) -> ()

        "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if the ResetOp on a static qubit is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testResetOpOnStaticInput
    func.func @testResetOpOnStaticInput() {
        // CHECK: "mqtref.reset"(%[[ANY:.*]]) : (!mqtref.Qubit) -> ()

        %q0 = mqtref.qubit 0
        "mqtref.reset"(%q0) : (!mqtref.Qubit) -> ()
        return
    }
}

// -----
// This test checks if no-target operations without controls are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testNoTargetNoControls
    func.func @testNoTargetNoControls() {
        // CHECK: %[[C0_F64:.*]] = arith.constant 3.000000e-01
        // CHECK: mqtref.gphase(%[[C0_F64]])

        %c0_f64 = arith.constant 3.000000e-01 : f64
        mqtref.gphase(%c0_f64)
        return
    }
}

// -----
// This test checks if no-target operations with controls are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testNoTargetWithControls
    func.func @testNoTargetWithControls() {
        // CHECK: %[[C0_F64:.*]] = arith.constant 3.000000e-01
        // CHECK: mqtref.gphase(%[[C0_F64]]) ctrl %[[Q0:.*]]
        // CHECK: mqtref.gphase(%[[C0_F64]]) ctrl %[[Q0]], %[[ANY:.*]]

        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q1 = "mqtref.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit

        %c0_f64 = arith.constant 3.000000e-01 : f64
        mqtref.gphase(%c0_f64) ctrl %q0
        mqtref.gphase(%c0_f64) ctrl %q0, %q1

        "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if no-target operations with positive and negative controls are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testNoTargetPositiveNegativeControls
    func.func @testNoTargetPositiveNegativeControls() {
        // CHECK: %[[C0_F64:.*]] = arith.constant 3.000000e-01
        // CHECK: mqtref.gphase(%[[C0_F64]]) ctrl %[[ANY:.*]]

        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
        %q0_0 = "mqtref.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q1_0 = "mqtref.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit

        %c0_f64 = arith.constant 3.000000e-01 : f64
        mqtref.gphase(%c0_f64) ctrl %q0_0 nctrl %q1_0

        "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if no-target operations with static controls are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testNoTargetWithStaticControls
    func.func @testNoTargetWithStaticControls() {
        // CHECK: %[[C0_F64:.*]] = arith.constant 3.000000e-01
        // CHECK: mqtref.gphase(%[[C0_F64]]) ctrl %[[Q0:.*]]
        // CHECK: mqtref.gphase(%[[C0_F64]]) ctrl %[[Q0]], %[[ANY:.*]]

        %q0 = mqtref.qubit 0
        %q1 = mqtref.qubit 1

        %c0_f64 = arith.constant 3.000000e-01 : f64
        mqtref.gphase(%c0_f64) ctrl %q0
        mqtref.gphase(%c0_f64) ctrl %q0, %q1

        return
    }
}

// -----
// This test checks if no-target operations with positive and negative static controls are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testNoTargetPositiveNegativeStaticControls
    func.func @testNoTargetPositiveNegativeStaticControls() {
        // CHECK: %[[C0_F64:.*]] = arith.constant 3.000000e-01
        // CHECK: mqtref.gphase(%[[C0_F64]]) ctrl %[[ANY:.*]]

        %q0 = mqtref.qubit 0
        %q1 = mqtref.qubit 1

        %c0_f64 = arith.constant 3.000000e-01 : f64
        mqtref.gphase(%c0_f64) ctrl %q0 nctrl %q1

        return
    }
}

// -----
// This test checks if single qubit gates on dynamic qubits are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testSingleQubitOp
    func.func @testSingleQubitOp() {
        // CHECK: mqtref.i() %[[Q0:.*]]
        // CHECK: mqtref.h() %[[Q0]]
        // CHECK: mqtref.x() %[[Q0]]
        // CHECK: mqtref.y() %[[Q0]]
        // CHECK: mqtref.z() %[[Q0]]
        // CHECK: mqtref.s() %[[Q0]]
        // CHECK: mqtref.sdg() %[[Q0]]
        // CHECK: mqtref.t() %[[Q0]]
        // CHECK: mqtref.tdg() %[[Q0]]
        // CHECK: mqtref.v() %[[Q0]]
        // CHECK: mqtref.vdg() %[[Q0]]
        // CHECK: mqtref.sx() %[[Q0]]
        // CHECK: mqtref.sxdg() %[[Q0]]

        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit

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

        "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if single qubit gates on static qubits are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testSingleQubitOpStatic
    func.func @testSingleQubitOpStatic() {
        // CHECK: mqtref.i() %[[Q0:.*]]
        // CHECK: mqtref.h() %[[Q0]]
        // CHECK: mqtref.x() %[[Q0]]
        // CHECK: mqtref.y() %[[Q0]]
        // CHECK: mqtref.z() %[[Q0]]
        // CHECK: mqtref.s() %[[Q0]]
        // CHECK: mqtref.sdg() %[[Q0]]
        // CHECK: mqtref.t() %[[Q0]]
        // CHECK: mqtref.tdg() %[[Q0]]
        // CHECK: mqtref.v() %[[Q0]]
        // CHECK: mqtref.vdg() %[[Q0]]
        // CHECK: mqtref.sx() %[[Q0]]
        // CHECK: mqtref.sxdg() %[[Q0]]

        %q0 = mqtref.qubit 0

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

        return
    }
}

// -----
// This test checks if parameterized single qubit gates on dynamic qubits are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testSingleQubitRotationOp
    func.func @testSingleQubitRotationOp() {
        // CHECK: %[[P0:.*]] = arith.constant 3.000000e-01
        // CHECK: mqtref.u(%[[P0]], %[[P0]], %[[P0]]) %[[Q0:.*]]
        // CHECK: mqtref.u2(%[[P0]], %[[P0]] static [] mask [false, false]) %[[Q0]]
        // CHECK: mqtref.p(%[[P0]]) %[[Q0]]
        // CHECK: mqtref.rx(%[[P0]]) %[[Q0]]
        // CHECK: mqtref.ry(%[[P0]]) %[[Q0]]
        // CHECK: mqtref.rz(%[[P0]]) %[[Q0]]

        %p0 = arith.constant 3.000000e-01 : f64
        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit

        mqtref.u(%p0, %p0, %p0) %q0
        mqtref.u2(%p0, %p0 static [] mask [false, false]) %q0
        mqtref.p(%p0) %q0
        mqtref.rx(%p0) %q0
        mqtref.ry(%p0) %q0
        mqtref.rz(%p0) %q0

        "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if parameterized single qubit gates on static qubits are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testSingleQubitRotationOpStatic
    func.func @testSingleQubitRotationOpStatic() {
        // CHECK: %[[P0:.*]] = arith.constant 3.000000e-01
        // CHECK: mqtref.u(%[[P0]], %[[P0]], %[[P0]]) %[[Q0:.*]]
        // CHECK: mqtref.u2(%[[P0]], %[[P0]] static [] mask [false, false]) %[[Q0]]
        // CHECK: mqtref.p(%[[P0]]) %[[Q0]]
        // CHECK: mqtref.rx(%[[P0]]) %[[Q0]]
        // CHECK: mqtref.ry(%[[P0]]) %[[Q0]]
        // CHECK: mqtref.rz(%[[P0]]) %[[Q0]]

        %p0 = arith.constant 3.000000e-01 : f64
        %q0 = mqtref.qubit 0

        mqtref.u(%p0, %p0, %p0) %q0
        mqtref.u2(%p0, %p0 static [] mask [false, false]) %q0
        mqtref.p(%p0) %q0
        mqtref.rx(%p0) %q0
        mqtref.ry(%p0) %q0
        mqtref.rz(%p0) %q0

        return
    }
}

// -----
// This test checks if controlled parameterized single qubit gates on dynamic qubits are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testControlledSingleQubitRotationOp
    func.func @testControlledSingleQubitRotationOp() {
        // CHECK: %[[P0:.*]] = arith.constant 3.000000e-01
        // CHECK: mqtref.u(%[[P0]], %[[P0]], %[[P0]]) %[[Q0:.*]] ctrl %[[Q1:.*]]
        // CHECK: mqtref.u2(%[[P0]], %[[P0]]) %[[Q0]] ctrl %[[Q1]]
        // CHECK: mqtref.p(%[[P0]]) %[[Q0]] ctrl %[[Q1]]
        // CHECK: mqtref.rx(%[[P0]]) %[[Q0]] ctrl %[[Q1]]
        // CHECK: mqtref.ry(%[[P0]]) %[[Q0]] ctrl %[[Q1]]
        // CHECK: mqtref.rz(%[[P0]]) %[[Q0]] ctrl %[[Q1]]

        %p0 = arith.constant 3.000000e-01 : f64
        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q1 = "mqtref.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit

        mqtref.u(%p0, %p0, %p0) %q0 ctrl %q1
        mqtref.u2(%p0, %p0) %q0 ctrl %q1
        mqtref.p(%p0) %q0 ctrl %q1
        mqtref.rx(%p0) %q0 ctrl %q1
        mqtref.ry(%p0) %q0 ctrl %q1
        mqtref.rz(%p0) %q0 ctrl %q1

        "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if controlled parameterized single qubit gates on static qubits are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testControlledSingleQubitRotationOpStatic
    func.func @testControlledSingleQubitRotationOpStatic() {
        // CHECK: %[[P0:.*]] = arith.constant 3.000000e-01
        // CHECK: mqtref.u(%[[P0]], %[[P0]], %[[P0]]) %[[Q0:.*]] ctrl %[[Q1:.*]]
        // CHECK: mqtref.u2(%[[P0]], %[[P0]]) %[[Q0]] ctrl %[[Q1]]
        // CHECK: mqtref.p(%[[P0]]) %[[Q0]] ctrl %[[Q1]]
        // CHECK: mqtref.rx(%[[P0]]) %[[Q0]] ctrl %[[Q1]]
        // CHECK: mqtref.ry(%[[P0]]) %[[Q0]] ctrl %[[Q1]]
        // CHECK: mqtref.rz(%[[P0]]) %[[Q0]] ctrl %[[Q1]]

        %p0 = arith.constant 3.000000e-01 : f64
        %q0 = mqtref.qubit 0
        %q1 = mqtref.qubit 1

        mqtref.u(%p0, %p0, %p0) %q0 ctrl %q1
        mqtref.u2(%p0, %p0) %q0 ctrl %q1
        mqtref.p(%p0) %q0 ctrl %q1
        mqtref.rx(%p0) %q0 ctrl %q1
        mqtref.ry(%p0) %q0 ctrl %q1
        mqtref.rz(%p0) %q0 ctrl %q1

        return
    }
}

// -----
// This test checks if an CX gate on dynamic qubits is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testCXOp
    func.func @testCXOp() {
        // CHECK: mqtref.x() %[[Q0:.*]] ctrl %[[Q1:.*]]

        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q1 = "mqtref.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit

        mqtref.x() %q0 ctrl %q1

        "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a negative CX gate on dynamic qubits is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testNegativeCXOp
    func.func @testNegativeCXOp() {
        // CHECK: mqtref.x() %[[Q1:.*]] nctrl %[[Q0:.*]]

        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q1 = "mqtref.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit

        mqtref.x() %q1 nctrl %q0

        "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if an CX gate on static qubits is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testCXOpStatic
    func.func @testCXOpStatic() {
        // CHECK: mqtref.x() %[[Q0:.*]] ctrl %[[Q1:.*]]

        %q0 = mqtref.qubit 0
        %q1 = mqtref.qubit 1

        mqtref.x() %q0 ctrl %q1

        return
    }
}

// -----
// This test checks if a negative CX gate on static qubits is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testNegativeCXOpStatic
    func.func @testNegativeCXOpStatic() {
        // CHECK: mqtref.x() %[[Q1:.*]] nctrl %[[Q0:.*]]

        %q0 = mqtref.qubit 0
        %q1 = mqtref.qubit 1

        mqtref.x() %q1 nctrl %q0

        return
    }
}

// -----
// This test checks if an MCX gate on dynamic qubits is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testMCXOp
    func.func @testMCXOp() {
        // CHECK: mqtref.x() %[[Q1:.*]] ctrl %[[Q0:.*]], %[[Q2:.*]]

        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q1 = "mqtref.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q2 = "mqtref.extractQubit"(%qreg) <{index_attr = 2 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit

        //===------------------------------------------------------------------===//
        // q0: ──■── q0
        //     ┌─┴─┐
        // q1: ┤ X ├ q1
        //     └─┬─┘
        // q2: ──■── q2
        //===----------------------------------------------------------------===//

        mqtref.x() %q1 ctrl %q0, %q2

        "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a negative MCX gate on dynamic qubits is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testNegativeMCXOp
    func.func @testNegativeMCXOp() {
        // CHECK: mqtref.x() %[[Q1:.*]] nctrl %[[Q0:.*]], %[[Q2:.*]]

        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q1 = "mqtref.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q2 = "mqtref.extractQubit"(%qreg) <{index_attr = 2 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit

        //===------------------------------------------------------------------===//
        // q0: ──○── q0
        //     ┌─┴─┐
        // q1: ┤ X ├ q1
        //     └─┬─┘
        // q2: ──○── q2
        //===----------------------------------------------------------------===//

        mqtref.x() %q1 nctrl %q0, %q2

        "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if an MCX gate on static qubits is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testMCXOpStatic
    func.func @testMCXOpStatic() {
        // CHECK: mqtref.x() %[[Q1:.*]] ctrl %[[Q0:.*]], %[[Q2:.*]]

        %q0 = mqtref.qubit 0
        %q1 = mqtref.qubit 1
        %q2 = mqtref.qubit 2

        //===------------------------------------------------------------------===//
        // q0: ──■── q0
        //     ┌─┴─┐
        // q1: ┤ X ├ q1
        //     └─┬─┘
        // q2: ──■── q2
        //===----------------------------------------------------------------===//

        mqtref.x() %q1 ctrl %q0, %q2

        return
    }
}

// -----
// This test checks if a negative MCX gate on static qubits is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testNegativeMCXOpStatic
    func.func @testNegativeMCXOpStatic() {
        // CHECK: mqtref.x() %[[Q1:.*]] nctrl %[[Q0:.*]], %[[Q2:.*]]

        %q0 = mqtref.qubit 0
        %q1 = mqtref.qubit 1
        %q2 = mqtref.qubit 2

        //===------------------------------------------------------------------===//
        // q0: ──○── q0
        //     ┌─┴─┐
        // q1: ┤ X ├ q1
        //     └─┬─┘
        // q2: ──○── q2
        //===----------------------------------------------------------------===//

        mqtref.x() %q1 nctrl %q0, %q2

        return
    }
}

// -----
// This test checks if an MCX gate on dynamic qubits is parsed and handled correctly using different types of controls.
module {
    // CHECK-LABEL: func.func @testMixedMCXOp
    func.func @testMixedMCXOp() {
        // CHECK: mqtref.x() %[[Q1:.*]] ctrl %[[Q0:.*]] nctrl %[[Q2:.*]]

        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q1 = "mqtref.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q2 = "mqtref.extractQubit"(%qreg) <{index_attr = 2 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit

        //===------------------------------------------------------------------===//
        // q0: ──■── q0
        //     ┌─┴─┐
        // q1: ┤ X ├ q1
        //     └─┬─┘
        // q2: ──○-─ q2
        //===----------------------------------------------------------------===//

        mqtref.x() %q1 ctrl %q0 nctrl %q2

       "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()
        return
    }
}


// -----
// This test checks if an MCX gate on static qubits is parsed and handled correctly using different types of controls.
module {
    // CHECK-LABEL: func.func @testMixedMCXOpStatic
    func.func @testMixedMCXOpStatic() {
        // CHECK: mqtref.x() %[[Q1:.*]] ctrl %[[Q0:.*]] nctrl %[[Q2:.*]]

        %q0 = mqtref.qubit 0
        %q1 = mqtref.qubit 1
        %q2 = mqtref.qubit 2

        //===------------------------------------------------------------------===//
        // q0: ──■── q0
        //     ┌─┴─┐
        // q1: ┤ X ├ q1
        //     └─┬─┘
        // q2: ──○-─ q2
        //===----------------------------------------------------------------===//

        mqtref.x() %q1 ctrl %q0 nctrl %q2

        return
    }
}

// -----
// This test checks if two target gates on dynamic qubits are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testTwoTargetOp
    func.func @testTwoTargetOp() {
        // CHECK: mqtref.swap() %[[Q0:.*]], %[[Q1:.*]]
        // CHECK: mqtref.iswap() %[[Q0]], %[[Q1]]
        // CHECK: mqtref.iswapdg() %[[Q0]], %[[Q1]]
        // CHECK: mqtref.peres() %[[Q0]], %[[Q1]]
        // CHECK: mqtref.peresdg() %[[Q0]], %[[Q1]]
        // CHECK: mqtref.dcx() %[[Q0]], %[[Q1]]
        // CHECK: mqtref.ecr() %[[Q0]], %[[Q1]]

        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q1 = "mqtref.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit

        mqtref.swap() %q0, %q1
        mqtref.iswap() %q0, %q1
        mqtref.iswapdg() %q0, %q1
        mqtref.peres() %q0, %q1
        mqtref.peresdg() %q0, %q1
        mqtref.dcx() %q0, %q1
        mqtref.ecr() %q0, %q1

        "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if two target gates on static qubits are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testTwoTargetOpStatic
    func.func @testTwoTargetOpStatic() {
        // CHECK: mqtref.swap() %[[Q0:.*]], %[[Q1:.*]]
        // CHECK: mqtref.iswap() %[[Q0]], %[[Q1]]
        // CHECK: mqtref.iswapdg() %[[Q0]], %[[Q1]]
        // CHECK: mqtref.peres() %[[Q0]], %[[Q1]]
        // CHECK: mqtref.peresdg() %[[Q0]], %[[Q1]]
        // CHECK: mqtref.dcx() %[[Q0]], %[[Q1]]
        // CHECK: mqtref.ecr() %[[Q0]], %[[Q1]]

        %q0 = mqtref.qubit 0
        %q1 = mqtref.qubit 1

        mqtref.swap() %q0, %q1
        mqtref.iswap() %q0, %q1
        mqtref.iswapdg() %q0, %q1
        mqtref.peres() %q0, %q1
        mqtref.peresdg() %q0, %q1
        mqtref.dcx() %q0, %q1
        mqtref.ecr() %q0, %q1

        return
    }
}

// -----
// This test checks if a controlled SWAP gate on dynamic qubits is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testControlledSWAPOp
    func.func @testControlledSWAPOp() {
        // CHECK: mqtref.swap() %[[Q0:.*]], %[[Q1:.*]] ctrl %[[Q2:.*]]

        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q1 = "mqtref.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q2 = "mqtref.extractQubit"(%qreg) <{index_attr = 2 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit

        mqtref.swap() %q0, %q1 ctrl %q2

        "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a negative controlled SWAP gate on dynamic qubits is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testNegativeControlledSWAPOp
    func.func @testNegativeControlledSWAPOp() {
        // CHECK: mqtref.swap() %[[Q0:.*]], %[[Q1:.*]] nctrl %[[Q2:.*]]

        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q1 = "mqtref.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q2 = "mqtref.extractQubit"(%qreg) <{index_attr = 2 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit

        mqtref.swap() %q0, %q1 nctrl %q2

        "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if a controlled SWAP gate on static qubits is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testControlledSWAPOpStatic
    func.func @testControlledSWAPOpStatic() {
        // CHECK: mqtref.swap() %[[Q0:.*]], %[[Q1:.*]] ctrl %[[Q2:.*]]

        %q0 = mqtref.qubit 0
        %q1 = mqtref.qubit 1
        %q2 = mqtref.qubit 2

        mqtref.swap() %q0, %q1 ctrl %q2

        return
    }
}

// -----
// This test checks if a negative controlled SWAP gate on static qubits is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testNegativeControlledSWAPOpStatic
    func.func @testNegativeControlledSWAPOpStatic() {
        // CHECK: mqtref.swap() %[[Q0:.*]], %[[Q1:.*]] nctrl %[[Q2:.*]]

        %q0 = mqtref.qubit 0
        %q1 = mqtref.qubit 1
        %q2 = mqtref.qubit 2

        mqtref.swap() %q0, %q1 nctrl %q2

        return
    }
}

// -----
// This test checks if a mixed controlled SWAP gate on dynamic qubits is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testMixedControlledSWAPOp
    func.func @testMixedControlledSWAPOp() {
        // CHECK: mqtref.swap() %[[Q0:.*]], %[[Q1:.*]] ctrl %[[Q2:.*]] nctrl %[[Q3:.*]]

        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 4 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q1 = "mqtref.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q2 = "mqtref.extractQubit"(%qreg) <{index_attr = 2 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q3 = "mqtref.extractQubit"(%qreg) <{index_attr = 3 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit

        //===------------------------------------------------------------------===//
        //      ┌──────┐
        // q0:  ┤      ├ q0
        //      │ SWAP │
        // q1:  ┤      ├ q1
        //      └───┬──┘
        // q2:  ────■─── q2
        //          │
        // q3:  ────○─── q3
        //===----------------------------------------------------------------===//

        mqtref.swap() %q0, %q1 ctrl %q2 nctrl %q3

        "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()

        return
    }
}

// -----
// This test checks if a mixed controlled SWAP gate on static qubits is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testMixedControlledSWAPOpStatic
    func.func @testMixedControlledSWAPOpStatic() {
        // CHECK: mqtref.swap() %[[Q0:.*]], %[[Q1:.*]] ctrl %[[Q2:.*]] nctrl %[[Q3:.*]]

        %q0 = mqtref.qubit 0
        %q1 = mqtref.qubit 1
        %q2 = mqtref.qubit 2
        %q3 = mqtref.qubit 3

        //===------------------------------------------------------------------===//
        //      ┌──────┐
        // q0:  ┤      ├ q0
        //      │ SWAP │
        // q1:  ┤      ├ q1
        //      └───┬──┘
        // q2:  ────■─── q2
        //          │
        // q3:  ────○─── q3
        //===----------------------------------------------------------------===//

        mqtref.swap() %q0, %q1 ctrl %q2 nctrl %q3

        return
    }
}


// -----
// This test checks if parameterized multiple qubit gates on dynamic qubits are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testMultipleQubitRotationOp
    func.func @testMultipleQubitRotationOp() {
        // CHECK: %[[P0:.*]] = arith.constant 3.000000e-01 : f64
        // CHECK: mqtref.rxx(%[[P0]]) %[[Q0:.*]], %[[Q1:.*]]
        // CHECK: mqtref.ryy(%[[P0]]) %[[Q0]], %[[Q1]]
        // CHECK: mqtref.rzz(%[[P0]]) %[[Q0]], %[[Q1]]
        // CHECK: mqtref.rzx(%[[P0]]) %[[Q0]], %[[Q1]]
        // CHECK: mqtref.xxminusyy(%[[P0]], %[[P0]]) %[[Q0]], %[[Q1]]
        // CHECK: mqtref.xxplusyy(%[[P0]], %[[P0]]) %[[Q0]], %[[Q1]]

        %p0 = arith.constant 3.000000e-01 : f64
        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q1 = "mqtref.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit

        mqtref.rxx(%p0) %q0, %q1
        mqtref.ryy(%p0) %q0, %q1
        mqtref.rzz(%p0) %q0, %q1
        mqtref.rzx(%p0) %q0, %q1
        mqtref.xxminusyy(%p0, %p0) %q0, %q1
        mqtref.xxplusyy(%p0, %p0) %q0, %q1

        "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if parameterized multiple qubit gates on static qubits are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testMultipleQubitRotationOpStatic
    func.func @testMultipleQubitRotationOpStatic() {
        // CHECK: %[[P0:.*]] = arith.constant 3.000000e-01 : f64
        // CHECK: mqtref.rxx(%[[P0]]) %[[Q0:.*]], %[[Q1:.*]]
        // CHECK: mqtref.ryy(%[[P0]]) %[[Q0]], %[[Q1]]
        // CHECK: mqtref.rzz(%[[P0]]) %[[Q0]], %[[Q1]]
        // CHECK: mqtref.rzx(%[[P0]]) %[[Q0]], %[[Q1]]
        // CHECK: mqtref.xxminusyy(%[[P0]], %[[P0]]) %[[Q0]], %[[Q1]]
        // CHECK: mqtref.xxplusyy(%[[P0]], %[[P0]]) %[[Q0]], %[[Q1]]

        %p0 = arith.constant 3.000000e-01 : f64
        %q0 = mqtref.qubit 0
        %q1 = mqtref.qubit 1

        mqtref.rxx(%p0) %q0, %q1
        mqtref.ryy(%p0) %q0, %q1
        mqtref.rzz(%p0) %q0, %q1
        mqtref.rzx(%p0) %q0, %q1
        mqtref.xxminusyy(%p0, %p0) %q0, %q1
        mqtref.xxplusyy(%p0, %p0) %q0, %q1

        return
    }
}

// -----
// This test checks if parameterized multiple qubit gates on dynamic qubits are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testControlledMultipleQubitRotationOp
    func.func @testControlledMultipleQubitRotationOp() {
        // CHECK: %[[P0:.*]] = arith.constant 3.000000e-01 : f64
        // CHECK: mqtref.rxx(%[[P0]]) %[[Q0:.*]], %[[Q1:.*]] ctrl %[[Q2:.*]]
        // CHECK: mqtref.ryy(%[[P0]]) %[[Q0]], %[[Q1]] ctrl %[[Q2]]
        // CHECK: mqtref.rzz(%[[P0]]) %[[Q0]], %[[Q1]] ctrl %[[Q2]]
        // CHECK: mqtref.rzx(%[[P0]]) %[[Q0]], %[[Q1]] ctrl %[[Q2]]
        // CHECK: mqtref.xxminusyy(%[[P0]], %[[P0]]) %[[Q0]], %[[Q1]] ctrl %[[Q2]]
        // CHECK: mqtref.xxplusyy(%[[P0]], %[[P0]]) %[[Q0]], %[[Q1]] ctrl %[[Q2]]

        %p0 = arith.constant 3.000000e-01 : f64
        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q1 = "mqtref.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q2 = "mqtref.extractQubit"(%qreg) <{index_attr = 2 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit

        mqtref.rxx(%p0) %q0, %q1 ctrl %q2
        mqtref.ryy(%p0) %q0, %q1 ctrl %q2
        mqtref.rzz(%p0) %q0, %q1 ctrl %q2
        mqtref.rzx(%p0) %q0, %q1 ctrl %q2
        mqtref.xxminusyy(%p0, %p0) %q0, %q1 ctrl %q2
        mqtref.xxplusyy(%p0, %p0) %q0, %q1 ctrl %q2

        "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()
        return
    }
}

// -----
// This test checks if parameterized multiple qubit gates on static qubits are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testControlledMultipleQubitRotationOpStatic
    func.func @testControlledMultipleQubitRotationOpStatic() {
        // CHECK: %[[P0:.*]] = arith.constant 3.000000e-01 : f64
        // CHECK: mqtref.rxx(%[[P0]]) %[[Q0:.*]], %[[Q1:.*]] ctrl %[[Q2:.*]]
        // CHECK: mqtref.ryy(%[[P0]]) %[[Q0]], %[[Q1]] ctrl %[[Q2]]
        // CHECK: mqtref.rzz(%[[P0]]) %[[Q0]], %[[Q1]] ctrl %[[Q2]]
        // CHECK: mqtref.rzx(%[[P0]]) %[[Q0]], %[[Q1]] ctrl %[[Q2]]
        // CHECK: mqtref.xxminusyy(%[[P0]], %[[P0]]) %[[Q0]], %[[Q1]] ctrl %[[Q2]]
        // CHECK: mqtref.xxplusyy(%[[P0]], %[[P0]]) %[[Q0]], %[[Q1]] ctrl %[[Q2]]

        %p0 = arith.constant 3.000000e-01 : f64
        %q0 = mqtref.qubit 0
        %q1 = mqtref.qubit 1
        %q2 = mqtref.qubit 2

        mqtref.rxx(%p0) %q0, %q1 ctrl %q2
        mqtref.ryy(%p0) %q0, %q1 ctrl %q2
        mqtref.rzz(%p0) %q0, %q1 ctrl %q2
        mqtref.rzx(%p0) %q0, %q1 ctrl %q2
        mqtref.xxminusyy(%p0, %p0) %q0, %q1 ctrl %q2
        mqtref.xxplusyy(%p0, %p0) %q0, %q1 ctrl %q2

        return
    }
}

// -----
// This test expects an error to be thrown when an alloc op does not define size operands nor attributes.
module {
    func.func @testAllocMissingSize() {
        // expected-error-re@+1 {{'mqtref.allocQubitRegister' op exactly one attribute ({{.*}}) or operand ({{.*}}) must be provided for 'size'}}
        %qreg = "mqtref.allocQubitRegister"() : () -> !mqtref.QubitRegister

        return
    }
}

// -----
// This test expects an error to be thrown when an alloc op defines size operands and attributes.
module {
    func.func @testAllocOperandAndAttribute() {
        %size = arith.constant 3 : i64

        // expected-error-re@+1 {{'mqtref.allocQubitRegister' op exactly one attribute ({{.*}}) or operand ({{.*}}) must be provided for 'size'}}
        %qreg = "mqtref.allocQubitRegister"(%size) <{size_attr = 3 : i64}> : (i64) -> !mqtref.QubitRegister

        return
    }
}

// -----
// This test expects an error to be thrown when an extract op defines index operands and attributes.
module {
    func.func @testExtractOperandAndAttribute() {
        %index = arith.constant 0 : i64
        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtref.QubitRegister

        // expected-error-re@+1 {{'mqtref.extractQubit' op exactly one attribute ({{.*}}) or operand ({{.*}}) must be provided for 'index'}}
        %q = "mqtref.extractQubit"(%qreg, %index) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister, i64) -> !mqtref.Qubit

        return
  }
}

// -----
// This test expects an error to be thrown when an extract op does not define index operands nor attributes.
module {
    func.func @testExtractMissingIndex() {
        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtref.QubitRegister
        %index = arith.constant 0 : i64

        // expected-error-re@+1 {{'mqtref.extractQubit' op exactly one attribute ({{.*}}) or operand ({{.*}}) must be provided for 'index'}}
        %q = "mqtref.extractQubit"(%qreg) : (!mqtref.QubitRegister) -> !mqtref.Qubit

        return
  }
}

// -----
// This test expects an error to be thrown when parsing a parameterised operation.
module {
    func.func @testParamOpInvalidFormat() {
        %p0 = arith.constant 3.000000e-01 : f64
        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit

        // expected-error@+1 {{operation expects exactly 3 parameters but got 2}}
        mqtref.u(%p0, %p0) %q0

        return
    }
}

// -----
// This test checks if a no-target arity constraint operation detects correctly when a target is provided.
module {
    func.func @testNoTargetContainsTarget() {
        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtref.QubitRegister
        %q_0 = "mqtref.extractQubit"(%qreg)  <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> (!mqtref.Qubit)

        %c0_f64 = arith.constant 3.000000e-01 : f64
        // expected-error@+1 {{number of input qubits (1) must be 0}}
        mqtref.gphase(%c0_f64) %q_0

        "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()

        return
    }
}

// -----
// This test checks if static parameters for rotation operations are parsed correctly.
module {
    // CHECK-LABEL: func.func @testStaticParameters
    func.func @testStaticParameters() {
        // CHECK: mqtref.u( static [1.000000e-01, 2.000000e-01, 3.000000e-01]) %[[ANY:.*]]
        // CHECK: mqtref.u( static [1.000000e-01, 2.000000e-01, 3.000000e-01] mask [true, true, true]) %[[ANY:.*]]

        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtref.QubitRegister
        %q_0 = "mqtref.extractQubit"(%qreg)  <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> (!mqtref.Qubit)

        mqtref.u(static [1.00000e-01, 2.00000e-01, 3.00000e-01]) %q_0
        mqtref.u(static [1.00000e-01, 2.00000e-01, 3.00000e-01] mask [true, true, true]) %q_0

        "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()

        return
    }
}

// -----
// This test checks if static parameters together with dynamic parameters for rotation operations are parsed correctly.
module {
    // CHECK-LABEL: func.func @testStaticAndDynamicParameters
    func.func @testStaticAndDynamicParameters() {
        // CHECK: mqtref.u(%[[ANY:.*]] static [1.000000e-01, 2.000000e-01] mask [true, false, true]) %[[ANY:.*]]

        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtref.QubitRegister
        %q_0 = "mqtref.extractQubit"(%qreg)  <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> (!mqtref.Qubit)

        %c0_f64 = arith.constant 3.000000e-01 : f64
        mqtref.u(%c0_f64 static [1.00000e-01, 2.00000e-01] mask [true, false, true]) %q_0

        "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()

        return
    }
}

// -----
// This test checks if static parameters and dynamic parameters surpassing the limit of parameters together is detected correctly.
module {
    func.func @testTooManyStaticAndDynamicParameters() {
        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtref.QubitRegister
        %q_0 = "mqtref.extractQubit"(%qreg)  <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> (!mqtref.Qubit)

        %c0_f64 = arith.constant 3.000000e-01 : f64
        // expected-error@+1 {{operation expects exactly 3 parameters but got 4}}
        mqtref.u(%c0_f64, %c0_f64 static [1.00000e-01, 2.00000e-01] mask [true, false, true]) %q_0

        "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()

        return
    }
}

// -----
// This test checks if static parameters and dynamic parameters being passed without a mask is detected correctly.
module {
    func.func @testStaticAndDynamicParametersNoMask() {
        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtref.QubitRegister
        %q_0 = "mqtref.extractQubit"(%qreg)  <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> (!mqtref.Qubit)

        %c0_f64 = arith.constant 3.000000e-01 : f64
        // expected-error@+1 {{operation has mixed dynamic and static parameters but no parameter mask}}
        mqtref.u(%c0_f64 static [1.00000e-01, 2.00000e-01]) %q_0

        "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()

        return
    }
}

// -----
// This test checks if a static parameter mask with incorrect size is detected correctly.
module {
    func.func @testStaticAndDynamicParametersWrongSizeMask() {
        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtref.QubitRegister
        %q_0 = "mqtref.extractQubit"(%qreg)  <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> (!mqtref.Qubit)

        %c0_f64 = arith.constant 3.000000e-01 : f64
        // expected-error@+1 {{operation expects exactly 3 parameters but has a parameter mask with 2 entries}}
        mqtref.u(%c0_f64 static [1.00000e-01, 2.00000e-01] mask [true, true]) %q_0

        "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()

        return
    }
}

// -----
// This test checks if a static parameter mask with an incorrect number of true entries is detected correctly.
module {
    func.func @testStaticAndDynamicParametersIncorrectTrueEntriesInMask() {
        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtref.QubitRegister
        %q_0 = "mqtref.extractQubit"(%qreg)  <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> (!mqtref.Qubit)

        %c0_f64 = arith.constant 3.000000e-01 : f64
        // expected-error@+1 {{operation has 2 static parameter(s) but has a parameter mask with 3 true entries}}
        mqtref.u(%c0_f64 static [1.00000e-01, 2.00000e-01] mask [true, true, true]) %q_0

        "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()

        return
    }
}

// -----
// This test checks if a static parameter mask with `true` parameters even though the operation has no static parameters is detected correctly.
module {
    func.func @testParametersMaskWithTrueEntriesButNoStaticParameters() {
        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 1 : i64}> : () -> !mqtref.QubitRegister
        %q_0 = "mqtref.extractQubit"(%qreg)  <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> (!mqtref.Qubit)

        %c0_f64 = arith.constant 3.000000e-01 : f64
        // expected-error@+1 {{operation has no static parameter but has a parameter mask with 1 true entries}}
        mqtref.u(%c0_f64, %c0_f64, %c0_f64 static [] mask [true, false, false]) %q_0

        "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()

        return
    }
}

// -----
// This test checks if a no-control gate being passed a control is detected correctly.
module {
    func.func @testNoControlWithControl() {
        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
        %q0_0 = "mqtref.extractQubit"(%qreg)  <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> (!mqtref.Qubit)
        %q1_0 = "mqtref.extractQubit"(%qreg)  <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> (!mqtref.Qubit)

        // expected-error@+1 {{'mqtref.barrier' op Gate marked as NoControl should not have control qubits}}
        mqtref.barrier() %q0_0 ctrl %q1_0

        "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()

        return
    }
}

// -----
// This test checks if a Bell state is parsed and handled correctly by using many instructions tested above.
module {
    // CHECK-LABEL: func.func @bellState()
    func.func @bellState() {
        // CHECK: %[[QREG:.*]] = "mqtref.allocQubitRegister"
        // CHECK: %[[Q0:.*]] = "mqtref.extractQubit"(%[[QREG]]) <{index_attr = 0 : i64}>
        // CHECK: %[[Q1:.*]] = "mqtref.extractQubit"(%[[QREG]]) <{index_attr = 1 : i64}>
        // CHECK: mqtref.h() %[[Q0]]
        // CHECK: mqtref.x() %[[Q1]] ctrl %[[Q0]]
        // CHECK: %[[M0:.*]] = mqtref.measure %[[Q0]]
        // CHECK: %[[M1:.*]] = mqtref.measure %[[Q1]]
        // CHECK: "mqtref.deallocQubitRegister"(%[[QREG]]) : (!mqtref.QubitRegister) -> ()

        %qreg = "mqtref.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtref.QubitRegister
        %q0 = "mqtref.extractQubit"(%qreg) <{index_attr = 0 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit
        %q1 = "mqtref.extractQubit"(%qreg) <{index_attr = 1 : i64}> : (!mqtref.QubitRegister) -> !mqtref.Qubit

        mqtref.h() %q0
        mqtref.x() %q1 ctrl %q0
        %m0 = mqtref.measure %q0
        %m1 = mqtref.measure %q1

        "mqtref.deallocQubitRegister"(%qreg) : (!mqtref.QubitRegister) -> ()

        return
    }
}
