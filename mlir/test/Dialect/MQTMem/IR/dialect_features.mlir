// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// -----
// This test checks if the QubitOp is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testAllocOpAttribute
    func.func @testAllocOpAttribute() {
        // CHECK: %[[Q0:.*]] = mqtmem.qubit 0
        // CHECK: %[[Q1:.*]] = mqtmem.qubit 1

        %q0 = "mqtmem.qubit"() <{index = 0 : i64}> : () -> !mqtmem.DeviceQubit
        %q1 = mqtmem.qubit 1

        return
    }
}

// -----
// This test checks if the MeasureOp applied to a single qubit is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testMeasureOp
    func.func @testMeasureOp() {
        // CHECK: [[M0:.*]] = "mqtmem.measure"(%[[ANY:.*]])

        %q0 = mqtmem.qubit 0
        %m0 = "mqtmem.measure"(%q0) : (!mqtmem.DeviceQubit) -> i1

        return
    }
}

// -----
// This test checks if the MeasureOp applied to multiple qubits is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testMeasureOpOnMultipleInputs
    func.func @testMeasureOpOnMultipleInputs() {
        // CHECK: [[M:.*]]:2 = "mqtmem.measure"(%[[ANY:.*]], %[[ANY:.*]])

        %q0 = mqtmem.qubit 0
        %q1 = mqtmem.qubit 1
        %m:2 = "mqtmem.measure"(%q0, %q1) : (!mqtmem.DeviceQubit, !mqtmem.DeviceQubit) -> (i1, i1)

        return
    }
}

// -----
// This test checks if no-target operations without controls are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testNoTargetNoControls
    func.func @testNoTargetNoControls() {
        // CHECK: %[[C0_F64:.*]] = arith.constant 3.000000e-01
        // CHECK: mqtmem.gphase(%[[C0_F64]])

        %c0_f64 = arith.constant 3.000000e-01 : f64
        mqtmem.gphase(%c0_f64)
        return
    }
}

// -----
// This test checks if no-target operations with controls are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testNoTargetWithControls
    func.func @testNoTargetWithControls() {
        // CHECK: %[[C0_F64:.*]] = arith.constant 3.000000e-01
        // CHECK: mqtmem.gphase(%[[C0_F64]]) ctrl %[[Q0:.*]]

        %q0 = mqtmem.qubit 0
        %c0_f64 = arith.constant 3.000000e-01 : f64
        mqtmem.gphase(%c0_f64) ctrl %q0

        return
    }
}

// -----
// This test checks if single qubit gates are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testSingleQubitOp
    func.func @testSingleQubitOp() {
        // CHECK: mqtmem.i() %[[Q0:.*]]
        // CHECK: mqtmem.h() %[[Q0]]
        // CHECK: mqtmem.x() %[[Q0]]
        // CHECK: mqtmem.y() %[[Q0]]
        // CHECK: mqtmem.z() %[[Q0]]
        // CHECK: mqtmem.s() %[[Q0]]
        // CHECK: mqtmem.sdg() %[[Q0]]
        // CHECK: mqtmem.t() %[[Q0]]
        // CHECK: mqtmem.tdg() %[[Q0]]
        // CHECK: mqtmem.v() %[[Q0]]
        // CHECK: mqtmem.vdg() %[[Q0]]
        // CHECK: mqtmem.sx() %[[Q0]]
        // CHECK: mqtmem.sxdg() %[[Q0]]

        %q0 = mqtmem.qubit 0

        mqtmem.i() %q0
        mqtmem.h() %q0
        mqtmem.x() %q0
        mqtmem.y() %q0
        mqtmem.z() %q0
        mqtmem.s() %q0
        mqtmem.sdg() %q0
        mqtmem.t() %q0
        mqtmem.tdg() %q0
        mqtmem.v() %q0
        mqtmem.vdg() %q0
        mqtmem.sx() %q0
        mqtmem.sxdg() %q0

        return
    }
}

// -----
// This test checks if parameterized single qubit gates are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testSingleQubitRotationOp
    func.func @testSingleQubitRotationOp() {
        // CHECK: %[[P0:.*]] = arith.constant 3.000000e-01
        // CHECK: mqtmem.u(%[[P0]], %[[P0]], %[[P0]]) %[[Q0:.*]]
        // CHECK: mqtmem.u2(%[[P0]], %[[P0]] static [] mask [false, false]) %[[Q0]]
        // CHECK: mqtmem.p(%[[P0]]) %[[Q0]]
        // CHECK: mqtmem.rx(%[[P0]]) %[[Q0]]
        // CHECK: mqtmem.ry(%[[P0]]) %[[Q0]]
        // CHECK: mqtmem.rz(%[[P0]]) %[[Q0]]

        %p0 = arith.constant 3.000000e-01 : f64
        %q0 = mqtmem.qubit 0

        mqtmem.u(%p0, %p0, %p0) %q0
        mqtmem.u2(%p0, %p0 static [] mask [false, false]) %q0
        mqtmem.p(%p0) %q0
        mqtmem.rx(%p0) %q0
        mqtmem.ry(%p0) %q0
        mqtmem.rz(%p0) %q0

        return
    }
}

// -----
// This test checks if controlled parameterized single qubit gates are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testControlledSingleQubitRotationOp
    func.func @testControlledSingleQubitRotationOp() {
        // CHECK: %[[P0:.*]] = arith.constant 3.000000e-01
        // CHECK: mqtmem.u(%[[P0]], %[[P0]], %[[P0]]) %[[Q0:.*]] ctrl %[[Q1:.*]]
        // CHECK: mqtmem.u2(%[[P0]], %[[P0]]) %[[Q0]] ctrl %[[Q1]]
        // CHECK: mqtmem.p(%[[P0]]) %[[Q0]] ctrl %[[Q1]]
        // CHECK: mqtmem.rx(%[[P0]]) %[[Q0]] ctrl %[[Q1]]
        // CHECK: mqtmem.ry(%[[P0]]) %[[Q0]] ctrl %[[Q1]]
        // CHECK: mqtmem.rz(%[[P0]]) %[[Q0]] ctrl %[[Q1]]

        %p0 = arith.constant 3.000000e-01 : f64
        %q0 = mqtmem.qubit 0
        %q1 = mqtmem.qubit 1

        mqtmem.u(%p0, %p0, %p0) %q0 ctrl %q1
        mqtmem.u2(%p0, %p0) %q0 ctrl %q1
        mqtmem.p(%p0) %q0 ctrl %q1
        mqtmem.rx(%p0) %q0 ctrl %q1
        mqtmem.ry(%p0) %q0 ctrl %q1
        mqtmem.rz(%p0) %q0 ctrl %q1

        return
    }
}

// -----
// This test checks if an CX gate is parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testCXOp
    func.func @testCXOp() {
        // CHECK: mqtmem.x() %[[Q0:.*]] ctrl %[[Q1:.*]]

        %q0 = mqtmem.qubit 0
        %q1 = mqtmem.qubit 1

        mqtmem.x() %q0 ctrl %q1

        return
    }
}

// -----
// This test checks if two target gates are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testTwoTargetOp
    func.func @testTwoTargetOp() {
        // CHECK: mqtmem.swap() %[[Q0:.*]], %[[Q1:.*]]
        // CHECK: mqtmem.iswap() %[[Q0]], %[[Q1]]
        // CHECK: mqtmem.iswapdg() %[[Q0]], %[[Q1]]
        // CHECK: mqtmem.peres() %[[Q0]], %[[Q1]]
        // CHECK: mqtmem.peresdg() %[[Q0]], %[[Q1]]
        // CHECK: mqtmem.dcx() %[[Q0]], %[[Q1]]
        // CHECK: mqtmem.ecr() %[[Q0]], %[[Q1]]

        %q0 = mqtmem.qubit 0
        %q1 = mqtmem.qubit 1

        mqtmem.swap() %q0, %q1
        mqtmem.iswap() %q0, %q1
        mqtmem.iswapdg() %q0, %q1
        mqtmem.peres() %q0, %q1
        mqtmem.peresdg() %q0, %q1
        mqtmem.dcx() %q0, %q1
        mqtmem.ecr() %q0, %q1

        return
    }
}


// -----
// This test checks if parameterized multiple qubit gates are parsed and handled correctly.
module {
    // CHECK-LABEL: func.func @testMultipleQubitRotationOp
    func.func @testMultipleQubitRotationOp() {
        // CHECK: %[[P0:.*]] = arith.constant 3.000000e-01 : f64
        // CHECK: mqtmem.rxx(%[[P0]]) %[[Q0:.*]], %[[Q1:.*]]
        // CHECK: mqtmem.ryy(%[[P0]]) %[[Q0]], %[[Q1]]
        // CHECK: mqtmem.rzz(%[[P0]]) %[[Q0]], %[[Q1]]
        // CHECK: mqtmem.rzx(%[[P0]]) %[[Q0]], %[[Q1]]
        // CHECK: mqtmem.xxminusyy(%[[P0]], %[[P0]]) %[[Q0]], %[[Q1]]
        // CHECK: mqtmem.xxplusyy(%[[P0]], %[[P0]]) %[[Q0]], %[[Q1]]

        %p0 = arith.constant 3.000000e-01 : f64
        %q0 = mqtmem.qubit 0
        %q1 = mqtmem.qubit 1

        mqtmem.rxx(%p0) %q0, %q1
        mqtmem.ryy(%p0) %q0, %q1
        mqtmem.rzz(%p0) %q0, %q1
        mqtmem.rzx(%p0) %q0, %q1
        mqtmem.xxminusyy(%p0, %p0) %q0, %q1
        mqtmem.xxplusyy(%p0, %p0) %q0, %q1

        return
    }
}

// -----
// This test expects an error to be thrown when supplying a secondary as well as a ctrl qubit.
module {
    func.func @testParamOpInvalidFormat() {
        %q0 = mqtmem.qubit 0
        %q1 = mqtmem.qubit 1

        // expected-error@+1 {{'mqtmem.swap' op expects either a secondary OR a control qubit.}}
        mqtmem.swap() %q0, %q1 ctrl %q1

        return
    }
}

// -----
// This test expects an error to be thrown when parsing a parameterised operation.
module {
    func.func @testParamOpInvalidFormat() {
        %p0 = arith.constant 3.000000e-01 : f64
        %q0 = mqtmem.qubit 0

        // expected-error@+1 {{operation expects exactly 3 parameters but got 2}}
        mqtmem.u(%p0, %p0) %q0

        return
    }
}

// -----
// This test checks if a measurement op with a mismatch between in-qubits and out-bits throws an error as expected.
module {
    func.func @testMeasureMismatchInOutBits() {
        %q0 = mqtmem.qubit 0
        %q1 = mqtmem.qubit 1

        // expected-error@+1 {{'mqtmem.measure' op number of input qubits (2) and output bits (0) must be the same}}
        "mqtmem.measure"(%q0, %q1) : (!mqtmem.DeviceQubit, !mqtmem.DeviceQubit) -> ()

        return
    }
}

// -----
// This test checks if a no-target arity constraint operation detects correctly when a target is provided.
module {
    func.func @testNoTargetContainsTarget() {
        %q0 = mqtmem.qubit 0

        %c0_f64 = arith.constant 3.000000e-01 : f64

        // expected-error@+1 {{number of input qubits (1) must be 0}}
        mqtmem.gphase(%c0_f64) %q0

        return
    }
}

// -----
// This test checks if static parameters for rotation operations are parsed correctly.
module {
    // CHECK-LABEL: func.func @testStaticParameters
    func.func @testStaticParameters() {
        // CHECK: mqtmem.u( static [1.000000e-01, 2.000000e-01, 3.000000e-01]) %[[ANY:.*]]
        // CHECK: mqtmem.u( static [1.000000e-01, 2.000000e-01, 3.000000e-01] mask [true, true, true]) %[[ANY:.*]]

        %q0 = mqtmem.qubit 0

        mqtmem.u(static [1.00000e-01, 2.00000e-01, 3.00000e-01]) %q0
        mqtmem.u(static [1.00000e-01, 2.00000e-01, 3.00000e-01] mask [true, true, true]) %q0

        return
    }
}

// -----
// This test checks if static parameters together with dynamic parameters for rotation operations are parsed correctly.
module {
    // CHECK-LABEL: func.func @testStaticAndDynamicParameters
    func.func @testStaticAndDynamicParameters() {
        // CHECK: mqtmem.u(%[[ANY:.*]] static [1.000000e-01, 2.000000e-01] mask [true, false, true]) %[[ANY:.*]]

        %q0 = mqtmem.qubit 0
        %c0_f64 = arith.constant 3.000000e-01 : f64

        mqtmem.u(%c0_f64 static [1.00000e-01, 2.00000e-01] mask [true, false, true]) %q0

        return
    }
}

// -----
// This test checks if static parameters and dynamic parameters surpassing the limit of parameters together is detected correctly.
module {
    func.func @testTooManyStaticAndDynamicParameters() {
        %q0 = mqtmem.qubit 0
        %c0_f64 = arith.constant 3.000000e-01 : f64

        // expected-error@+1 {{operation expects exactly 3 parameters but got 4}}
        mqtmem.u(%c0_f64, %c0_f64 static [1.00000e-01, 2.00000e-01] mask [true, false, true]) %q0

        return
    }
}

// -----
// This test checks if static parameters and dynamic parameters being passed without a mask is detected correctly.
module {
    func.func @testStaticAndDynamicParametersNoMask() {
        %q0 = mqtmem.qubit 0
        %c0_f64 = arith.constant 3.000000e-01 : f64

        // expected-error@+1 {{operation has mixed dynamic and static parameters but no parameter mask}}
        mqtmem.u(%c0_f64 static [1.00000e-01, 2.00000e-01]) %q0

        return
    }
}

// -----
// This test checks if a static parameter mask with incorrect size is detected correctly.
module {
    func.func @testStaticAndDynamicParametersWrongSizeMask() {
        %q0 = mqtmem.qubit 0
        %c0_f64 = arith.constant 3.000000e-01 : f64

        // expected-error@+1 {{operation expects exactly 3 parameters but has a parameter mask with 2 entries}}
        mqtmem.u(%c0_f64 static [1.00000e-01, 2.00000e-01] mask [true, true]) %q0

        return
    }
}

// -----
// This test checks if a static parameter mask with an incorrect number of true entries is detected correctly.
module {
    func.func @testStaticAndDynamicParametersIncorrectTrueEntriesInMask() {
        %q0 = mqtmem.qubit 0
        %c0_f64 = arith.constant 3.000000e-01 : f64

        // expected-error@+1 {{operation has 2 static parameter(s) but has a parameter mask with 3 true entries}}
        mqtmem.u(%c0_f64 static [1.00000e-01, 2.00000e-01] mask [true, true, true]) %q0

        return
    }
}

// -----
// This test checks if a static parameter mask with `true` parameters even though the operation has no static parameters is detected correctly.
module {
    func.func @testParametersMaskWithTrueEntriesButNoStaticParameters() {
        %q0 = mqtmem.qubit 0
        %c0_f64 = arith.constant 3.000000e-01 : f64

        // expected-error@+1 {{operation has no static parameter but has a parameter mask with 1 true entries}}
        mqtmem.u(%c0_f64, %c0_f64, %c0_f64 static [] mask [true, false, false]) %q0

        return
    }
}

// -----
// This test checks if a no-control gate being passed a control is detected correctly.
module {
    func.func @testNoControlWithControl() {
        %q0 = mqtmem.qubit 0
        %q1 = mqtmem.qubit 1

        // expected-error@+1 {{'mqtmem.barrier' op Gate marked as NoControl should not have control qubits}}
        mqtmem.barrier() %q0 ctrl %q1

        return
    }
}

// -----
// This test checks if a Bell state is parsed and handled correctly by using many instructions tested above.
module {
    // CHECK-LABEL: func.func @bellState()
    func.func @bellState() {
        // CHECK: %[[Q0:.*]] = mqtmem.qubit 0
        // CHECK: %[[Q1:.*]] = mqtmem.qubit 1
        // CHECK: mqtmem.h() %[[Q0]]
        // CHECK: mqtmem.x() %[[Q1]] ctrl %[[Q0]]
        // CHECK: %[[M0:.*]] = "mqtmem.measure"(%[[Q0]]) : (!mqtmem.DeviceQubit) -> i1
        // CHECK: %[[M1:.*]] = "mqtmem.measure"(%[[Q1]]) : (!mqtmem.DeviceQubit) -> i1

        %q0 = mqtmem.qubit 0
        %q1 = mqtmem.qubit 1

        mqtmem.h() %q0
        mqtmem.x() %q1 ctrl %q0
        %m0 = "mqtmem.measure"(%q0) : (!mqtmem.DeviceQubit) -> i1
        %m1 = "mqtmem.measure"(%q1) : (!mqtmem.DeviceQubit) -> i1

        return
    }
}
