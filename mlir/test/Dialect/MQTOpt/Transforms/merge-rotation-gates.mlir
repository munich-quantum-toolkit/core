// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --merge-rotation-gates | FileCheck %s

// -----
// This test checks that consecutive p gates are merged correctly.

module {
  // CHECK-LABEL: func.func @testMergeRxGates
  func.func @testMergeRxGates() {
    // CHECK: %[[Res_2:.*]] = arith.constant 2.000000e+00 : f64
    // CHECK: %[[ANY:.*]] = mqtopt.p(%[[Res_2]]) %[[ANY:.*]] : !mqtopt.Qubit
    // CHECK-NOT: %[[ANY:.*]] = mqtopt.p(%[[ANY:.*]]) %[[ANY:.*]] : !mqtopt.Qubit

    %i0 = arith.constant 0 : index
    %qreg = memref.alloc() : memref<1x!mqtopt.Qubit>
    %q0_0 = memref.load %qreg[%i0] : memref<1x!mqtopt.Qubit>

    %c_0 = arith.constant 1.000000e+00 : f64
    %q0_1 = mqtopt.p(%c_0) %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.p(%c_0) %q0_1 : !mqtopt.Qubit

    memref.store %q0_2, %qreg[%i0] : memref<1x!mqtopt.Qubit>
    memref.dealloc %qreg : memref<1x!mqtopt.Qubit>

    return
  }
}

// -----
// This test checks that consecutive rx gates are merged correctly.

module {
  // CHECK-LABEL: func.func @testMergeRxGates
  func.func @testMergeRxGates() {
    // CHECK: %[[Res_3:.*]] = arith.constant 3.000000e+00 : f64
    // CHECK: %[[ANY:.*]] = mqtopt.rx(%[[Res_3]]) %[[ANY:.*]] : !mqtopt.Qubit
    // CHECK-NOT: %[[ANY:.*]] = mqtopt.rx(%[[ANY:.*]]) %[[ANY:.*]] : !mqtopt.Qubit

    %i0 = arith.constant 0 : index
    %qreg = memref.alloc() : memref<1x!mqtopt.Qubit>
    %q0_0 = memref.load %qreg[%i0] : memref<1x!mqtopt.Qubit>

    %c_0 = arith.constant 1.000000e+00 : f64
    %q0_1 = mqtopt.rx(%c_0) %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.rx(%c_0) %q0_1 : !mqtopt.Qubit
    %q0_3 = mqtopt.rx(%c_0) %q0_2 : !mqtopt.Qubit

    memref.store %q0_3, %qreg[%i0] : memref<1x!mqtopt.Qubit>
    memref.dealloc %qreg : memref<1x!mqtopt.Qubit>

    return
  }
}

// -----
// This test checks that consecutive ry gates are merged correctly.

module {
  // CHECK-LABEL: func.func @testMergeRyGates
  func.func @testMergeRyGates(%c_0 : f64, %c_1 : f64) {
    // CHECK: %[[Res:.*]] = arith.addf %arg0, %arg1 : f64
    // CHECK: %[[ANY:.*]] = mqtopt.ry(%[[Res]]) %[[ANY:.*]] : !mqtopt.Qubit
    // CHECK-NOT: %[[ANY:.*]] = mqtopt.ry(%[[ANY:.*]]) %[[ANY:.*]] : !mqtopt.Qubit

    %i0 = arith.constant 0 : index
    %qreg = memref.alloc() : memref<1x!mqtopt.Qubit>
    %q0_0 = memref.load %qreg[%i0] : memref<1x!mqtopt.Qubit>

    %q0_1 = mqtopt.ry(%c_0) %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.ry(%c_1) %q0_1 : !mqtopt.Qubit

    memref.store %q0_2, %qreg[%i0] : memref<1x!mqtopt.Qubit>
    memref.dealloc %qreg : memref<1x!mqtopt.Qubit>

    return
  }
}

// -----
// This test checks that consecutive rz gates are merged correctly.

module {
  // CHECK-LABEL: func.func @testMergeRzGates
  func.func @testMergeRzGates() {
    // CHECK: %[[Res_3:.*]] = arith.constant 3.000000e+00 : f64
    // CHECK: %[[ANY:.*]] = mqtopt.rz(%[[Res_3]]) %[[ANY:.*]] : !mqtopt.Qubit
    // CHECK-NOT: %[[ANY:.*]] = mqtopt.rz(%[[ANY:.*]]) %[[ANY:.*]] : !mqtopt.Qubit

    %i0 = arith.constant 0 : index
    %qreg = memref.alloc() : memref<1x!mqtopt.Qubit>
    %q0_0 = memref.load %qreg[%i0] : memref<1x!mqtopt.Qubit>

    %c_0 = arith.constant 1.000000e+00 : f64
    %c_1 = arith.constant 2.000000e+00 : f64
    %q0_1 = mqtopt.rz(%c_0) %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.rz(%c_1) %q0_1 : !mqtopt.Qubit

    memref.store %q0_2, %qreg[%i0] : memref<1x!mqtopt.Qubit>
    memref.dealloc %qreg : memref<1x!mqtopt.Qubit>

    return
  }
}

// -----
// This test checks that incompatible single-qubit gates are not merged.
// The gates cannot be merged because their types are different.

module {
  // CHECK-LABEL: func.func @testDoNotMergeSingleQubitGatesDifferentGates
  func.func @testDoNotMergeSingleQubitGatesDifferentGates() {
    // CHECK: %[[Res_2:.*]] = arith.constant 2.000000e+00 : f64
    // CHECK: %[[Res_1:.*]] = arith.constant 1.000000e+00 : f64
    // CHECK: %[[Q0_1:.*]] = mqtopt.rx(%[[Res_1]]) %[[ANY:.*]] : !mqtopt.Qubit
    // CHECK: %[[Q0_2:.*]] = mqtopt.ry(%[[Res_1]]) %[[Q0_1]] : !mqtopt.Qubit
    // CHECK: %[[ANY:.*]] = mqtopt.rz(%[[Res_2]]) %[[Q0_2]] : !mqtopt.Qubit

    %i0 = arith.constant 0 : index
    %qreg = memref.alloc() : memref<1x!mqtopt.Qubit>
    %q0_0 = memref.load %qreg[%i0] : memref<1x!mqtopt.Qubit>

    %c_0 = arith.constant 1.000000e+00 : f64
    %c_1 = arith.constant 2.000000e+00 : f64
    %q0_1 = mqtopt.rx(%c_0) %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.ry(%c_0) %q0_1 : !mqtopt.Qubit
    %q0_3 = mqtopt.rz(%c_1) %q0_2 : !mqtopt.Qubit

    memref.store %q0_3, %qreg[%i0] : memref<1x!mqtopt.Qubit>
    memref.dealloc %qreg : memref<1x!mqtopt.Qubit>

    return
  }
}

// -----
// This test checks that incompatible single-qubit gates are not merged.
// The gates cannot be merged because they act on different qubits.

module {
  // CHECK-LABEL: func.func @testDoNotMergeSingleQubitGatesIndependentGates
  func.func @testDoNotMergeSingleQubitGatesIndependentGates() {
    // CHECK: %[[Res_2:.*]] = arith.constant 2.000000e+00 : f64
    // CHECK: %[[Res_1:.*]] = arith.constant 1.000000e+00 : f64
    // CHECK: %[[ANY:.*]] = mqtopt.rx(%[[Res_1]]) %[[ANY:.*]] : !mqtopt.Qubit
    // CHECK: %[[ANY:.*]] = mqtopt.rx(%[[Res_2]]) %[[ANY:.*]] : !mqtopt.Qubit

    %i1 = arith.constant 1 : index
    %i0 = arith.constant 0 : index
    %qreg = memref.alloc() : memref<2x!mqtopt.Qubit>
    %q0_0 = memref.load %qreg[%i0] : memref<2x!mqtopt.Qubit>
    %q1_0 = memref.load %qreg[%i1] : memref<2x!mqtopt.Qubit>

    %c_0 = arith.constant 1.000000e+00 : f64
    %c_1 = arith.constant 2.000000e+00 : f64
    %q0_1 = mqtopt.rx(%c_0) %q0_0 : !mqtopt.Qubit
    %q1_1 = mqtopt.rx(%c_1) %q1_0 : !mqtopt.Qubit

    memref.store %q0_1, %qreg[%i0] : memref<2x!mqtopt.Qubit>
    memref.store %q1_1, %qreg[%i1] : memref<2x!mqtopt.Qubit>
    memref.dealloc %qreg : memref<2x!mqtopt.Qubit>

    return
  }
}

// -----
// This test checks that consecutive rxx gates are merged correctly.

module {
  // CHECK-LABEL: func.func @testMergeRxxGates
  func.func @testMergeRxxGates() {
    // CHECK: %[[Res_3:.*]] = arith.constant 3.000000e+00 : f64
    // CHECK: %[[ANY:.*]]:2 = mqtopt.rxx(%[[Res_3]]) %[[ANY:.*]], %[[ANY:.*]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK-NOT: %[[ANY:.*]]:2 = mqtopt.rxx(%[[ANY:.*]]) %[[ANY:.*]], %[[ANY:.*]] : !mqtopt.Qubit, !mqtopt.Qubit

    %i1 = arith.constant 1 : index
    %i0 = arith.constant 0 : index
    %qreg = memref.alloc() : memref<2x!mqtopt.Qubit>
    %q0_0 = memref.load %qreg[%i0] : memref<2x!mqtopt.Qubit>
    %q1_0 = memref.load %qreg[%i1] : memref<2x!mqtopt.Qubit>

    %c_0 = arith.constant 1.000000e+00 : f64
    %q01_1:2 = mqtopt.rxx(%c_0) %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
    %q01_2:2 = mqtopt.rxx(%c_0) %q01_1#0, %q01_1#1 : !mqtopt.Qubit, !mqtopt.Qubit
    %q01_3:2 = mqtopt.rxx(%c_0) %q01_2#0, %q01_2#1 : !mqtopt.Qubit, !mqtopt.Qubit

    memref.store %q01_3#0, %qreg[%i0] : memref<2x!mqtopt.Qubit>
    memref.store %q01_3#1, %qreg[%i1] : memref<2x!mqtopt.Qubit>
    memref.dealloc %qreg : memref<2x!mqtopt.Qubit>

    return
  }
}

// -----
// This test checks that consecutive ryy gates are merged correctly.

module {
  // CHECK-LABEL: func.func @testMergeRyyGates
  func.func @testMergeRyyGates(%c_0 : f64, %c_1 : f64) {
    // CHECK: %[[Res:.*]] = arith.addf %arg0, %arg1 : f64
    // CHECK: %[[ANY:.*]]:2 = mqtopt.ryy(%[[Res]]) %[[ANY:.*]], %[[ANY:.*]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK-NOT: %[[ANY:.*]]:2 = mqtopt.ryy(%[[ANY:.*]]) %[[ANY:.*]], %[[ANY:.*]] : !mqtopt.Qubit, !mqtopt.Qubit

    %i1 = arith.constant 1 : index
    %i0 = arith.constant 0 : index
    %qreg = memref.alloc() : memref<2x!mqtopt.Qubit>
    %q0_0 = memref.load %qreg[%i0] : memref<2x!mqtopt.Qubit>
    %q1_0 = memref.load %qreg[%i1] : memref<2x!mqtopt.Qubit>

    %q01_1:2 = mqtopt.ryy(%c_0) %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
    %q01_2:2 = mqtopt.ryy(%c_1) %q01_1#0, %q01_1#1 : !mqtopt.Qubit, !mqtopt.Qubit

    memref.store %q01_2#0, %qreg[%i0] : memref<2x!mqtopt.Qubit>
    memref.store %q01_2#1, %qreg[%i1] : memref<2x!mqtopt.Qubit>
    memref.dealloc %qreg : memref<2x!mqtopt.Qubit>

    return
  }
}

// -----
// This test checks that consecutive rzz gates are merged correctly.

module {
  // CHECK-LABEL: func.func @testMergeRzzGates
  func.func @testMergeRzzGates() {
    // CHECK: %[[Res_3:.*]] = arith.constant 3.000000e+00 : f64
    // CHECK: %[[ANY:.*]]:2 = mqtopt.rzz(%[[Res_3]]) %[[ANY:.*]], %[[ANY:.*]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK-NOT: %[[ANY:.*]]:2 = mqtopt.rzz(%[[ANY:.*]]) %[[ANY:.*]], %[[ANY:.*]] : !mqtopt.Qubit, !mqtopt.Qubit

    %i1 = arith.constant 1 : index
    %i0 = arith.constant 0 : index
    %qreg = memref.alloc() : memref<2x!mqtopt.Qubit>
    %q0_0 = memref.load %qreg[%i0] : memref<2x!mqtopt.Qubit>
    %q1_0 = memref.load %qreg[%i1] : memref<2x!mqtopt.Qubit>

    %c_0 = arith.constant 1.000000e+00 : f64
    %c_1 = arith.constant 2.000000e+00 : f64
    %q01_1:2 = mqtopt.rzz(%c_0) %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
    %q01_2:2 = mqtopt.rzz(%c_1) %q01_1#0, %q01_1#1 : !mqtopt.Qubit, !mqtopt.Qubit

    memref.store %q01_2#0, %qreg[%i0] : memref<2x!mqtopt.Qubit>
    memref.store %q01_2#1, %qreg[%i1] : memref<2x!mqtopt.Qubit>
    memref.dealloc %qreg : memref<2x!mqtopt.Qubit>

    return
  }
}

// -----
// This test checks that consecutive rzx gates are merged correctly.

module {
  // CHECK-LABEL: func.func @testMergeRzxGates
  func.func @testMergeRzxGates() {
    // CHECK: %[[Res_3:.*]] = arith.constant 3.000000e+00 : f64
    // CHECK: %[[ANY:.*]]:2 = mqtopt.rzx(%[[Res_3]]) %[[ANY:.*]], %[[ANY:.*]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK-NOT: %[[ANY:.*]]:2 = mqtopt.rzx(%[[ANY:.*]]) %[[ANY:.*]], %[[ANY:.*]] : !mqtopt.Qubit, !mqtopt.Qubit

    %i1 = arith.constant 1 : index
    %i0 = arith.constant 0 : index
    %qreg = memref.alloc() : memref<2x!mqtopt.Qubit>
    %q0_0 = memref.load %qreg[%i0] : memref<2x!mqtopt.Qubit>
    %q1_0 = memref.load %qreg[%i1] : memref<2x!mqtopt.Qubit>

    %c_0 = arith.constant 1.000000e+00 : f64
    %c_1 = arith.constant 2.000000e+00 : f64
    %q01_1:2 = mqtopt.rzx(%c_0) %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
    %q01_2:2 = mqtopt.rzx(%c_1) %q01_1#0, %q01_1#1 : !mqtopt.Qubit, !mqtopt.Qubit

    memref.store %q01_2#0, %qreg[%i0] : memref<2x!mqtopt.Qubit>
    memref.store %q01_2#1, %qreg[%i1] : memref<2x!mqtopt.Qubit>
    memref.dealloc %qreg : memref<2x!mqtopt.Qubit>

    return
  }
}

// -----
// This test checks that incompatible multi-qubit gates are not merged.
// The gates cannot be merged because their types are different.

module {
  // CHECK-LABEL: func.func @testDoNotMergeMultiQubitGatesDifferentGates
  func.func @testDoNotMergeMultiQubitGatesDifferentGates() {
    // CHECK: %[[Res_2:.*]] = arith.constant 2.000000e+00 : f64
    // CHECK: %[[Res_1:.*]] = arith.constant 1.000000e+00 : f64
    // CHECK: %[[Q01_1:.*]]:2 = mqtopt.rxx(%[[Res_1]]) %[[ANY:.*]], %[[ANY:.*]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[Q01_2:.*]]:2 = mqtopt.ryy(%[[Res_1]]) %[[Q01_1]]#0, %[[Q01_1]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[ANY:.*]]:2 = mqtopt.rzz(%[[Res_2]]) %[[Q01_2]]#0, %[[Q01_2]]#1 : !mqtopt.Qubit, !mqtopt.Qubit

    %i1 = arith.constant 1 : index
    %i0 = arith.constant 0 : index
    %qreg = memref.alloc() : memref<2x!mqtopt.Qubit>
    %q0_0 = memref.load %qreg[%i0] : memref<2x!mqtopt.Qubit>
    %q1_0 = memref.load %qreg[%i1] : memref<2x!mqtopt.Qubit>

    %c_0 = arith.constant 1.000000e+00 : f64
    %c_1 = arith.constant 2.000000e+00 : f64
    %q01_1:2 = mqtopt.rxx(%c_0) %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
    %q01_2:2 = mqtopt.ryy(%c_0) %q01_1#0, %q01_1#1 : !mqtopt.Qubit, !mqtopt.Qubit
    %q01_3:2 = mqtopt.rzz(%c_1) %q01_2#0, %q01_2#1 : !mqtopt.Qubit, !mqtopt.Qubit

    memref.store %q01_3#0, %qreg[%i0] : memref<2x!mqtopt.Qubit>
    memref.store %q01_3#1, %qreg[%i1] : memref<2x!mqtopt.Qubit>
    memref.dealloc %qreg : memref<2x!mqtopt.Qubit>

    return
  }
}

// -----
// This test checks that incompatible multi-qubit gates are not merged.
// The gates cannot be merged because their types are different.

module {
  // CHECK-LABEL: func.func @testDoNotMergeMultiQubitGatesIndependentGates
  func.func @testDoNotMergeMultiQubitGatesIndependentGates() {
    // CHECK: %[[Res_2:.*]] = arith.constant 2.000000e+00 : f64
    // CHECK: %[[Res_1:.*]] = arith.constant 1.000000e+00 : f64
    // CHECK: %[[Q0_1:.*]]:2 = mqtopt.rxx(%[[Res_1]]) %[[ANY:.*]], %[[ANY:.*]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[ANY:.*]]:2 = mqtopt.rxx(%[[Res_2]]) %[[Q0_1:.*]]#1, %[[ANY:.*]] : !mqtopt.Qubit, !mqtopt.Qubit

    %i2 = arith.constant 2 : index
    %i1 = arith.constant 1 : index
    %i0 = arith.constant 0 : index
    %qreg = memref.alloc() : memref<3x!mqtopt.Qubit>
    %q0_0 = memref.load %qreg[%i0] : memref<3x!mqtopt.Qubit>
    %q1_0 = memref.load %qreg[%i1] : memref<3x!mqtopt.Qubit>
    %q2_0 = memref.load %qreg[%i2] : memref<3x!mqtopt.Qubit>

    %c_0 = arith.constant 1.000000e+00 : f64
    %c_1 = arith.constant 2.000000e+00 : f64
    %q01_1:2 = mqtopt.rxx(%c_0) %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
    %q12_1:2 = mqtopt.rxx(%c_1) %q01_1#1, %q2_0 : !mqtopt.Qubit, !mqtopt.Qubit

    memref.store %q01_1#0, %qreg[%i0] : memref<3x!mqtopt.Qubit>
    memref.store %q12_1#0, %qreg[%i1] : memref<3x!mqtopt.Qubit>
    memref.store %q12_1#1, %qreg[%i2] : memref<3x!mqtopt.Qubit>
    memref.dealloc %qreg : memref<3x!mqtopt.Qubit>

    return
  }
}

// -----
// This test checks that incompatible multi-qubit gates are not merged.
// The gates cannot be merged because their input qubits do not have the same order.
// This test should fail when a canonicalization pass is implemented with #1031.

module {
  // CHECK-LABEL: func.func @testDoNotMergeMultiQubitGatesDifferentInputQubitOrder
  func.func @testDoNotMergeMultiQubitGatesDifferentInputQubitOrder() {
    // CHECK: %[[Res_2:.*]] = arith.constant 2.000000e+00 : f64
    // CHECK: %[[Res_1:.*]] = arith.constant 1.000000e+00 : f64
    // CHECK: %[[Q0_1:.*]]:2 = mqtopt.rxx(%[[Res_1]]) %[[ANY:.*]], %[[ANY:.*]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[ANY:.*]]:2 = mqtopt.rxx(%[[Res_2]]) %[[Q0_1]]#1, %[[Q0_1]]#0 : !mqtopt.Qubit, !mqtopt.Qubit

    %i1 = arith.constant 1 : index
    %i0 = arith.constant 0 : index
    %qreg = memref.alloc() : memref<2x!mqtopt.Qubit>
    %q0_0 = memref.load %qreg[%i0] : memref<2x!mqtopt.Qubit>
    %q1_0 = memref.load %qreg[%i1] : memref<2x!mqtopt.Qubit>

    %c_0 = arith.constant 1.000000e+00 : f64
    %c_1 = arith.constant 2.000000e+00 : f64
    %q01_1:2 = mqtopt.rxx(%c_0) %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
    %q01_2:2 = mqtopt.rxx(%c_1) %q01_1#1, %q01_1#0 : !mqtopt.Qubit, !mqtopt.Qubit

    memref.store %q01_2#0, %qreg[%i0] : memref<2x!mqtopt.Qubit>
    memref.store %q01_2#1, %qreg[%i1] : memref<2x!mqtopt.Qubit>
    memref.dealloc %qreg : memref<2x!mqtopt.Qubit>

    return
  }
}

// -----
// This test checks that consecutive gphase gates with controls are merged correctly.

module {
  // CHECK-LABEL: func.func @testMergeGphaseWithControls
  func.func @testMergeGphaseWithControls() {
    // CHECK: %[[Res_3:.*]] = arith.constant 3.000000e+00 : f64
    // CHECK: %[[ANY:.*]] = mqtopt.gphase(%[[Res_3]]) ctrl %[[ANY:.*]] : ctrl !mqtopt.Qubit
    // CHECK-NOT: %[[ANY:.*]] = mqtopt.gphase(%[[ANY:.*]]) %[[ANY:.*]] : !mqtopt.Qubit

    %i0 = arith.constant 0 : index
    %qreg = memref.alloc() : memref<1x!mqtopt.Qubit>
    %q0_0 = memref.load %qreg[%i0] : memref<1x!mqtopt.Qubit>

    %c_0 = arith.constant 1.000000e+00 : f64
    %q0_1 = mqtopt.gphase(%c_0) ctrl %q0_0 : ctrl !mqtopt.Qubit
    %q0_2 = mqtopt.gphase(%c_0) ctrl %q0_1 : ctrl !mqtopt.Qubit
    %q0_3 = mqtopt.gphase(%c_0) ctrl %q0_2 : ctrl !mqtopt.Qubit

    memref.store %q0_3, %qreg[%i0] : memref<1x!mqtopt.Qubit>
    memref.dealloc %qreg : memref<1x!mqtopt.Qubit>

    return
  }
}

// -----
// This test checks that consecutive gphase gates without controls are not merged.
// The current implementation does not support merging gates without users.

module {
  // CHECK-LABEL: func.func @testDoNotMergeGphaseWithoutControls
  func.func @testDoNotMergeGphaseWithoutControls() {
    // CHECK: %[[Res_1:.*]] = arith.constant 1.000000e+00 : f64
    // CHECK: mqtopt.gphase(%[[Res_1]])
    // CHECK: mqtopt.gphase(%[[Res_1]])
    // CHECK: mqtopt.gphase(%[[Res_1]])
    // CHECK-NOT: mqtopt.gphase(%[[ANY:.*]])

    %i0 = arith.constant 0 : index
    %qreg = memref.alloc() : memref<1x!mqtopt.Qubit>
    %q0_0 = memref.load %qreg[%i0] : memref<1x!mqtopt.Qubit>

    %c_0 = arith.constant 1.000000e+00 : f64
    mqtopt.gphase(%c_0) : ()
    mqtopt.gphase(%c_0) : ()
    mqtopt.gphase(%c_0) : ()

    memref.store %q0_0, %qreg[%i0] : memref<1x!mqtopt.Qubit>
    memref.dealloc %qreg : memref<1x!mqtopt.Qubit>

    return
  }
}

// -----
// This test checks that controlled rotation gates that have different pos/neg ctrl distributions are not merged.

module {
  // CHECK-LABEL: func.func @testDoNotMergeMultiQubitGatesDifferentControlSizes
  func.func @testDoNotMergeMultiQubitGatesDifferentControlSizes() {
    // CHECK: %[[Res_2:.*]] = arith.constant 2.000000e+00 : f64
    // CHECK: %[[Res_1:.*]] = arith.constant 1.000000e+00 : f64
    // CHECK: %[[Q0_1:.*]]:2, %[[Q2_1:.*]] = mqtopt.gphase(%[[Res_1]]) ctrl %[[ANY:.*]], %[[ANY:.*]] nctrl %[[ANY:.*]]: ctrl !mqtopt.Qubit, !mqtopt.Qubit nctrl !mqtopt.Qubit
    // CHECK: %[[ANY:.*]], %[[ANY:.*]]:2 = mqtopt.gphase(%[[Res_2]]) ctrl %[[Q0_1]]#0 nctrl %[[Q0_1]]#1, %[[Q2_1]] : ctrl !mqtopt.Qubit nctrl !mqtopt.Qubit, !mqtopt.Qubit

    %i2 = arith.constant 2 : index
    %i1 = arith.constant 1 : index
    %i0 = arith.constant 0 : index
    %qreg = memref.alloc() : memref<3x!mqtopt.Qubit>
    %q0_0 = memref.load %qreg[%i0] : memref<3x!mqtopt.Qubit>
    %q1_0 = memref.load %qreg[%i1] : memref<3x!mqtopt.Qubit>
    %q2_0 = memref.load %qreg[%i2] : memref<3x!mqtopt.Qubit>

    %c_0 = arith.constant 1.000000e+00 : f64
    %c_1 = arith.constant 2.000000e+00 : f64
    %q012_1:3 = mqtopt.gphase(%c_0) ctrl %q0_0, %q1_0 nctrl %q2_0 : ctrl !mqtopt.Qubit, !mqtopt.Qubit nctrl !mqtopt.Qubit
    %q012_2:3 = mqtopt.gphase(%c_1) ctrl %q012_1#0 nctrl %q012_1#1, %q012_1#2 : ctrl !mqtopt.Qubit nctrl !mqtopt.Qubit, !mqtopt.Qubit

    memref.store %q012_2#0, %qreg[%i0] : memref<3x!mqtopt.Qubit>
    memref.store %q012_2#1, %qreg[%i1] : memref<3x!mqtopt.Qubit>
    memref.store %q012_2#2, %qreg[%i2] : memref<3x!mqtopt.Qubit>
    memref.dealloc %qreg : memref<3x!mqtopt.Qubit>

    return
  }
}
