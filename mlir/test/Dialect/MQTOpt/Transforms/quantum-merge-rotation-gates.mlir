// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --pass-pipeline='builtin.module(merge-rotation-gates{quaternion-folding})' | FileCheck %s

// 11 Chain merging: Rx+Ry+Rz (tests multiple passes)
// 11 Precedence: Same gates use additive, not quaternion
// 11 Negative cases: P gate excluded, independent qubits
// 11 Numerical accuracy: Use your Python script to generate expected values
// # 1. Test for making sure same gates are merged additvly and not via u-gate (better accuracy)
// # 2. check that 2 u gates are merged via quat
// # 3. check that u and other gate are merged via quat
//
// 11 U gate combinations: U+U, U+Rx, Ry+U, U+Rz, etc.

// -----
// This test checks that consecutive rx and ry gates are merged correctly into a u gate.
module {
  // CHECK-LABEL: func.func @testQuaternionMergeRxRyGates
  func.func @testQuaternionMergeRxRyGates() {
    // CHECK-DAG: %[[Res_1:.*]] = arith.constant 0.49536728921867329 : f64
    // CHECK-DAG: %[[Res_2:.*]] = arith.constant 1.2745557823062943 : f64
    // CHECK-DAG: %[[Res_3:.*]] = arith.constant -1.0754290375762232 : f64
    // CHECK: %[[ANY:.*]] = mqtopt.u(%[[Res_1]], %[[Res_2]], %[[Res_3]]) %[[ANY:.*]] : !mqtopt.Qubit
    // CHECK-NOT: mqtopt.u

    %i0 = arith.constant 0 : index
    %qreg = memref.alloc() : memref<1x!mqtopt.Qubit>
    %q0_0 = memref.load %qreg[%i0] : memref<1x!mqtopt.Qubit>

    %c_0 = arith.constant 1.000000e+00 : f64
    %q0_1 = mqtopt.rx(%c_0) %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.ry(%c_0) %q0_1 : !mqtopt.Qubit

    memref.store %q0_2, %qreg[%i0] : memref<1x!mqtopt.Qubit>
    memref.dealloc %qreg : memref<1x!mqtopt.Qubit>

    return
  }
}

// -----
// This test checks that consecutive rx and rz gates are merged correctly into a u gate.
module {
  // CHECK-LABEL: func.func @testQuaternionMergeRxRzGates
  func.func @testQuaternionMergeRxRzGates() {
    // CHECK-DAG: %[[Res_1:.*]] = arith.constant 1.5707963267948966 : f64
    // CHECK-DAG: %[[Res_2:.*]] = arith.constant 0.99999999999999988 : f64
    // CHECK-DAG: %[[Res_3:.*]] = arith.constant -0.57079632679489656 : f64
    // CHECK: %[[ANY:.*]] = mqtopt.u(%[[Res_1]], %[[Res_2]], %[[Res_3]]) %[[ANY:.*]] : !mqtopt.Qubit
    // CHECK-NOT: mqtopt.u

    %i0 = arith.constant 0 : index
    %qreg = memref.alloc() : memref<1x!mqtopt.Qubit>
    %q0_0 = memref.load %qreg[%i0] : memref<1x!mqtopt.Qubit>

    %c_0 = arith.constant 1.000000e+00 : f64
    %q0_1 = mqtopt.rx(%c_0) %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.rz(%c_0) %q0_1 : !mqtopt.Qubit

    memref.store %q0_2, %qreg[%i0] : memref<1x!mqtopt.Qubit>
    memref.dealloc %qreg : memref<1x!mqtopt.Qubit>

    return
  }
}

// -----
// This test checks that consecutive ry and rx gates are merged correctly into a u gate.
module {
  // CHECK-LABEL: func.func @testQuaternionMergeRyRxGates
  func.func @testQuaternionMergeRyRxGates() {
    // CHECK-DAG: %[[Res_1:.*]] = arith.constant 1.0754290375762232 : f64
    // CHECK-DAG: %[[Res_2:.*]] = arith.constant 1.2745557823062943 : f64
    // CHECK-DAG: %[[Res_3:.*]] = arith.constant -0.49536728921867329 : f64
    // CHECK: %[[ANY:.*]] = mqtopt.u(%[[Res_1]], %[[Res_2]], %[[Res_3]]) %[[ANY:.*]] : !mqtopt.Qubit
    // CHECK-NOT: mqtopt.u

    %i0 = arith.constant 0 : index
    %qreg = memref.alloc() : memref<1x!mqtopt.Qubit>
    %q0_0 = memref.load %qreg[%i0] : memref<1x!mqtopt.Qubit>

    %c_0 = arith.constant 1.000000e+00 : f64
    %q0_1 = mqtopt.ry(%c_0) %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.rx(%c_0) %q0_1 : !mqtopt.Qubit

    memref.store %q0_2, %qreg[%i0] : memref<1x!mqtopt.Qubit>
    memref.dealloc %qreg : memref<1x!mqtopt.Qubit>

    return
  }
}


// -----
// This test checks that consecutive ry and rz gates are merged correctly into a u gate.
module {
  // CHECK-LABEL: func.func @testQuaternionMergeRyRzGates
  func.func @testQuaternionMergeRyRzGates() {
    // CHECK-DAG: %[[Res_1:.*]] = arith.constant 0.000000e+00 : f64
    // CHECK-DAG: %[[Res_2:.*]] = arith.constant 0.99999999999999988 : f64
    // CHECK-DAG: %[[Res_3:.*]] = arith.constant 1.000000e+00 : f64
    // CHECK: %[[ANY:.*]] = mqtopt.u(%[[Res_1]], %[[Res_2]], %[[Res_3]]) %[[ANY:.*]] : !mqtopt.Qubit
    // CHECK-NOT: mqtopt.u

    %i0 = arith.constant 0 : index
    %qreg = memref.alloc() : memref<1x!mqtopt.Qubit>
    %q0_0 = memref.load %qreg[%i0] : memref<1x!mqtopt.Qubit>

    %c_0 = arith.constant 1.000000e+00 : f64
    %q0_1 = mqtopt.ry(%c_0) %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.rz(%c_0) %q0_1 : !mqtopt.Qubit

    memref.store %q0_2, %qreg[%i0] : memref<1x!mqtopt.Qubit>
    memref.dealloc %qreg : memref<1x!mqtopt.Qubit>

    return
  }
}

// -----
// This test checks that consecutive rz and rx gates are merged correctly into a u gate.
module {
  // CHECK-LABEL: func.func @testQuaternionMergeRzRxGates
  func.func @testQuaternionMergeRzRxGates() {
    // CHECK-DAG: %[[Res_1:.*]] = arith.constant 2.5707963267948966 : f64
    // CHECK-DAG: %[[Res_2:.*]] = arith.constant 0.99999999999999988 : f64
    // CHECK-DAG: %[[Res_3:.*]] = arith.constant -1.5707963267948966 : f64
    // CHECK: %[[ANY:.*]] = mqtopt.u(%[[Res_1]], %[[Res_2]], %[[Res_3]]) %[[ANY:.*]] : !mqtopt.Qubit
    // CHECK-NOT: mqtopt.u

    %i0 = arith.constant 0 : index
    %qreg = memref.alloc() : memref<1x!mqtopt.Qubit>
    %q0_0 = memref.load %qreg[%i0] : memref<1x!mqtopt.Qubit>

    %c_0 = arith.constant 1.000000e+00 : f64
    %q0_1 = mqtopt.rz(%c_0) %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.rx(%c_0) %q0_1 : !mqtopt.Qubit

    memref.store %q0_2, %qreg[%i0] : memref<1x!mqtopt.Qubit>
    memref.dealloc %qreg : memref<1x!mqtopt.Qubit>

    return
  }
}

// -----
// This test checks that consecutive rz and ry gates are merged correctly into a u gate.
module {
  // CHECK-LABEL: func.func @testQuaternionMergeRzRyGates
  func.func @testQuaternionMergeRzRyGates() {
    // CHECK-DAG: %[[Res_1:.*]] = arith.constant 1.000000e+00 : f64
    // CHECK-DAG: %[[Res_2:.*]] = arith.constant 0.99999999999999988 : f64
    // CHECK-DAG: %[[Res_3:.*]] = arith.constant 0.000000e+00 : f64
    // CHECK: %[[ANY:.*]] = mqtopt.u(%[[Res_1]], %[[Res_2]], %[[Res_3]]) %[[ANY:.*]] : !mqtopt.Qubit
    // CHECK-NOT: mqtopt.u

    %i0 = arith.constant 0 : index
    %qreg = memref.alloc() : memref<1x!mqtopt.Qubit>
    %q0_0 = memref.load %qreg[%i0] : memref<1x!mqtopt.Qubit>

    %c_0 = arith.constant 1.000000e+00 : f64
    %q0_1 = mqtopt.rz(%c_0) %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.ry(%c_0) %q0_1 : !mqtopt.Qubit

    memref.store %q0_2, %qreg[%i0] : memref<1x!mqtopt.Qubit>
    memref.dealloc %qreg : memref<1x!mqtopt.Qubit>

    return
  }
}
