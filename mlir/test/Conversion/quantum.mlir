// Copyright (c) 2025 Chair for Design Automation, TUM
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s --quantum-to-mqtopt | FileCheck %s

module {
  // CHECK-LABEL: func @bar()
  func.func @bar() {
    // CHECK: %[[PHI:.*]] = arith.constant 3.000000e-01 : f64
    %phi0 = arith.constant 3.000000e-01 : f64

    // CHECK: %[[QREG:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister
    %r0 = quantum.alloc( 3) : !quantum.reg

    // CHECK: %[[QR1:.*]], %[[Q0:.*]] = "mqtopt.extractQubit"(%[[QREG]]) <{index_attr = 0 : i64}>
    // CHECK: %[[QR2:.*]], %[[Q1:.*]] = "mqtopt.extractQubit"(%[[QR1]]) <{index_attr = 1 : i64}>
    // CHECK: %[[QR3:.*]], %[[Q2:.*]] = "mqtopt.extractQubit"(%[[QR2]]) <{index_attr = 2 : i64}>
    %q0 = quantum.extract %r0[ 0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %r0[ 1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %r0[ 2] : !quantum.reg -> !quantum.bit

    // CHECK: %[[H:.*]] = mqtopt.H() %[[Q0]] : !mqtopt.Qubit
    // CHECK: %[[X:.*]] = mqtopt.x() %[[H]] : !mqtopt.Qubit
    // CHECK: %[[Y:.*]] = mqtopt.y() %[[X]] : !mqtopt.Qubit
    // CHECK: %[[Z:.*]] = mqtopt.z() %[[Y]] : !mqtopt.Qubit
    %out_h = quantum.custom "Hadamard"() %q0 : !quantum.bit
    %out_x = quantum.custom "PauliX"() %out_h : !quantum.bit
    %out_y = quantum.custom "PauliY"() %out_x : !quantum.bit
    %out_z = quantum.custom "PauliZ"() %out_y : !quantum.bit

    // CHECK: %[[CNOT:.*]]:2 = mqtopt.x() %[[Q0]] ctrl %[[Q1]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[CY:.*]]:2 = mqtopt.y() %[[CNOT]]#0 ctrl %[[CNOT]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[CZ:.*]]:2 = mqtopt.z() %[[CY]]#0 ctrl %[[CY]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[SWAP:.*]]:2 = mqtopt.swap() %[[CZ]]#1, %[[CZ]]#0 : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[TOF:.*]]:3 = mqtopt.x() %[[SWAP]]#0 ctrl %[[Q2]], %[[SWAP]]#1 : !mqtopt.Qubit, !mqtopt.Qubit, !mqtopt.Qubit
    %cnot:2 = quantum.custom "CNOT"() %q1, %q0 : !quantum.bit, !quantum.bit
    %cy:2 = quantum.custom "CY"() %cnot#1, %cnot#0 : !quantum.bit, !quantum.bit
    %cz:2 = quantum.custom "CZ"() %cy#1, %cy#0 : !quantum.bit, !quantum.bit
    %swap:2 = quantum.custom "SWAP"() %cz#1, %cz#0 : !quantum.bit, !quantum.bit
    %toffoli:3 = quantum.custom "Toffoli"() %q2, %swap#1, %swap#0 : !quantum.bit, !quantum.bit, !quantum.bit

    // CHECK: %[[RX:.*]] = mqtopt.rx(%[[PHI]]) %[[TOF]]#0
    // CHECK: %[[RY:.*]] = mqtopt.ry(%[[PHI]]) %[[RX]]
    // CHECK: %[[RZ:.*]] = mqtopt.rz(%[[PHI]]) %[[RY]]
    // CHECK: %[[P:.*]] = mqtopt.p(%[[PHI]]) %[[RZ]]
    %rx = quantum.custom "RX"(%phi0) %toffoli#0 : !quantum.bit
    %ry = quantum.custom "RY"(%phi0) %rx : !quantum.bit
    %rz = quantum.custom "RZ"(%phi0) %ry : !quantum.bit
    %phaseShift = quantum.custom "PhaseShift"(%phi0) %rz : !quantum.bit

    // CHECK: %[[CRX:.*]]:2 = mqtopt.rx(%[[PHI]]) %[[P]] ctrl %[[TOF]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[CRY:.*]]:2 = mqtopt.ry(%[[PHI]]) %[[CRX]]#0 ctrl %[[CRX]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[CRZ:.*]]:2 = mqtopt.ry(%[[PHI]]) %[[CRY]]#0 ctrl %[[CRY]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[CPS:.*]]:2 = mqtopt.p(%[[PHI]]) %[[CRZ]]#0 ctrl %[[CRZ]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
    %crx:2 = quantum.custom "CRX"(%phi0) %toffoli#1, %phaseShift : !quantum.bit, !quantum.bit
    %cry:2 = quantum.custom "CRY"(%phi0) %crx#1, %crx#0 : !quantum.bit, !quantum.bit
    %crz:2 = quantum.custom "CRY"(%phi0) %cry#1, %cry#0 : !quantum.bit, !quantum.bit
    %cps:2 = quantum.custom "ControlledPhaseShift"(%phi0) %crz#1, %crz#0 : !quantum.bit, !quantum.bit

    // CHECK: %[[QMEAS:.*]], %[[MEAS:.*]] = "mqtopt.measure"(%[[TOF]]#2) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
    %meas:2 = quantum.measure %toffoli#2 : i1, !quantum.bit

    // CHECK: %[[R1:.*]] = "mqtopt.insertQubit"(%[[QR3]], %[[QMEAS]]) <{index_attr = 2 : i64}>
    // CHECK: %[[R2:.*]] = "mqtopt.insertQubit"(%[[R1]], %[[CPS]]#1) <{index_attr = 1 : i64}>
    // CHECK: %[[R3:.*]] = "mqtopt.insertQubit"(%[[R2]], %[[CPS]]#0) <{index_attr = 0 : i64}>
    %r1_2 = quantum.insert %r0[2], %meas#1 : !quantum.reg, !quantum.bit
    %r1_1 = quantum.insert %r1_2[1], %cps#1 : !quantum.reg, !quantum.bit
    %r1_0 = quantum.insert %r1_1[0], %cps#0 : !quantum.reg, !quantum.bit

    // CHECK: "mqtopt.deallocQubitRegister"(%[[R3]]) : (!mqtopt.QubitRegister) -> ()
    quantum.dealloc %r1_0 : !quantum.reg
    return
  }
}
