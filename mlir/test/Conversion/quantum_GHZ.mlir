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
    // CHECK: %[[QREG:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister
    %0 = quantum.alloc( 3) : !quantum.reg

    // CHECK: %[[QR1:.*]], %[[Q0:.*]] = "mqtopt.extractQubit"(%[[QREG]]) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[QR2:.*]], %[[Q1:.*]] = "mqtopt.extractQubit"(%[[QR1]]) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[QR3:.*]], %[[Q2:.*]] = "mqtopt.extractQubit"(%[[QR2]]) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit

    // CHECK: %[[H:.*]] = mqtopt.H( static [] mask []) %[[Q0]] : !mqtopt.Qubit
    // CHECK: %[[CX1:.*]]:2 = mqtopt.x( static [] mask []) %[[Q1]] ctrl %[[H]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[CX2:.*]]:2 = mqtopt.x( static [] mask []) %[[Q2]] ctrl %[[CX1]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
    %out_h = quantum.custom "Hadamard"() %1 : !quantum.bit
    %out_qubits:2 = quantum.custom "CNOT"() %out_h, %2 : !quantum.bit, !quantum.bit
    %out_qubits_0:2 = quantum.custom "CNOT"() %out_qubits#1, %3 : !quantum.bit, !quantum.bit

    // CHECK: %[[R0:.*]] = "mqtopt.insertQubit"(%[[QR3]], %[[CX1]]#0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %[[R1:.*]] = "mqtopt.insertQubit"(%[[R0]], %[[CX2]]#0) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %[[R2:.*]] = "mqtopt.insertQubit"(%[[R1]], %[[CX2]]#1) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    %4 = quantum.insert %0[ 0], %out_qubits#0 : !quantum.reg, !quantum.bit
    %5 = quantum.insert %4[ 1], %out_qubits_0#0 : !quantum.reg, !quantum.bit
    %6 = quantum.insert %5[ 2], %out_qubits_0#1 : !quantum.reg, !quantum.bit

    // CHECK: "mqtopt.deallocQubitRegister"(%[[R2]]) : (!mqtopt.QubitRegister) -> ()
    quantum.dealloc %6 : !quantum.reg

    return
  }
}
