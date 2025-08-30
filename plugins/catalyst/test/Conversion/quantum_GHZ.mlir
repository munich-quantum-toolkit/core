// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: catalyst --tool=opt \
// RUN:   --load-pass-plugin=%mqt_plugin_path% \
// RUN:   --load-dialect-plugin=%mqt_plugin_path% \
// RUN:   --debug \
// RUN:   --catalyst-pipeline="builtin.module(catalystquantum-to-mqtopt)" \
// RUN:   %s | FileCheck %s

module {
  // CHECK-LABEL: func @bar()
  func.func @bar() {
  // CHECK: %[[QREG:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister
  // CHECK: %[[QREG1:.*]], %[[Q0:.*]] = "mqtopt.extractQubit"(%[[QREG]]) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
  // CHECK: %[[QREG2:.*]], %[[Q1:.*]] = "mqtopt.extractQubit"(%[[QREG1]]) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
  // CHECK: %[[QREG3:.*]], %[[Q2:.*]] = "mqtopt.extractQubit"(%[[QREG2]]) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
  // CHECK: %[[H:.*]] = mqtopt.h( static [] mask []) %[[Q0]] : !mqtopt.Qubit
  // CHECK: %[[X1_T:.*]], %[[X1_C:.*]] = mqtopt.x( static [] mask []) %[[Q1]] ctrl %[[H]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
  // CHECK: %[[X2_T:.*]], %[[X2_C:.*]] = mqtopt.x( static [] mask []) %[[Q2]] ctrl %[[X1_C]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
  // CHECK: %[[I0:.*]] = "mqtopt.insertQubit"(%[[QREG3]], %[[X1_T]]) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
  // CHECK: %[[I1:.*]] = "mqtopt.insertQubit"(%[[I0]], %[[X2_T]]) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
  // CHECK: %[[I2:.*]] = "mqtopt.insertQubit"(%[[I1]], %[[X2_C]]) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
  // CHECK: "mqtopt.deallocQubitRegister"(%[[I2]]) : (!mqtopt.QubitRegister) -> ()

    %0 = quantum.alloc( 3) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit
    %true = arith.constant true
    %false = arith.constant false
    %out_qubits = quantum.custom "Hadamard"() %1 : !quantum.bit
    %true_0 = arith.constant true
    %false_1 = arith.constant false
    %out_qubits_2, %out_ctrl_qubits = quantum.custom "CNOT"() %2 ctrls(%out_qubits) ctrlvals(%true_0) : !quantum.bit ctrls !quantum.bit
    %true_3 = arith.constant true
    %false_4 = arith.constant false
    %out_qubits_5, %out_ctrl_qubits_6 = quantum.custom "CNOT"() %3 ctrls(%out_ctrl_qubits) ctrlvals(%true_3) : !quantum.bit ctrls !quantum.bit
    %4 = quantum.insert %0[ 0], %out_qubits_2 : !quantum.reg, !quantum.bit
    %5 = quantum.insert %4[ 1], %out_qubits_5 : !quantum.reg, !quantum.bit
    %6 = quantum.insert %5[ 2], %out_ctrl_qubits_6 : !quantum.reg, !quantum.bit
    quantum.dealloc %6 : !quantum.reg
    return
  }
}
