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
// RUN:   --catalyst-pipeline="builtin.module(mqtopt-to-catalystquantum)" \
// RUN:   %s | FileCheck %s


// CHECK-LABEL: func @bar()
func.func @bar() {
  // CHECK: %[[QREG:.*]] = quantum.alloc( 3) : !quantum.reg
  // CHECK: %[[Q0:.*]] = quantum.extract %[[QREG]][ 0] : !quantum.reg -> !quantum.bit
  // CHECK: %[[Q1:.*]] = quantum.extract %[[QREG]][ 1] : !quantum.reg -> !quantum.bit
  // CHECK: %[[Q2:.*]] = quantum.extract %[[QREG]][ 2] : !quantum.reg -> !quantum.bit
  // CHECK: %[[H_OUT:.*]] = quantum.custom "Hadamard"() %[[Q0]] : !quantum.bit
  // CHECK: %[[TV1:.*]] = arith.constant true
  // CHECK: %[[CNOT1_TGT:.*]], %[[CNOT1_CTRL:.*]] = quantum.custom "CNOT"() %[[Q1]] ctrls(%[[H_OUT]]) ctrlvals(%[[TV1]]) : !quantum.bit ctrls !quantum.bit
  // CHECK: %[[TV2:.*]] = arith.constant true
  // CHECK: %[[CNOT2_TGT:.*]], %[[CNOT2_CTRL:.*]] = quantum.custom "CNOT"() %[[Q2]] ctrls(%[[CNOT1_CTRL]]) ctrlvals(%[[TV2]]) : !quantum.bit ctrls !quantum.bit
  // CHECK: %[[INS0:.*]] = quantum.insert %[[QREG]][ 0], %[[CNOT1_TGT]] : !quantum.reg, !quantum.bit
  // CHECK: %[[INS1:.*]] = quantum.insert %[[INS0]][ 1], %[[CNOT2_TGT]] : !quantum.reg, !quantum.bit
  // CHECK: %[[INS2:.*]] = quantum.insert %[[INS1]][ 2], %[[CNOT2_CTRL]] : !quantum.reg, !quantum.bit
  // CHECK: quantum.dealloc %[[INS2]] : !quantum.reg

    %0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister
    %out_qreg, %out_qubit = "mqtopt.extractQubit"(%0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    %out_qreg_0, %out_qubit_1 = "mqtopt.extractQubit"(%out_qreg) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    %out_qreg_2, %out_qubit_3 = "mqtopt.extractQubit"(%out_qreg_0) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    %out_qubits = mqtopt.h( static [] mask []) %out_qubit : !mqtopt.Qubit
    %out_qubits_6, %pos_ctrl_out_qubits = mqtopt.x( static [] mask []) %out_qubit_1 ctrl %out_qubits : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %out_qubits_9, %pos_ctrl_out_qubits_10 = mqtopt.x( static [] mask []) %out_qubit_3 ctrl %pos_ctrl_out_qubits : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %1 = "mqtopt.insertQubit"(%out_qreg_2, %out_qubits_6) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    %2 = "mqtopt.insertQubit"(%1, %out_qubits_9) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    %3 = "mqtopt.insertQubit"(%2, %pos_ctrl_out_qubits_10) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    "mqtopt.deallocQubitRegister"(%3) : (!mqtopt.QubitRegister) -> ()
    return
}
