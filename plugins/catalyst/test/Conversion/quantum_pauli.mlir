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
// RUN:   --catalyst-pipeline="builtin.module(catalystquantum-to-mqtopt)" \
// RUN:   %s | FileCheck %s

module {
  // CHECK-LABEL: func @testCatalystQuantumToMQTOptPauli()
  func.func @testCatalystQuantumToMQTOptPauli() {
    // CHECK: %{{.*}} = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister
    // CHECK: %out_qreg, %out_qubit = "mqtopt.extractQubit"(%{{.*}}) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %out_qreg_{{.*}}, %out_qubit_{{.*}} = "mqtopt.extractQubit"(%out_qreg) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %out_qreg_{{.*}}, %out_qubit_{{.*}} = "mqtopt.extractQubit"(%out_qreg_{{.*}}) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %out_qubits = mqtopt.h( static [] mask []) %out_qubit : !mqtopt.Qubit
    // CHECK: %out_qubits_{{.*}} = mqtopt.i( static [] mask []) %out_qubits : !mqtopt.Qubit
    // CHECK: %out_qubits_{{.*}} = mqtopt.x( static [] mask []) %out_qubits_{{.*}} : !mqtopt.Qubit
    // CHECK: %out_qubits_{{.*}} = mqtopt.y( static [] mask []) %out_qubits_{{.*}} : !mqtopt.Qubit
    // CHECK: %out_qubits_{{.*}} = mqtopt.z( static [] mask []) %out_qubits_{{.*}} : !mqtopt.Qubit
    // CHECK: %out_qubits_{{.*}}, %pos_ctrl_out_qubits{{.*}} = mqtopt.x( static [] mask []) %out_qubit_{{.*}} ctrl %out_qubits_{{.*}} : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %out_qubits_{{.*}}, %pos_ctrl_out_qubits_{{.*}} = mqtopt.y( static [] mask []) %out_qubits_{{.*}} ctrl %pos_ctrl_out_qubits{{.*}} : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %out_qubits_{{.*}}, %pos_ctrl_out_qubits_{{.*}} = mqtopt.z( static [] mask []) %out_qubits_{{.*}} ctrl %pos_ctrl_out_qubits_{{.*}} : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %{{.*}} = "mqtopt.insertQubit"(%out_qreg_{{.*}}, %out_qubits_{{.*}}) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %{{.*}} = "mqtopt.insertQubit"(%{{.*}}, %pos_ctrl_out_qubits_{{.*}}) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %{{.*}} = "mqtopt.insertQubit"(%{{.*}}, %out_qubit_{{.*}}) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: "mqtopt.deallocQubitRegister"(%{{.*}}) : (!mqtopt.QubitRegister) -> ()

    %qreg = quantum.alloc( 3) : !quantum.reg
    %q0 = quantum.extract %qreg[ 0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %qreg[ 1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %qreg[ 2] : !quantum.reg -> !quantum.bit

    %q0_h = quantum.custom "Hadamard"() %q0 : !quantum.bit
    %q0_i = quantum.custom "Identity"() %q0_h : !quantum.bit
    %q0_x = quantum.custom "PauliX"() %q0_i : !quantum.bit
    %q0_y = quantum.custom "PauliY"() %q0_x : !quantum.bit
    %q0_z = quantum.custom "PauliZ"() %q0_y : !quantum.bit

    %true = arith.constant true
    %q1_out, %q0_out = quantum.custom "PauliX"() %q1 ctrls(%q0_z) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %q1_out2, %q0_out2 = quantum.custom "PauliY"() %q1_out ctrls(%q0_out) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %q1_out3, %q0_out3 = quantum.custom "PauliZ"() %q1_out2 ctrls(%q0_out2) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit

    %qreg1 = quantum.insert %qreg[ 0], %q1_out3 : !quantum.reg, !quantum.bit
    %qreg2 = quantum.insert %qreg1[ 1], %q0_out3 : !quantum.reg, !quantum.bit
    %qreg3 = quantum.insert %qreg2[ 2], %q2 : !quantum.reg, !quantum.bit
    quantum.dealloc %qreg3 : !quantum.reg
    return
  }
}
