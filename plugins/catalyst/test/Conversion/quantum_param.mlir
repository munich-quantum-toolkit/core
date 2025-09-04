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
  // CHECK-LABEL: func @testCatalystQuantumToMQTOptParam()
  func.func @testCatalystQuantumToMQTOptParam() {
    // CHECK: %{{.*}} = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
    // CHECK: %out_qreg, %out_qubit = "mqtopt.extractQubit"(%{{.*}}) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %out_qreg_{{.*}}, %out_qubit_{{.*}} = "mqtopt.extractQubit"(%out_qreg) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %cst = arith.constant {{.*}} : f64
    // CHECK: %out_qubits = mqtopt.rx(%cst static [] mask [false]) %out_qubit : !mqtopt.Qubit
    // CHECK: %out_qubits_{{.*}} = mqtopt.ry(%cst static [] mask [false]) %out_qubits : !mqtopt.Qubit
    // CHECK: %out_qubits_{{.*}} = mqtopt.rz(%cst static [] mask [false]) %out_qubits_{{.*}} : !mqtopt.Qubit
    // CHECK: %out_qubits_{{.*}} = mqtopt.p(%cst static [] mask [false]) %out_qubits_{{.*}} : !mqtopt.Qubit
    // CHECK: %out_qubits_{{.*}}, %pos_ctrl_out_qubits{{.*}} = mqtopt.rx(%cst static [] mask [false]) %out_qubit_{{.*}} ctrl %out_qubits_{{.*}} : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %out_qubits_{{.*}}, %pos_ctrl_out_qubits_{{.*}} = mqtopt.ry(%cst static [] mask [false]) %out_qubits_{{.*}} ctrl %pos_ctrl_out_qubits{{.*}} : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %{{.*}} = "mqtopt.insertQubit"(%out_qreg_{{.*}}, %out_qubits_{{.*}}) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %{{.*}} = "mqtopt.insertQubit"(%{{.*}}, %pos_ctrl_out_qubits_{{.*}}) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: "mqtopt.deallocQubitRegister"(%{{.*}}) : (!mqtopt.QubitRegister) -> ()

    %qreg = quantum.alloc( 2) : !quantum.reg
    %q0 = quantum.extract %qreg[ 0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %qreg[ 1] : !quantum.reg -> !quantum.bit

    %angle = arith.constant 1.5707963267948966 : f64
    %q0_rx = quantum.custom "RX"(%angle) %q0 : !quantum.bit
    %q0_ry = quantum.custom "RY"(%angle) %q0_rx : !quantum.bit
    %q0_rz = quantum.custom "RZ"(%angle) %q0_ry : !quantum.bit
    %q0_p = quantum.custom "PhaseShift"(%angle) %q0_rz : !quantum.bit

    %true = arith.constant true
    %q1_out, %q0_out = quantum.custom "RX"(%angle) %q1 ctrls(%q0_p) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %q1_out2, %q0_out2 = quantum.custom "RY"(%angle) %q1_out ctrls(%q0_out) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit

    %qreg1 = quantum.insert %qreg[ 0], %q1_out2 : !quantum.reg, !quantum.bit
    %qreg2 = quantum.insert %qreg1[ 1], %q0_out2 : !quantum.reg, !quantum.bit
    quantum.dealloc %qreg2 : !quantum.reg
    return
  }
}