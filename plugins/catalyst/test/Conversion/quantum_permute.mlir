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
  // CHECK-LABEL: func @testCatalystQuantumToMQTOptPermute()
  func.func @testCatalystQuantumToMQTOptPermute() {
    // CHECK: %{{.*}} = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister
    // CHECK: %out_qreg, %out_qubit = "mqtopt.extractQubit"(%{{.*}}) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %out_qreg_{{.*}}, %out_qubit_{{.*}} = "mqtopt.extractQubit"(%out_qreg) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %out_qreg_{{.*}}, %out_qubit_{{.*}} = "mqtopt.extractQubit"(%out_qreg_{{.*}}) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %out_qubits:2 = mqtopt.swap( static [] mask []) %out_qubit, %out_qubit_{{.*}} : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %out_qubits_{{.*}}:2 = mqtopt.iswap( static [] mask []) %out_qubits#0, %out_qubits#1 : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %out_qubits_{{.*}}:2 = mqtopt.ecr( static [] mask []) %out_qubits_{{.*}}#0, %out_qubits_{{.*}}#1 : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %out_qubits_{{.*}}:2, %pos_ctrl_out_qubits{{.*}} = mqtopt.swap( static [] mask []) %out_qubits_{{.*}}#0, %out_qubits_{{.*}}#1 ctrl %out_qubit_{{.*}} : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %{{.*}} = "mqtopt.insertQubit"(%out_qreg_{{.*}}, %out_qubits_{{.*}}#0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %{{.*}} = "mqtopt.insertQubit"(%{{.*}}, %out_qubits_{{.*}}#1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %{{.*}} = "mqtopt.insertQubit"(%{{.*}}, %pos_ctrl_out_qubits{{.*}}) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: "mqtopt.deallocQubitRegister"(%{{.*}}) : (!mqtopt.QubitRegister) -> ()

    %qreg = quantum.alloc( 3) : !quantum.reg
    %q0 = quantum.extract %qreg[ 0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %qreg[ 1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %qreg[ 2] : !quantum.reg -> !quantum.bit

    %q0_out, %q1_out = quantum.custom "SWAP"() %q0, %q1 : !quantum.bit, !quantum.bit
    %q0_out2, %q1_out2 = quantum.custom "ISWAP"() %q0_out, %q1_out : !quantum.bit, !quantum.bit
    %q0_out3, %q1_out3 = quantum.custom "ECR"() %q0_out2, %q1_out2 : !quantum.bit, !quantum.bit

    %true = arith.constant true
    %q0_out4, %q1_out4, %q2_out = quantum.custom "SWAP"() %q0_out3, %q1_out3 ctrls(%q2) ctrlvals(%true) : !quantum.bit, !quantum.bit ctrls !quantum.bit

    %qreg1 = quantum.insert %qreg[ 0], %q0_out4 : !quantum.reg, !quantum.bit
    %qreg2 = quantum.insert %qreg1[ 1], %q1_out4 : !quantum.reg, !quantum.bit
    %qreg3 = quantum.insert %qreg2[ 2], %q2_out : !quantum.reg, !quantum.bit
    quantum.dealloc %qreg3 : !quantum.reg
    return
  }
}