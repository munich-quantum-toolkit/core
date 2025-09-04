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
  // CHECK-LABEL: func @testCatalystQuantumToMQTOptIsing()
  func.func @testCatalystQuantumToMQTOptIsing() {
    // CHECK: %{{.*}} = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister
    // CHECK: %out_qreg, %out_qubit = "mqtopt.extractQubit"(%{{.*}}) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %out_qreg_{{.*}}, %out_qubit_{{.*}} = "mqtopt.extractQubit"(%out_qreg) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %out_qreg_{{.*}}, %out_qubit_{{.*}} = "mqtopt.extractQubit"(%out_qreg_{{.*}}) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %cst{{.*}} = arith.constant {{.*}} : f64
    // CHECK: %cst_{{.*}} = arith.constant {{.*}} : f64
    // CHECK: %cst_{{.*}} = arith.constant {{.*}} : f64
    // CHECK: %out_qubits:2 = mqtopt.xx_plus_yy(%cst{{.*}}, %cst_{{.*}} static [] mask [false, false]) %out_qubit, %out_qubit_{{.*}} : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %out_qubits_{{.*}}:2 = mqtopt.xx_plus_yy(%cst_{{.*}}, %cst_{{.*}} static [] mask [false, false]) %out_qubits#0, %out_qubits#1 : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %out_qubits_{{.*}}:2 = mqtopt.xx_plus_yy(%cst_{{.*}}, %cst{{.*}} static [] mask [false, false]) %out_qubits_{{.*}}#0, %out_qubits_{{.*}}#1 : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %out_qubits_{{.*}}:2 = mqtopt.xx_plus_yy(%cst{{.*}}, %cst_{{.*}} static [] mask [false, false]) %out_qubits_{{.*}}#0, %out_qubits_{{.*}}#1 : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %{{.*}} = "mqtopt.insertQubit"(%out_qreg_{{.*}}, %out_qubits_{{.*}}#0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %{{.*}} = "mqtopt.insertQubit"(%{{.*}}, %out_qubits_{{.*}}#1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %{{.*}} = "mqtopt.insertQubit"(%{{.*}}, %out_qubit_{{.*}}) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: "mqtopt.deallocQubitRegister"(%{{.*}}) : (!mqtopt.QubitRegister) -> ()

    %qreg = quantum.alloc( 3) : !quantum.reg
    %q0 = quantum.extract %qreg[ 0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %qreg[ 1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %qreg[ 2] : !quantum.reg -> !quantum.bit

    %angle1 = arith.constant 1.5707963267948966 : f64
    %angle2 = arith.constant 2.356194490192345 : f64  
    %angle3 = arith.constant 3.141592653589793 : f64

    %q0_out, %q1_out = quantum.custom "IsingXY"(%angle1, %angle2) %q0, %q1 : !quantum.bit, !quantum.bit
    %q0_out2, %q1_out2 = quantum.custom "IsingXY"(%angle2, %angle3) %q0_out, %q1_out : !quantum.bit, !quantum.bit
    %q0_out3, %q1_out3 = quantum.custom "IsingXY"(%angle3, %angle1) %q0_out2, %q1_out2 : !quantum.bit, !quantum.bit
    %q0_out4, %q1_out4 = quantum.custom "IsingXY"(%angle1, %angle2) %q0_out3, %q1_out3 : !quantum.bit, !quantum.bit

    %qreg1 = quantum.insert %qreg[ 0], %q0_out4 : !quantum.reg, !quantum.bit
    %qreg2 = quantum.insert %qreg1[ 1], %q1_out4 : !quantum.reg, !quantum.bit
    %qreg3 = quantum.insert %qreg2[ 2], %q2 : !quantum.reg, !quantum.bit
    quantum.dealloc %qreg3 : !quantum.reg
    return
  }
}