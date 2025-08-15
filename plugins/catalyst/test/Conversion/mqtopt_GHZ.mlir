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
  %0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister

  // CHECK: %[[Q0:.*]] = quantum.extract %[[QREG]][ 0] : !quantum.reg -> !quantum.bit
  // CHECK: %[[Q1:.*]] = quantum.extract %[[QREG]][ 1] : !quantum.reg -> !quantum.bit
  // CHECK: %[[Q2:.*]] = quantum.extract %[[QREG]][ 2] : !quantum.reg -> !quantum.bit
  %out_qureg, %out_qubit = "mqtopt.extractQubit"(%0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
  %out_qureg_0, %out_qubit_1 = "mqtopt.extractQubit"(%out_qureg) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
  %out_qureg_2, %out_qubit_3 = "mqtopt.extractQubit"(%out_qureg_0) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

  // CHECK: %[[H:.*]] = quantum.custom "Hadamard"() %[[Q0]] : !quantum.bit
  %1 = mqtopt.h() %out_qubit : !mqtopt.Qubit

  // CHECK: %[[CX1:.*]]:2 = quantum.custom "CNOT"() %[[H]], %[[Q1]] : !quantum.bit, !quantum.bit
  // CHECK: %[[CX2:.*]]:2 = quantum.custom "CNOT"() %[[CX1]]#0, %[[Q2]] : !quantum.bit, !quantum.bit
  %21, %20 = mqtopt.x() %out_qubit_1 ctrl %1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
  %31, %30 = mqtopt.x() %out_qubit_3 ctrl %20 : !mqtopt.Qubit ctrl !mqtopt.Qubit

  // CHECK: %[[R0:.*]] = quantum.insert %[[QREG]][ 0], %[[CX1]]#0 : !quantum.reg, !quantum.bit
  // CHECK: %[[R1:.*]] = quantum.insert %[[R0]][ 1], %[[CX2]]#0 : !quantum.reg, !quantum.bit
  // CHECK: %[[R2:.*]] = quantum.insert %[[R1]][ 2], %[[CX2]]#1 : !quantum.reg, !quantum.bit
  %4 = "mqtopt.insertQubit"(%out_qureg_2, %20) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
  %5 = "mqtopt.insertQubit"(%4, %30) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
  %6 = "mqtopt.insertQubit"(%5, %31) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

  "mqtopt.deallocQubitRegister"(%6) : (!mqtopt.QubitRegister) -> ()
  return
}
