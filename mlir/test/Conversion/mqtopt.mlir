// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s --mqtopt-to-quantum | FileCheck %s

// CHECK-LABEL: func @bar()
func.func @bar() {
  // CHECK: %cst = arith.constant 3.000000e-01 : f64
  %cst = arith.constant 3.000000e-01 : f64

  // CHECK: %[[QREG:.*]] = quantum.alloc( 3) : !quantum.reg
  %0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister

  // CHECK: %[[Q0:.*]] = quantum.extract %[[QREG]][ 0] : !quantum.reg -> !quantum.bit
  // CHECK: %[[Q1:.*]] = quantum.extract %[[QREG]][ 1] : !quantum.reg -> !quantum.bit
  // CHECK: %[[Q2:.*]] = quantum.extract %[[QREG]][ 2] : !quantum.reg -> !quantum.bit
  %out_qureg, %out_qubit = "mqtopt.extractQubit"(%0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
  %out_qureg_0, %out_qubit_1 = "mqtopt.extractQubit"(%out_qureg) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
  %out_qureg_2, %out_qubit_3 = "mqtopt.extractQubit"(%out_qureg_0) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

  // CHECK: %[[H:.*]] = quantum.custom "Hadamard"() %[[Q0]] : !quantum.bit
  // CHECK: %[[X:.*]] = quantum.custom "PauliX"() %[[H]] : !quantum.bit
  // CHECK: %[[Y:.*]] = quantum.custom "PauliY"() %[[X]] : !quantum.bit
  // CHECK: %[[Z:.*]] = quantum.custom "PauliZ"() %[[Y]] : !quantum.bit
  %1 = mqtopt.H() %out_qubit : !mqtopt.Qubit
  %2 = mqtopt.x() %1 : !mqtopt.Qubit
  %3 = mqtopt.y() %2 : !mqtopt.Qubit
  %4 = mqtopt.z() %3 : !mqtopt.Qubit

  // CHECK: %[[T1:.*]] = arith.constant true
  // CHECK: %[[CNOT:.*]], %[[CTRL1:.*]] = quantum.custom "CNOT"() %[[Z]] ctrls(%[[Q1]]) ctrlvals(%[[T1]]) : !quantum.bit ctrls !quantum.bit
  // CHECK: %[[T2:.*]] = arith.constant true
  // CHECK: %[[CY:.*]], %[[CTRL2:.*]] = quantum.custom "CY"() %[[CNOT]] ctrls(%[[CTRL1]]) ctrlvals(%[[T2]]) : !quantum.bit ctrls !quantum.bit
  // CHECK: %[[T3:.*]] = arith.constant true
  // CHECK: %[[CZ:.*]], %[[CTRL3:.*]] = quantum.custom "CZ"() %[[CY]] ctrls(%[[CTRL2]]) ctrlvals(%[[T3]]) : !quantum.bit ctrls !quantum.bit
  // CHECK: %[[SW0:.*]]:2 = quantum.custom "SWAP"() %[[CTRL3]], %[[CZ]] : !quantum.bit, !quantum.bit
  // CHECK: %[[T4:.*]] = arith.constant true
  // CHECK: %[[TOF:.*]], %[[CT1:.*]]:2 = quantum.custom "Toffoli"() %[[SW0]]#0 ctrls(%[[Q2]], %[[SW0]]#1) ctrlvals(%[[T4]], %[[T4]]) : !quantum.bit ctrls !quantum.bit, !quantum.bit
  %5:2 = mqtopt.x() %4 ctrl %out_qubit_1 : !mqtopt.Qubit, !mqtopt.Qubit
  %6:2 = mqtopt.y() %5#0 ctrl %5#1 : !mqtopt.Qubit, !mqtopt.Qubit
  %7:2 = mqtopt.z() %6#0 ctrl %6#1 : !mqtopt.Qubit, !mqtopt.Qubit
  %8:2 = mqtopt.swap() %7#1, %7#0 : !mqtopt.Qubit, !mqtopt.Qubit
  %9:3 = mqtopt.x() %8#0 ctrl %out_qubit_3, %8#1 : !mqtopt.Qubit, !mqtopt.Qubit, !mqtopt.Qubit

  // CHECK: %[[RX:.*]] = quantum.custom "RX"(%cst) %[[TOF]] : !quantum.bit
  // CHECK: %[[RY:.*]] = quantum.custom "RY"(%cst) %[[RX]] : !quantum.bit
  // CHECK: %[[RZ:.*]] = quantum.custom "RZ"(%cst) %[[RY]] : !quantum.bit
  // CHECK: %[[PS:.*]] = quantum.custom "PhaseShift"(%cst) %[[RZ]] : !quantum.bit
  %10 = mqtopt.rx(%cst) %9#0 : !mqtopt.Qubit
  %11 = mqtopt.ry(%cst) %10 : !mqtopt.Qubit
  %12 = mqtopt.rz(%cst) %11 : !mqtopt.Qubit
  %13 = mqtopt.p(%cst) %12 : !mqtopt.Qubit

  // CHECK: %[[T5:.*]] = arith.constant true
  // CHECK: %[[CRX:.*]], %[[CR1:.*]] = quantum.custom "CRX"(%cst) %[[PS]] ctrls(%[[CT1]]#0) ctrlvals(%[[T5]]) : !quantum.bit ctrls !quantum.bit
  // CHECK: %[[T6:.*]] = arith.constant true
  // CHECK: %[[CRY:.*]], %[[CR2:.*]] = quantum.custom "CRY"(%cst) %[[CRX]] ctrls(%[[CR1]]) ctrlvals(%[[T6]]) : !quantum.bit ctrls !quantum.bit
  // CHECK: %[[T7:.*]] = arith.constant true
  // CHECK: %[[CRY2:.*]], %[[CR3:.*]] = quantum.custom "CRY"(%cst) %[[CRY]] ctrls(%[[CR2]]) ctrlvals(%[[T7]]) : !quantum.bit ctrls !quantum.bit
  // CHECK: %[[T8:.*]] = arith.constant true
  // CHECK: %[[CPS:.*]], %[[CR4:.*]] = quantum.custom "ControlledPhaseShift"(%cst) %[[CRY2]] ctrls(%[[CR3]]) ctrlvals(%[[T8]]) : !quantum.bit ctrls !quantum.bit
  %14:2 = mqtopt.rx(%cst) %13 ctrl %9#1 : !mqtopt.Qubit, !mqtopt.Qubit
  %15:2 = mqtopt.ry(%cst) %14#0 ctrl %14#1 : !mqtopt.Qubit, !mqtopt.Qubit
  %16:2 = mqtopt.ry(%cst) %15#0 ctrl %15#1 : !mqtopt.Qubit, !mqtopt.Qubit
  %17:2 = mqtopt.p(%cst) %16#0 ctrl %16#1 : !mqtopt.Qubit, !mqtopt.Qubit

  // CHECK: %[[MRES:.*]], %[[QMEAS:.*]] = quantum.measure %[[CT1]]#1 : i1, !quantum.bit
  %q_meas, %c0_0 = "mqtopt.measure"(%9#2) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

  // CHECK: %[[R1:.*]] = quantum.insert %[[QREG]][ 2], %[[QMEAS]] : !quantum.reg, !quantum.bit
  // CHECK: %[[R2:.*]] = quantum.insert %[[R1]][ 1], %[[CR4]] : !quantum.reg, !quantum.bit
  // CHECK: %[[R3:.*]] = quantum.insert %[[R2]][ 0], %[[CPS]] : !quantum.reg, !quantum.bit
  %18 = "mqtopt.insertQubit"(%out_qureg_2, %q_meas) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
  %19 = "mqtopt.insertQubit"(%18, %17#1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
  %20 = "mqtopt.insertQubit"(%19, %17#0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

  // CHECK: quantum.dealloc %[[R3]] : !quantum.reg
  "mqtopt.deallocQubitRegister"(%20) : (!mqtopt.QubitRegister) -> ()

  return
}
