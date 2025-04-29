// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s --mqtopt-to-catalystquantum | FileCheck %s

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
  %1 = mqtopt.h() %out_qubit : !mqtopt.Qubit
  %2 = mqtopt.x() %1 : !mqtopt.Qubit
  %3 = mqtopt.y() %2 : !mqtopt.Qubit
  %4 = mqtopt.z() %3 : !mqtopt.Qubit

  // CHECK: %[[CNOT:.*]]:2 = quantum.custom "CNOT"() %[[Z]], %[[Q1]] : !quantum.bit, !quantum.bit
  // CHECK: %[[CY:.*]]:2 = quantum.custom "CY"() %[[CNOT]]#0, %[[CNOT]]#1 : !quantum.bit, !quantum.bit
  // CHECK: %[[CZ:.*]]:2 = quantum.custom "CZ"() %[[CY]]#0, %[[CY]]#1 : !quantum.bit, !quantum.bit
  // CHECK: %[[SW0:.*]]:2 = quantum.custom "SWAP"() %[[CZ]]#1, %[[CZ]]#0 : !quantum.bit, !quantum.bit
  // CHECK: %[[TOF:.*]]:3 = quantum.custom "Toffoli"() %[[SW0]]#0, %[[Q2]], %[[SW0]]#1 : !quantum.bit, !quantum.bit, !quantum.bit
    %5, %6 = mqtopt.x() %4 ctrl %out_qubit_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %7, %8 = mqtopt.y() %5 ctrl %6 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %9, %10 = mqtopt.z() %7 ctrl %8 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %11, %12 = mqtopt.swap() %10, %9 : !mqtopt.Qubit, !mqtopt.Qubit
    %13, %14, %15 = mqtopt.x() %11 ctrl %out_qubit_3, %12 : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit

  // CHECK: %[[RX:.*]] = quantum.custom "RX"(%cst) %[[TOF]]#0 : !quantum.bit
  // CHECK: %[[RY:.*]] = quantum.custom "RY"(%cst) %[[RX]] : !quantum.bit
  // CHECK: %[[RZ:.*]] = quantum.custom "RZ"(%cst) %[[RY]] : !quantum.bit
  // CHECK: %[[PS:.*]] = quantum.custom "PhaseShift"(%cst) %[[RZ]] : !quantum.bit
  %16 = mqtopt.rx(%cst) %13 : !mqtopt.Qubit
  %17 = mqtopt.ry(%cst) %16 : !mqtopt.Qubit
  %18 = mqtopt.rz(%cst) %17 : !mqtopt.Qubit
  %19 = mqtopt.p(%cst) %18 : !mqtopt.Qubit

  // CHECK: %[[CRX:.*]]:2 = quantum.custom "CRX"(%cst) %[[PS]], %[[TOF]]#1 : !quantum.bit, !quantum.bit
  // CHECK: %[[CRY:.*]]:2 = quantum.custom "CRY"(%cst) %[[CRX]]#0, %[[CRX]]#1 : !quantum.bit, !quantum.bit
  // CHECK: %[[CRZ:.*]]:2 = quantum.custom "CRZ"(%cst) %[[CRY]]#0, %[[CRY]]#1 : !quantum.bit, !quantum.bit
  // CHECK: %[[CPS:.*]]:2 = quantum.custom "ControlledPhaseShift"(%cst) %[[CRZ]]#0, %[[CRZ]]#1 : !quantum.bit, !quantum.bit
  %200, %201 = mqtopt.rx(%cst) %19 ctrl %14 : !mqtopt.Qubit ctrl !mqtopt.Qubit
  %210, %211 = mqtopt.ry(%cst) %200 ctrl %201 : !mqtopt.Qubit ctrl !mqtopt.Qubit
  %220, %221 = mqtopt.rz(%cst) %210 ctrl %211 : !mqtopt.Qubit ctrl !mqtopt.Qubit
  %230, %231 = mqtopt.p(%cst) %220 ctrl %221 : !mqtopt.Qubit ctrl !mqtopt.Qubit

  // CHECK: %[[MRES:.*]], %[[QMEAS:.*]] = quantum.measure %[[TOF]]#2 : i1, !quantum.bit
  %q_meas, %c0_0 = "mqtopt.measure"(%15) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

  // CHECK: %[[R1:.*]] = quantum.insert %[[QREG]][ 2], %[[QMEAS]] : !quantum.reg, !quantum.bit
  // CHECK: %[[R2:.*]] = quantum.insert %[[R1]][ 1], %[[CPS]]#0 : !quantum.reg, !quantum.bit
  // CHECK: %[[R3:.*]] = quantum.insert %[[R2]][ 0], %[[CPS]]#1 : !quantum.reg, !quantum.bit
  %240 = "mqtopt.insertQubit"(%out_qureg_2, %q_meas) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
  %250 = "mqtopt.insertQubit"(%240, %230) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
  %260 = "mqtopt.insertQubit"(%250, %231) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister

  // CHECK: quantum.dealloc %[[R3]] : !quantum.reg
  "mqtopt.deallocQubitRegister"(%260) : (!mqtopt.QubitRegister) -> ()

  return
}
