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


// ============================================================================
// IsingXY (XX+YY), IsingZZ (via RZX), DCX and controlled variants
// Groups: Constants / Allocation & extraction / Uncontrolled chain / Controlled chain / Reinsertion
// ============================================================================
module {
  // CHECK-LABEL: func.func @testMQTOptToCatalystQuantumXY_RZX_DCX
  func.func @testMQTOptToCatalystQuantumXY_RZX_DCX() {
    // --- Allocation & extraction ---------------------------------------------------------------
    // CHECK: %cst = arith.constant 3.000000e-01 : f64
    // CHECK: %[[QREG:.*]] = quantum.alloc( 3) : !quantum.reg
    // CHECK: %[[Q0_0:.*]] = quantum.extract %[[QREG]][ 0] : !quantum.reg -> !quantum.bit
    // CHECK: %[[Q1_0:.*]] = quantum.extract %[[QREG]][ 1] : !quantum.reg -> !quantum.bit
    // CHECK: %[[Q2_0:.*]] = quantum.extract %[[QREG]][ 2] : !quantum.reg -> !quantum.bit

    // --- Uncontrolled -------------------------------------------------------------------
    // CHECK: %[[XY_P:.*]]:2 = quantum.custom "IsingXY"(%cst, %cst) %[[Q0_0]], %[[Q1_0]] : !quantum.bit, !quantum.bit

    // CHECK: %[[XX_P:.*]]:2 = quantum.custom "IsingXX"(%cst) %[[XY_P]]#0, %[[XY_P]]#1 : !quantum.bit, !quantum.bit
    // CHECK: %[[YY_P:.*]]:2 = quantum.custom "IsingYY"(%cst) %[[XX_P]]#0, %[[XX_P]]#1 : !quantum.bit, !quantum.bit
    // CHECK: %[[ZZ_P1:.*]]:2 = quantum.custom "IsingZZ"(%cst) %[[YY_P]]#0, %[[YY_P]]#1 : !quantum.bit, !quantum.bit

    // CHECK: %[[H1:.*]] = quantum.custom "Hadamard"() %[[ZZ_P1]]#1 : !quantum.bit 
    // CHECK: %[[ZZ_P2:.*]]:2 = quantum.custom "IsingZZ"(%cst) %[[ZZ_P1]]#0, %[[H1]] : !quantum.bit, !quantum.bit
    // CHECK: %[[H2:.*]] = quantum.custom "Hadamard"() %[[ZZ_P2]]#1 : !quantum.bit

    // --- Controlled ---------------------------------------------------------------------
    // CHECK: %[[XY_C:.*]]:2, %[[CTRL1:.*]] = quantum.custom "IsingXY"(%cst, %cst) %[[ZZ_P2]]#0, %[[H2]] ctrls(%[[Q2_0]]) ctrlvals(%true{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit

    // CHECK: %[[XX_C:.*]]:2, %[[CTRL2:.*]] = quantum.custom "IsingXX"(%cst) %[[XY_C]]#0, %[[XY_C]]#1 ctrls(%[[CTRL1]]) ctrlvals(%true{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[YY_C:.*]]:2, %[[CTRL3:.*]] = quantum.custom "IsingYY"(%cst) %[[XX_C]]#0, %[[XX_C]]#1 ctrls(%[[CTRL2]]) ctrlvals(%true{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[ZZ_C1:.*]]:2, %[[CTRL4:.*]] = quantum.custom "IsingZZ"(%cst) %[[YY_C]]#0, %[[YY_C]]#1 ctrls(%[[CTRL3]]) ctrlvals(%true{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit

    // CHECK: %[[H1:.*]], %[[CTRL5:.*]] = quantum.custom "Hadamard"() %[[ZZ_C1]]#1 ctrls(%[[CTRL4]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CZZ_P2:.*]]:2, %[[CTRL6:.*]] = quantum.custom "IsingZZ"(%cst) %[[ZZ_C1]]#0, %[[H1]] ctrls(%[[CTRL5]]) ctrlvals(%true{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[H2:.*]], %[[CTRL7:.*]] = quantum.custom "Hadamard"() %[[CZZ_P2]]#1 ctrls(%[[CTRL6]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit

    // --- Reinsertion ---------------------------------------------------------------------------
    // CHECK: %[[R1:.*]] = quantum.insert %[[QREG]][ 0], %[[CZZ_P2]]#0 : !quantum.reg, !quantum.bit
    // CHECK: %[[R2:.*]] = quantum.insert %[[R1]][ 1], %[[H2]] : !quantum.reg, !quantum.bit
    // CHECK: %[[R3:.*]] = quantum.insert %[[R2]][ 2], %[[CTRL7]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc %[[R3]] : !quantum.reg

    // Prepare
    %cst = arith.constant 3.000000e-01 : f64
    %r0_0 = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister
    %r0_1, %q0_0 = "mqtopt.extractQubit"(%r0_0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    %r0_2, %q1_0 = "mqtopt.extractQubit"(%r0_1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    %r0_3, %q2_0 = "mqtopt.extractQubit"(%r0_2) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // Uncontrolled
    %q0_1, %q1_1 = mqtopt.xx_plus_yy(%cst, %cst) %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
    %q0_3, %q1_3 = mqtopt.rxx(%cst) %q0_1, %q1_1 : !mqtopt.Qubit, !mqtopt.Qubit
    %q0_4, %q1_4 = mqtopt.ryy(%cst) %q0_3, %q1_3 : !mqtopt.Qubit, !mqtopt.Qubit
    %q0_5, %q1_5 = mqtopt.rzz(%cst) %q0_4, %q1_4 : !mqtopt.Qubit, !mqtopt.Qubit
    %q0_6, %q1_6 = mqtopt.rzx(%cst) %q0_5, %q1_5 : !mqtopt.Qubit, !mqtopt.Qubit

    // Controlled 
    %q0_7,  %q1_7,  %q2_1 = mqtopt.xx_plus_yy(%cst, %cst) %q0_6, %q1_6 ctrl %q2_0 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_9,  %q1_9,  %q2_3 = mqtopt.rxx(%cst) %q0_7, %q1_7 ctrl %q2_1 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_10, %q1_10, %q2_4 = mqtopt.ryy(%cst) %q0_9, %q1_9 ctrl %q2_3 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_11, %q1_11, %q2_5 = mqtopt.rzz(%cst) %q0_10, %q1_10 ctrl %q2_4 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_12, %q1_12, %q2_6 = mqtopt.rzx(%cst) %q0_11, %q1_11 ctrl %q2_5 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit    

    // Release
    %r0_4 = "mqtopt.insertQubit"(%r0_3, %q0_12) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    %r0_5 = "mqtopt.insertQubit"(%r0_4, %q1_12) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    %r0_6 = "mqtopt.insertQubit"(%r0_5, %q2_6) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    "mqtopt.deallocQubitRegister"(%r0_6) : (!mqtopt.QubitRegister) -> ()
    return
  }
}