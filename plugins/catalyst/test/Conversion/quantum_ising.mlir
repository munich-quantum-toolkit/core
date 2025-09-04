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


// ============================================================================
// Ising-type gates and controlled variants
// Groups: Constants / Allocation & extraction / Uncontrolled chain / Controlled chain / Reinsertion
// ============================================================================
module {
  // CHECK-LABEL: func.func @testCatalystQuantumToMQTOptIsing
  func.func @testCatalystQuantumToMQTOptIsing() {
    // --- Constants & Allocation & extraction ---------------------------------------------------
    // CHECK: %cst = arith.constant 3.000000e-01 : f64
    // CHECK: %[[QREG:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 3 : i64}> : () -> !mqtopt.QubitRegister
    // CHECK: %[[QR1:.*]], %[[Q0:.*]] = "mqtopt.extractQubit"(%[[QREG]]) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[QR2:.*]], %[[Q1:.*]] = "mqtopt.extractQubit"(%[[QR1]]) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[QR3:.*]], %[[Q2:.*]] = "mqtopt.extractQubit"(%[[QR2]]) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // --- Uncontrolled ---------------------------------------------------------------------------
    // CHECK: %[[XY:.*]]:2 = mqtopt.xx_plus_yy(%cst, %cst static [] mask [false, false]) %[[Q0]], %[[Q1]] : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[XX:.*]]:2 = mqtopt.xx_plus_yy(%cst, %cst static [] mask [false, false]) %[[XY]]#0, %[[XY]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[YY:.*]]:2 = mqtopt.xx_plus_yy(%cst, %cst static [] mask [false, false]) %[[XX]]#0, %[[XX]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[ZZ:.*]]:2 = mqtopt.xx_plus_yy(%cst, %cst static [] mask [false, false]) %[[YY]]#0, %[[YY]]#1 : !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[ZX:.*]]:2 = mqtopt.xx_plus_yy(%cst, %cst static [] mask [false, false]) %[[ZZ]]#0, %[[ZZ]]#1 : !mqtopt.Qubit, !mqtopt.Qubit

    // --- Controlled -----------------------------------------------------------------------------
    // CHECK: %[[CXY_T:.*]]:2, %[[CXY_C:.*]] = mqtopt.xx_plus_yy(%cst, %cst static [] mask [false, false]) %[[ZX]]#0, %[[ZX]]#1 ctrl %[[Q2]] : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CXX_T:.*]]:2, %[[CXX_C:.*]] = mqtopt.xx_plus_yy(%cst, %cst static [] mask [false, false]) %[[CXY_T]]#0, %[[CXY_T]]#1 ctrl %[[CXY_C]] : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CYY_T:.*]]:2, %[[CYY_C:.*]] = mqtopt.xx_plus_yy(%cst, %cst static [] mask [false, false]) %[[CXX_T]]#0, %[[CXX_T]]#1 ctrl %[[CXX_C]] : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CZZ_T:.*]]:2, %[[CZZ_C:.*]] = mqtopt.xx_plus_yy(%cst, %cst static [] mask [false, false]) %[[CYY_T]]#0, %[[CYY_T]]#1 ctrl %[[CYY_C]] : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CZX_T:.*]]:2, %[[CZX_C:.*]] = mqtopt.xx_plus_yy(%cst, %cst static [] mask [false, false]) %[[CZZ_T]]#0, %[[CZZ_T]]#1 ctrl %[[CZZ_C]] : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit

    // --- Reinsertion ---------------------------------------------------------------------------
    // CHECK: %[[R1:.*]] = "mqtopt.insertQubit"(%[[QR3]], %[[CZX_T]]#0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %[[R2:.*]] = "mqtopt.insertQubit"(%[[R1]], %[[CZX_T]]#1) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %[[R3:.*]] = "mqtopt.insertQubit"(%[[R2]], %[[CZX_C]]) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: "mqtopt.deallocQubitRegister"(%[[R3]]) : (!mqtopt.QubitRegister) -> ()

    // Prepare qubits
    %angle = arith.constant 3.000000e-01 : f64
    %qreg = quantum.alloc( 3) : !quantum.reg
    %q0 = quantum.extract %qreg[ 0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %qreg[ 1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %qreg[ 2] : !quantum.reg -> !quantum.bit

    // Uncontrolled Ising gates
    %q0_xy, %q1_xy = quantum.custom "IsingXY"(%angle, %angle) %q0, %q1 : !quantum.bit, !quantum.bit
    %q0_xx, %q1_xx = quantum.custom "IsingXY"(%angle, %angle) %q0_xy, %q1_xy : !quantum.bit, !quantum.bit
    %q0_yy, %q1_yy = quantum.custom "IsingXY"(%angle, %angle) %q0_xx, %q1_xx : !quantum.bit, !quantum.bit
    %q0_zz, %q1_zz = quantum.custom "IsingXY"(%angle, %angle) %q0_yy, %q1_yy : !quantum.bit, !quantum.bit
    %q0_zx, %q1_zx = quantum.custom "IsingXY"(%angle, %angle) %q0_zz, %q1_zz : !quantum.bit, !quantum.bit

    // Controlled Ising gates
    %true = arith.constant true
    %q0_cxy, %q1_cxy, %q2_cxy = quantum.custom "IsingXY"(%angle, %angle) %q0_zx, %q1_zx ctrls(%q2) ctrlvals(%true) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    %q0_cxx, %q1_cxx, %q2_cxx = quantum.custom "IsingXY"(%angle, %angle) %q0_cxy, %q1_cxy ctrls(%q2_cxy) ctrlvals(%true) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    %q0_cyy, %q1_cyy, %q2_cyy = quantum.custom "IsingXY"(%angle, %angle) %q0_cxx, %q1_cxx ctrls(%q2_cxx) ctrlvals(%true) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    %q0_czz, %q1_czz, %q2_czz = quantum.custom "IsingXY"(%angle, %angle) %q0_cyy, %q1_cyy ctrls(%q2_cyy) ctrlvals(%true) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    %q0_czx, %q1_czx, %q2_czx = quantum.custom "IsingXY"(%angle, %angle) %q0_czz, %q1_czz ctrls(%q2_czz) ctrlvals(%true) : !quantum.bit, !quantum.bit ctrls !quantum.bit

    // Release qubits
    %qreg1 = quantum.insert %qreg[ 0], %q0_czx : !quantum.reg, !quantum.bit
    %qreg2 = quantum.insert %qreg1[ 1], %q1_czx : !quantum.reg, !quantum.bit
    %qreg3 = quantum.insert %qreg2[ 2], %q2_czx : !quantum.reg, !quantum.bit
    quantum.dealloc %qreg3 : !quantum.reg
    return
  }
}
