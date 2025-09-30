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
// Clifford + T and controlled variants
// Groups: Allocation & extraction / Uncontrolled sequence / Controlled sequence / Reinsertion
// ============================================================================
module {
  // CHECK-LABEL: func.func @testCatalystQuantumToMQTOptCliffordT
  func.func @testCatalystQuantumToMQTOptCliffordT() {
    // --- Allocation & extraction ---------------------------------------------------------------
    // CHECK: %[[QREG:.*]] = memref.alloc() : memref<2x!mqtopt.Qubit>
    // CHECK: %[[CAST:.*]] = memref.cast %[[QREG]] : memref<2x!mqtopt.Qubit> to memref<?x!mqtopt.Qubit>
    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[Q0:.*]] = memref.load %[[QREG]][%[[C0]]] : memref<2x!mqtopt.Qubit>
    // CHECK: %[[C1:.*]] = arith.constant 1 : index
    // CHECK: %[[Q1:.*]] = memref.load %[[QREG]][%[[C1]]] : memref<2x!mqtopt.Qubit>

    // --- Uncontrolled sequence -----------------------------------------------------------------
    // CHECK: %[[H:.*]]   = mqtopt.h(static [] mask []) %[[Q0]] : !mqtopt.Qubit
    // CHECK: %[[V:.*]]   = mqtopt.sx(static [] mask []) %[[H]] : !mqtopt.Qubit
    // CHECK: %[[VDG:.*]] = mqtopt.sx(static [] mask []) %[[V]] : !mqtopt.Qubit
    // CHECK: %[[S:.*]]   = mqtopt.s(static [] mask []) %[[VDG]] : !mqtopt.Qubit
    // CHECK: %[[SDG:.*]] = mqtopt.s(static [] mask []) %[[S]] : !mqtopt.Qubit
    // CHECK: %[[T:.*]]   = mqtopt.t(static [] mask []) %[[SDG]] : !mqtopt.Qubit
    // CHECK: %[[TDG:.*]] = mqtopt.t(static [] mask []) %[[T]] : !mqtopt.Qubit

    // --- Controlled sequence -------------------------------------------------------------------
    // CHECK: %[[CH_T:.*]], %[[CH_C:.*]]   = mqtopt.h(static [] mask []) %[[TDG]] ctrl %[[Q1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CV_T:.*]], %[[CV_C:.*]]   = mqtopt.sx(static [] mask []) %[[CH_T]] ctrl %[[CH_C]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CVDG_T:.*]], %[[CVDG_C:.*]] = mqtopt.sx(static [] mask []) %[[CV_T]] ctrl %[[CV_C]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CS_T:.*]], %[[CS_C:.*]]   = mqtopt.s(static [] mask []) %[[CVDG_T]] ctrl %[[CVDG_C]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CSDG_T:.*]], %[[CSDG_C:.*]] = mqtopt.s(static [] mask []) %[[CS_T]] ctrl %[[CS_C]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CT_T:.*]], %[[CT_C:.*]]   = mqtopt.t(static [] mask []) %[[CSDG_T]] ctrl %[[CSDG_C]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[CTDG_T:.*]], %[[CTDG_C:.*]] = mqtopt.t(static [] mask []) %[[CT_T]] ctrl %[[CT_C]] : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // --- Reinsertion ---------------------------------------------------------------------------
    // CHECK: %[[C0_FINAL:.*]] = arith.constant 0 : index
    // CHECK: memref.store %[[CTDG_T]], %[[QREG]][%[[C0_FINAL]]] : memref<2x!mqtopt.Qubit>
    // CHECK: %[[C1_FINAL:.*]] = arith.constant 1 : index
    // CHECK: memref.store %[[CTDG_C]], %[[QREG]][%[[C1_FINAL]]] : memref<2x!mqtopt.Qubit>
    // CHECK: memref.dealloc %[[CAST]] : memref<?x!mqtopt.Qubit>

    // Prepare qubits
    %qreg = quantum.alloc( 2) : !quantum.reg
    %q0 = quantum.extract %qreg[ 0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %qreg[ 1] : !quantum.reg -> !quantum.bit

    // Non-controlled Clifford+T gates
    %q0_h = quantum.custom "Hadamard"() %q0 : !quantum.bit
    %q0_v = quantum.custom "SX"() %q0_h : !quantum.bit
    %q0_vdg = quantum.custom "SX"() %q0_v {adjoint} : !quantum.bit
    %q0_s = quantum.custom "S"() %q0_vdg : !quantum.bit
    %q0_sdg = quantum.custom "S"() %q0_s {adjoint} : !quantum.bit
    %q0_t = quantum.custom "T"() %q0_sdg : !quantum.bit
    %q0_tdg = quantum.custom "T"() %q0_t {adjoint} : !quantum.bit

    // Controlled Clifford+T gates
    %true = arith.constant true
    %q0_ch, %q1_ch = quantum.custom "Hadamard"() %q0_tdg ctrls(%q1) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %q0_cv, %q1_cv = quantum.custom "SX"() %q0_ch ctrls(%q1_ch) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %q0_cvdg, %q1_cvdg = quantum.custom "SX"() %q0_cv {adjoint} ctrls(%q1_cv) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %q0_cs, %q1_cs = quantum.custom "S"() %q0_cvdg ctrls(%q1_cvdg) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %q0_csdg, %q1_csdg = quantum.custom "S"() %q0_cs {adjoint} ctrls(%q1_cs) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %q0_ct, %q1_ct = quantum.custom "T"() %q0_csdg ctrls(%q1_csdg) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit
    %q0_ctdg, %q1_ctdg = quantum.custom "T"() %q0_ct {adjoint} ctrls(%q1_ct) ctrlvals(%true) : !quantum.bit ctrls !quantum.bit

    // Release qubits
    %qreg1 = quantum.insert %qreg[ 0], %q0_ctdg : !quantum.reg, !quantum.bit
    %qreg2 = quantum.insert %qreg1[ 1], %q1_ctdg : !quantum.reg, !quantum.bit
    quantum.dealloc %qreg2 : !quantum.reg
    return
  }
}
