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
// RUN:   --catalyst-pipeline="builtin.module(mqtopt-to-catalystquantum)" \
// RUN:   %s | FileCheck %s


// ============================================================================
// Clifford + T and controlled variants
// Groups: Allocation & extraction / Uncontrolled / Controlled / Reinsertion
// ============================================================================
module {
  // CHECK-LABEL: func.func @testMQTOptToCatalystQuantumCliffordT
  func.func @testMQTOptToCatalystQuantumCliffordT() {
    // --- Allocation & extraction ---------------------------------------------------------------
    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[C1:.*]] = arith.constant 1 : index
    // CHECK: %[[C2:.*]] = arith.constant 2 : index
    // CHECK: %[[C3_I64:.*]] = arith.constant 3 : i64
    // CHECK: %[[QREG:.*]] = quantum.alloc(%[[C3_I64]]) : !quantum.reg
    // CHECK: %[[IDX0:.*]] = arith.index_cast %[[C0]] : index to i64
    // CHECK: %[[Q0:.*]] = quantum.extract %[[QREG]][%[[IDX0]]] : !quantum.reg -> !quantum.bit
    // CHECK: %[[IDX1:.*]] = arith.index_cast %[[C1]] : index to i64
    // CHECK: %[[Q1:.*]] = quantum.extract %[[QREG]][%[[IDX1]]] : !quantum.reg -> !quantum.bit
    // CHECK: %[[IDX2:.*]] = arith.index_cast %[[C2]] : index to i64
    // CHECK: %[[Q2:.*]] = quantum.extract %[[QREG]][%[[IDX2]]] : !quantum.reg -> !quantum.bit

    // --- Uncontrolled Clifford+T gates ---------------------------------------------------------
    // CHECK: %[[I:.*]]   = quantum.custom "Identity"() %[[Q0]] : !quantum.bit
    // CHECK: %[[H:.*]]   = quantum.custom "Hadamard"() %[[I]] : !quantum.bit

    // V gate gets decomposed into a sequence of single-qubit rotations
    // CHECK: %[[CST:.*]] = arith.constant {{.*}} : f64
    // CHECK: %[[RZ1:.*]] = quantum.custom "RZ"(%[[CST]]) %[[H]] : !quantum.bit
    // CHECK: %[[RY1:.*]] = quantum.custom "RY"(%[[CST]]) %[[RZ1]] : !quantum.bit
    // CHECK: %[[RZ2:.*]] = quantum.custom "RZ"(%[[CST]]) %[[RY1]] adj : !quantum.bit


    // CHECK: %[[NEG_CST:.*]] = arith.constant -{{.*}} : f64
    // CHECK: %[[RZ3:.*]] = quantum.custom "RZ"(%[[NEG_CST]]) %[[RZ2]] adj : !quantum.bit
    // CHECK: %[[RY2:.*]] = quantum.custom "RY"(%[[NEG_CST]]) %[[RZ3]] : !quantum.bit
    // CHECK: %[[RZ4:.*]] = quantum.custom "RZ"(%[[NEG_CST]]) %[[RY2]] : !quantum.bit

    // CHECK: %[[S:.*]]   = quantum.custom "S"() %[[RZ4]] : !quantum.bit
    // CHECK: %[[SDG:.*]] = quantum.custom "S"() %[[S]] adj : !quantum.bit
    // CHECK: %[[T:.*]]   = quantum.custom "T"() %[[SDG]] : !quantum.bit
    // CHECK: %[[TDG:.*]] = quantum.custom "T"() %[[T]] adj : !quantum.bit

    // --- Peres gate decomposition -------------------------------------------------------------------
    // CHECK: %[[PERES_CNOT:.*]]:2 = quantum.custom "CNOT"() %[[TDG]], %[[Q1]] : !quantum.bit, !quantum.bit
    // CHECK: %[[PERES_X:.*]] = quantum.custom "PauliX"() %[[PERES_CNOT]]#0 : !quantum.bit

    // --- Peresdg gate decomposition -------------------------------------------------------------------
    // CHECK: %[[PERESDG_X:.*]] = quantum.custom "PauliX"() %[[PERES_X]] : !quantum.bit
    // CHECK: %[[PERESDG_CNOT:.*]]:2 = quantum.custom "CNOT"() %[[PERESDG_X]], %[[PERES_CNOT]]#1 : !quantum.bit, !quantum.bit

    // --- Controlled Hadamard -------------------------------------------------------------------
    // CHECK: %[[CH_T:.*]], %[[CH_C:.*]] = quantum.custom "Hadamard"() %[[PERESDG_CNOT]]#0 ctrls(%[[PERESDG_CNOT]]#1) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit

    // --- Controlled V gate decomposition (controlled RZ-RY-RZ sequence) -------------------------------------------------------------------
    // CHECK: %[[CST:.*]] = arith.constant {{.*}} : f64
    // CHECK: %[[CRZ1_T:.*]], %[[CRZ1_C:.*]] = quantum.custom "RZ"(%[[CST]]) %[[CH_T]] ctrls(%[[CH_C]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CRY1_T:.*]], %[[CRY1_C:.*]] = quantum.custom "RY"(%[[CST]]) %[[CRZ1_T]] ctrls(%[[CRZ1_C]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CRZ2_T:.*]], %[[CRZ2_C:.*]] = quantum.custom "RZ"(%[[CST]]) %[[CRY1_T]] adj ctrls(%[[CRY1_C]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit

    // --- Controlled Vdg gate decomposition -------------------------------------------------------------------
    // CHECK: %[[NEG_CST:.*]] = arith.constant -{{.*}} : f64
    // CHECK: %[[CRZ3_T:.*]], %[[CRZ3_C:.*]] = quantum.custom "RZ"(%[[NEG_CST]]) %[[CRZ2_T]] adj ctrls(%[[CRZ2_C]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CRY2_T:.*]], %[[CRY2_C:.*]] = quantum.custom "RY"(%[[NEG_CST]]) %[[CRZ3_T]] ctrls(%[[CRZ3_C]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CRZ4_T:.*]], %[[CRZ4_C:.*]] = quantum.custom "RZ"(%[[NEG_CST]]) %[[CRY2_T]] ctrls(%[[CRY2_C]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit

    // --- Controlled S and Sdg -------------------------------------------------------------------
    // CHECK: %[[CS_T:.*]], %[[CS_C:.*]] = quantum.custom "S"() %[[CRZ4_T]] ctrls(%[[CRZ4_C]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CSD_T:.*]], %[[CSD_C:.*]] = quantum.custom "S"() %[[CS_T]] adj ctrls(%[[CS_C]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit

    // --- Controlled Peres gate -------------------------------------------------------------------
    // CHECK: %[[CPERES_CNOT:.*]]:2, %[[CPERES_CTRL:.*]] = quantum.custom "CNOT"() %[[CSD_T]], %[[CSD_C]] ctrls(%[[Q2]]) ctrlvals(%true{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CPERES_X:.*]], %[[CPERES_X_CTRL:.*]] = quantum.custom "PauliX"() %[[CPERES_CNOT]]#0 ctrls(%[[CPERES_CTRL]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit

    // --- Controlled Peresdg gate -------------------------------------------------------------------
    // CHECK: %[[CPERESDG_X:.*]], %[[CPERESDG_X_CTRL:.*]] = quantum.custom "PauliX"() %[[CPERES_X]] ctrls(%[[CPERES_X_CTRL]]) ctrlvals(%true{{.*}}) : !quantum.bit ctrls !quantum.bit
    // CHECK: %[[CPERESDG_CNOT:.*]]:2, %[[CPERESDG_CTRL:.*]] = quantum.custom "CNOT"() %[[CPERESDG_X]], %[[CPERES_CNOT]]#1 ctrls(%[[CPERESDG_X_CTRL]]) ctrlvals(%true{{.*}}) : !quantum.bit, !quantum.bit ctrls !quantum.bit

    // --- Reinsertion ---------------------------------------------------------------------------
    // CHECK: quantum.insert %[[QREG]][{{.*}}], %[[CPERESDG_CNOT]]#0 : !quantum.reg, !quantum.bit
    // CHECK: quantum.insert %[[QREG]][{{.*}}], %[[CPERESDG_CNOT]]#1 : !quantum.reg, !quantum.bit
    // CHECK: quantum.insert %[[QREG]][{{.*}}], %[[CPERESDG_CTRL]] : !quantum.reg, !quantum.bit
    // CHECK: quantum.dealloc %[[QREG]] : !quantum.reg

    // Prepare qubits
    %i0 = arith.constant 0 : index
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %r0_0 = memref.alloc() : memref<3x!mqtopt.Qubit>
    %q0_0 = memref.load %r0_0[%i0] : memref<3x!mqtopt.Qubit>
    %q1_0 = memref.load %r0_0[%i1] : memref<3x!mqtopt.Qubit>
    %q2_0 = memref.load %r0_0[%i2] : memref<3x!mqtopt.Qubit>

    // I/H/V/Vdg/S/Sdg/T/Tdg/Peres/Peresdg (non-controlled)
    %q0_1 = mqtopt.i()   %q0_0 : !mqtopt.Qubit
    %q0_2 = mqtopt.h()   %q0_1 : !mqtopt.Qubit
    %q0_3 = mqtopt.v()   %q0_2 : !mqtopt.Qubit
    %q0_4 = mqtopt.vdg() %q0_3 : !mqtopt.Qubit
    %q0_5 = mqtopt.s()   %q0_4 : !mqtopt.Qubit
    %q0_6 = mqtopt.sdg() %q0_5 : !mqtopt.Qubit
    %q0_7 = mqtopt.t()   %q0_6 : !mqtopt.Qubit
    %q0_8 = mqtopt.tdg() %q0_7 : !mqtopt.Qubit
    %q0_9, %q1_1 = mqtopt.peres() %q0_8, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
    %q0_10, %q1_2 = mqtopt.peresdg() %q0_9, %q1_1 : !mqtopt.Qubit, !mqtopt.Qubit

    // Controlled H/V/Vdg/S/Sdg/T/Tdg
    %q0_11,  %q1_3 = mqtopt.h()   %q0_10 ctrl %q1_2 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_12, %q1_4 = mqtopt.v()   %q0_11 ctrl %q1_3 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_13, %q1_5 = mqtopt.vdg() %q0_12 ctrl %q1_4 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_14, %q1_6 = mqtopt.s()   %q0_13 ctrl %q1_5 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_15, %q1_7 = mqtopt.sdg() %q0_14 ctrl %q1_6 : !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_16, %q1_8, %q2_1 = mqtopt.peres()   %q0_15, %q1_7 ctrl %q2_0 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit
    %q0_17, %q1_9, %q2_2 = mqtopt.peresdg() %q0_16, %q1_8 ctrl %q2_1 : !mqtopt.Qubit, !mqtopt.Qubit ctrl !mqtopt.Qubit

    // Release qubits
    memref.store %q0_17, %r0_0[%i0] : memref<3x!mqtopt.Qubit>
    memref.store %q1_9, %r0_0[%i1] : memref<3x!mqtopt.Qubit>
    memref.store %q2_2, %r0_0[%i2] : memref<3x!mqtopt.Qubit>
    memref.dealloc %r0_0 : memref<3x!mqtopt.Qubit>
    return
  }
}
