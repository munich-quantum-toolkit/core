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
// Pauli family (X, Y, Z) and controlled variants
// Groups: Allocation & extraction / Uncontrolled / Controlled / Reinsertion
// ============================================================================
module {
  // CHECK-LABEL: func.func @testCatalystQuantumToMQTOptPauliGates
  func.func @testCatalystQuantumToMQTOptPauliGates() {
    // Prepare qubits
    // CHECK: %[[QREG:.*]] = "mqtopt.allocQubitRegister"() <{size_attr = 4 : i64}> : () -> !mqtopt.QubitRegister
    // CHECK: %[[R0:.*]], %[[Q0:.*]] = "mqtopt.extractQubit"(%[[QREG]]) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[R1:.*]], %[[Q1:.*]] = "mqtopt.extractQubit"(%[[R0]]) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[R2:.*]], %[[Q2:.*]] = "mqtopt.extractQubit"(%[[R1]]) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
    // CHECK: %[[R3:.*]], %[[Q3:.*]] = "mqtopt.extractQubit"(%[[R2]]) <{index_attr = 3 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)

    // Non-controlled Pauli gates
    // CHECK: %[[X1:.*]] = mqtopt.x(static [] mask []) %[[Q0]] : !mqtopt.Qubit
    // CHECK: %[[Y1:.*]] = mqtopt.y(static [] mask []) %[[X1]] : !mqtopt.Qubit
    // CHECK: %[[Z1:.*]] = mqtopt.z(static [] mask []) %[[Y1]] : !mqtopt.Qubit
    // CHECK: %[[I1:.*]] = mqtopt.i(static [] mask []) %[[Z1]] : !mqtopt.Qubit

    // Controlled Pauli gates
    // CHECK: %[[TRUE:.*]] = arith.constant true
    // CHECK: %[[T1:.*]], %[[C1:.*]] = mqtopt.x(static [] mask []) %[[I1]] ctrl %[[Q1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[T2:.*]], %[[C2:.*]] = mqtopt.y(static [] mask []) %[[T1]] ctrl %[[C1]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[T3:.*]], %[[C3:.*]] = mqtopt.z(static [] mask []) %[[T2]] ctrl %[[C2]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[T4:.*]], %[[C4:.*]] = mqtopt.i(static [] mask []) %[[T3]] ctrl %[[C3]] : !mqtopt.Qubit ctrl !mqtopt.Qubit

    // C gates
    // CHECK: %[[T5:.*]], %[[C5:.*]] = mqtopt.x(static [] mask []) %[[C4]] ctrl %[[T4]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[T6:.*]], %[[C6:.*]] = mqtopt.y(static [] mask []) %[[C5]] ctrl %[[T5]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[T7:.*]], %[[C7:.*]] = mqtopt.z(static [] mask []) %[[C6]] ctrl %[[T6]] : !mqtopt.Qubit ctrl !mqtopt.Qubit
    // CHECK: %[[T8:.*]], %[[C8:.*]]:2 = mqtopt.x(static [] mask []) %[[Q2]] ctrl %[[T7]], %[[C7]] : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit

    // Controlled-C gates
    // CHECK: %[[T9:.*]], %[[C9:.*]]:2 = mqtopt.x(static [] mask []) %[[C8]]#0 ctrl %[[C8]]#1, %[[T8]] : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[T10:.*]], %[[C10:.*]]:2 = mqtopt.y(static [] mask []) %[[C9]]#0 ctrl %[[C9]]#1, %[[T9]] : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[T11:.*]], %[[C11:.*]]:2 = mqtopt.z(static [] mask []) %[[C10]]#0 ctrl %[[C10]]#1, %[[T10]] : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit
    // CHECK: %[[T12:.*]], %[[C12:.*]]:3 = mqtopt.x(static [] mask []) %[[C11]]#1 ctrl %[[Q3]], %[[T11]], %[[C11]]#0 : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit, !mqtopt.Qubit

    // Release qubits
    // CHECK: %[[IR1:.*]] = "mqtopt.insertQubit"(%[[R3]], %[[C12]]#2) <{index_attr = 3 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %[[IR2:.*]] = "mqtopt.insertQubit"(%[[IR1]], %[[C12]]#1) <{index_attr = 2 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %[[IR3:.*]] = "mqtopt.insertQubit"(%[[IR2]], %[[C12]]#0) <{index_attr = 1 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: %[[IR4:.*]] = "mqtopt.insertQubit"(%[[IR3]], %[[T12]]) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
    // CHECK: "mqtopt.deallocQubitRegister"(%[[IR4]]) : (!mqtopt.QubitRegister) -> ()

    // Prepare qubits
    %qreg = quantum.alloc( 4) : !quantum.reg
    %q0 = quantum.extract %qreg[ 0] : !quantum.reg -> !quantum.bit
    %q1 = quantum.extract %qreg[ 1] : !quantum.reg -> !quantum.bit
    %q2 = quantum.extract %qreg[ 2] : !quantum.reg -> !quantum.bit
    %q3 = quantum.extract %qreg[ 3] : !quantum.reg -> !quantum.bit

    // Non-controlled Pauli gates
    %q0_x = quantum.custom "PauliX"() %q0 : !quantum.bit
    %q0_y = quantum.custom "PauliY"() %q0_x : !quantum.bit
    %q0_z = quantum.custom "PauliZ"() %q0_y : !quantum.bit
    %q0_i = quantum.custom "Identity"() %q0_z : !quantum.bit

    %true = arith.constant true

    // Controlled Pauli gates
    %q0_ctrlx, %q1_ctrlx = quantum.custom "PauliX"() %q0_i ctrls(%q1) ctrlvals(%true) :!quantum.bit ctrls !quantum.bit
    %q0_ctrly, %q1_ctrly = quantum.custom "PauliY"() %q0_ctrlx ctrls(%q1_ctrlx) ctrlvals(%true) :!quantum.bit ctrls !quantum.bit
    %q0_ctrlz, %q1_ctrlz = quantum.custom "PauliZ"() %q0_ctrly ctrls(%q1_ctrly) ctrlvals(%true) :!quantum.bit ctrls !quantum.bit
    %q0_ctrli, %q1_ctrli = quantum.custom "Identity"() %q0_ctrlz ctrls(%q1_ctrlz) ctrlvals(%true) :!quantum.bit ctrls !quantum.bit

    // C gates
    %q0_cx, %q1_cx = quantum.custom "CNOT"() %q0_ctrli, %q1_ctrli : !quantum.bit, !quantum.bit
    %q0_cy, %q1_cy = quantum.custom "CY"() %q0_cx, %q1_cx : !quantum.bit, !quantum.bit
    %q0_cz, %q1_cz = quantum.custom "CZ"() %q0_cy, %q1_cy : !quantum.bit, !quantum.bit
    %q0_ct, %q1_ct, %q2_ct = quantum.custom "Toffoli"() %q0_cz, %q1_cz, %q2 : !quantum.bit, !quantum.bit, !quantum.bit

    // Controlled-C gates
    %q0_ccx, %q1_ccx, %q2_ccx = quantum.custom "CNOT"() %q0_ct, %q1_ct ctrls(%q2_ct) ctrlvals(%true) :!quantum.bit, !quantum.bit ctrls !quantum.bit
    %q0_ccy, %q1_ccy, %q2_ccy = quantum.custom "CY"() %q0_ccx, %q1_ccx ctrls(%q2_ccx) ctrlvals(%true) :!quantum.bit, !quantum.bit ctrls !quantum.bit
    %q0_ccz, %q1_ccz, %q2_ccz = quantum.custom "CZ"() %q0_ccy, %q1_ccy ctrls(%q2_ccy) ctrlvals(%true) :!quantum.bit, !quantum.bit ctrls !quantum.bit
    %q0_cccx, %q1_cccx, %q2_cccx, %q3_cccx = quantum.custom "Toffoli"() %q0_ccz, %q1_ccz, %q2_ccz ctrls(%q3) ctrlvals(%true) :!quantum.bit, !quantum.bit, !quantum.bit ctrls !quantum.bit

    // Release qubits
    %qreg1 = quantum.insert %qreg[ 3], %q3_cccx : !quantum.reg, !quantum.bit
    %qreg2 = quantum.insert %qreg1[ 2], %q2_cccx : !quantum.reg, !quantum.bit
    %qreg3 = quantum.insert %qreg2[ 1], %q1_cccx : !quantum.reg, !quantum.bit
    %qreg4 = quantum.insert %qreg3[ 0], %q0_cccx : !quantum.reg, !quantum.bit

    quantum.dealloc %qreg4 : !quantum.reg
    return
  }
}
