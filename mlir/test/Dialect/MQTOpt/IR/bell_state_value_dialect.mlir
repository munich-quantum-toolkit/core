// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s | FileCheck %s

module {
    // CHECK-LABEL: func @bell_state()
    func.func @bell_state() -> (i1, i1) {
        %0 = "mqtopt.allocQubitRegister"() <{size_attr = 2 : i64}> : () -> !mqtopt.QubitRegister
        %out_qreg, %out_qubit = "mqtopt.extractQubit"(%0) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %c2_i64 = arith.constant 2 : i64
        %1 = "mqtopt.allocQubitRegister"(%c2_i64) : (i64) -> !mqtopt.QubitRegister
        %c0_i64 = arith.constant 0 : i64
        %out_qreg_0, %out_qubit_1 = "mqtopt.extractQubit"(%1, %c0_i64) : (!mqtopt.QubitRegister, i64) -> (!mqtopt.QubitRegister, !mqtopt.Qubit)
        %2 = mqtopt.x() %out_qubit : !mqtopt.Qubit
        %3, %4 = mqtopt.x() %2 ctrl %out_qubit_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit
        %cst = arith.constant 3.000000e-01 : f64
        %5, %6 = mqtopt.rz(%cst) %3 nctrl %4 : !mqtopt.Qubit nctrl !mqtopt.Qubit
        %7 = mqtopt.u(%cst, %cst static [3.000000e-01] mask [false, true, false]) %5 : !mqtopt.Qubit
        %out_qubits, %out_bits = "mqtopt.measure"(%7) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %out_qubits_2, %out_bits_3 = "mqtopt.measure"(%6) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %8 = "mqtopt.insertQubit"(%out_qreg, %out_qubits) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        %9 = "mqtopt.insertQubit"(%out_qreg_0, %out_qubits_2) <{index_attr = 0 : i64}> : (!mqtopt.QubitRegister, !mqtopt.Qubit) -> !mqtopt.QubitRegister
        "mqtopt.deallocQubitRegister"(%8) : (!mqtopt.QubitRegister) -> ()
        "mqtopt.deallocQubitRegister"(%9) : (!mqtopt.QubitRegister) -> ()
        return %out_bits, %out_bits_3 : i1, i1
    }
}
