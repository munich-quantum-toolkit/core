// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --verify-routing-sc -verify-diagnostics

module {
    func.func @tooManyQubits() attributes {passthrough = ["entry_point"]} {
        %q0_0 = mqtopt.qubit 0
        %q1_0 = mqtopt.qubit 1
        %q2_0 = mqtopt.qubit 2

        // expected-error@+1 {{'mqtopt.x' op acts on more than two qubits}}
        %q0_1, %q1_1, %q2_1 = mqtopt.x() %q0_0 ctrl %q1_0, %q2_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit, !mqtopt.Qubit

        return
    }
}

// -----

module {
    func.func @gateNotExecutable() attributes {passthrough = ["entry_point"]} {
        %q0_0 = mqtopt.qubit 0
        %q4_0 = mqtopt.qubit 4

        // expected-error@+1 {{'mqtopt.x' op (0,4) is not executable on target architecture 'MQT-Test'}}
        %q0_1, %q4_1 = mqtopt.x() %q0_0 ctrl %q4_0 : !mqtopt.Qubit ctrl !mqtopt.Qubit

        return
    }
}

// -----

module {
    func.func @layoutsDontMatch() attributes {passthrough = ["entry_point"]} {
        %q0_0 = mqtopt.qubit 0
        %q1_0 = mqtopt.qubit 1
        %c0 = arith.constant 0 : i1

        %q0_1, %q1_1 = scf.if %c0 -> (!mqtopt.Qubit, !mqtopt.Qubit) {
            %q0_1, %q1_1 = mqtopt.swap() %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
            // expected-error@+1 {{'scf.yield' op layouts must match after restoration}}
            scf.yield %q0_1, %q1_1 : !mqtopt.Qubit, !mqtopt.Qubit
        } else {
            scf.yield %q0_0, %q1_0 : !mqtopt.Qubit, !mqtopt.Qubit
        }

        return
    }
}
