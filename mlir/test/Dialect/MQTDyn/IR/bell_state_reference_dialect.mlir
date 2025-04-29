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
        %r0 = "mqtdyn.allocQubitRegister" () {"size_attr" = 2 : i64} : () -> !mqtdyn.QubitRegister
        %q0 = "mqtdyn.extractQubit" (%r0) {"index_attr" = 0 : i64} : (!mqtdyn.QubitRegister) -> !mqtdyn.Qubit

        %i = arith.constant 0 : i64
        %q1 = "mqtdyn.extractQubit" (%r0, %i) : (!mqtdyn.QubitRegister, i64) -> !mqtdyn.Qubit

        mqtdyn.x () %q0
        mqtdyn.x () %q1 ctrl %q0

        %c0 = "mqtdyn.measure" (%q0) : (!mqtdyn.Qubit) -> i1
        %c1 = "mqtdyn.measure" (%q1) : (!mqtdyn.Qubit) -> i1


        "mqtdyn.deallocQubitRegister" (%r0) : (!mqtdyn.QubitRegister) -> ()

        return %c0, %c1 : i1, i1
    }
}
