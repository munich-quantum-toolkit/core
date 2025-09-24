// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --pass-pipeline="builtin.module(route-sc,verify-routing-sc)" -verify-diagnostics


module {
    func.func @entryTooManyQubits() attributes { entry_point } {
        %q0_0 = mqtopt.allocQubit
        %q1_0 = mqtopt.allocQubit
        %q2_0 = mqtopt.allocQubit
        %q3_0 = mqtopt.allocQubit
        %q4_0 = mqtopt.allocQubit
        %q5_0 = mqtopt.allocQubit

        // expected-error@+1 {{'mqtopt.allocQubit' op requires one too many qubits for the targeted architecture}}
        %q6_0 = mqtopt.allocQubit

        mqtopt.deallocQubit %q0_0
        mqtopt.deallocQubit %q1_0
        mqtopt.deallocQubit %q2_0
        mqtopt.deallocQubit %q3_0
        mqtopt.deallocQubit %q4_0
        mqtopt.deallocQubit %q5_0
        mqtopt.deallocQubit %q6_0

        return
    }
}
