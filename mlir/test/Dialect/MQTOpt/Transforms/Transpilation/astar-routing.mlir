// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --pass-pipeline="builtin.module(placement-sc{strategy=random}, route-sc{method=qmap},verify-routing-sc)" -verify-diagnostics | FileCheck %s

module {
    // CHECK-LABEL: func.func @entrySABRE
    func.func @entrySABRE() attributes {passthrough = ["entry_point"]} {

        //
        // Figure 4 in SABRE Paper "Tackling the Qubit Mapping Problem for NISQ-Era Quantum Devices".
        //
        //                        ┌───┐
        // 0: ───────■────────────┤ 8 ├
        //         ┌─┴─┐          └─┬─┘
        // 1: ──■──┤ 3 ├──■─────────┼──
        //    ┌─┴─┐└───┘┌─┴─┐       │
        // 2: ┤ 1 ├──■──┤ 5 ├───────┼──
        //    └───┘┌─┴─┐└───┘       │
        // 3: ─────┤ 4 ├──■────■────■──
        //         └───┘  │  ┌─┴─┐
        // 4: ──■─────────┼──┤ 7 ├─────
        //    ┌─┴─┐     ┌─┴─┐└───┘
        // 5: ┤ 2 ├─────┤ 6 ├──────────
        //    └───┘     └───┘

        %q0_0 = mqtopt.allocQubit
        %q1_0 = mqtopt.allocQubit
        %q2_0 = mqtopt.allocQubit
        %q3_0 = mqtopt.allocQubit
        %q4_0 = mqtopt.allocQubit
        %q5_0 = mqtopt.allocQubit

        %q0_1 = mqtopt.h() %q0_0 : !mqtopt.Qubit
        %q1_1 = mqtopt.h() %q1_0 : !mqtopt.Qubit
        %q4_1 = mqtopt.h() %q4_0 : !mqtopt.Qubit

        %q0_2 = mqtopt.z() %q0_1 : !mqtopt.Qubit
        %q2_1, %q1_2 = mqtopt.x() %q2_0 ctrl %q1_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit // g1
        %q5_1, %q4_2 = mqtopt.x() %q5_0 ctrl %q4_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit // g2

        %q1_3, %q0_3 = mqtopt.x() %q1_2 ctrl %q0_2 : !mqtopt.Qubit ctrl !mqtopt.Qubit // g3
        %q3_1, %q2_2 = mqtopt.x() %q3_0 ctrl %q2_1 : !mqtopt.Qubit ctrl !mqtopt.Qubit // g4

        %q2_3 = mqtopt.h() %q2_2 : !mqtopt.Qubit
        %q3_2 = mqtopt.h() %q3_1 : !mqtopt.Qubit

        %q2_4, %q1_4 = mqtopt.x() %q2_3 ctrl %q1_3 : !mqtopt.Qubit ctrl !mqtopt.Qubit // g5
        %q5_2, %q3_3 = mqtopt.x() %q5_1 ctrl %q3_2 : !mqtopt.Qubit ctrl !mqtopt.Qubit // g6

        %q3_4 = mqtopt.z() %q3_3 : !mqtopt.Qubit

        %q3_5, %q4_3 = mqtopt.x() %q3_4 ctrl %q4_2 : !mqtopt.Qubit ctrl !mqtopt.Qubit // g7

        %q0_4, %q3_6 = mqtopt.x() %q0_3 ctrl %q3_5 : !mqtopt.Qubit ctrl !mqtopt.Qubit // g8

        mqtopt.deallocQubit %q0_4
        mqtopt.deallocQubit %q1_4
        mqtopt.deallocQubit %q2_4
        mqtopt.deallocQubit %q3_6
        mqtopt.deallocQubit %q4_3
        mqtopt.deallocQubit %q5_2

        return
    }

    // CHECK-LABEL: func.func @noEntryPoint
    func.func @noEntryPoint() {
        // CHECK: %[[ANY:.*]] = mqtopt.allocQubit
        %q0 = mqtopt.allocQubit
        return
    }
}
