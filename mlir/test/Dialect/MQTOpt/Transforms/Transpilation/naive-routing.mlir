// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

// RUN: quantum-opt %s -split-input-file --pass-pipeline="builtin.module(routing,verify-routing)" | FileCheck %s

module {
    // CHECK-LABEL: func.func @entrySABRE
    func.func @entrySABRE() attributes { entry_point } {

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

    // CHECK-LABEL: func.func @entryBell
    func.func @entryBell() attributes { entry_point } {

        //
        // The bell state.
        //
        // This test shows that alloc's don't have to be grouped.
        //
        //    ┌───┐
        // 0:─┤ H ├────■───
        //    └───┘  ┌─┴─┐
        // 1:────────┤ X ├─
        //           └───┘

        %q0_0_bell = mqtopt.allocQubit
        %q0_1_bell = mqtopt.h() %q0_0_bell : !mqtopt.Qubit

        %q1_0_bell = mqtopt.allocQubit
        %q1_1_bell, %q0_2_bell = mqtopt.x() %q1_0_bell ctrl %q0_1_bell : !mqtopt.Qubit ctrl !mqtopt.Qubit

        %q0_3_bell, %m0_0_bell = "mqtopt.measure"(%q0_2_bell) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %q1_2_bell, %m1_0_bell = "mqtopt.measure"(%q1_1_bell) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

        mqtopt.deallocQubit %q0_3_bell
        mqtopt.deallocQubit %q1_2_bell

        return
    }

    // CHECK-LABEL: func.func @entryBellLoop
    func.func @entryBellLoop() attributes { entry_point } {

        //
        // Bell in a loop.
        //
        // This test shows that the routing algorithm can handle
        // alloc statements inside the loop body.
        //
        // ┌    ┌───┐         ┐^1000
        // │ 0:─┤ H ├────■─── │
        // │    └───┘  ┌─┴─┐  │
        // │ 1:────────┤ X ├─ │
        // └           └───┘  ┘

        %lb = index.constant 0
        %ub = index.constant 1000
        %step = index.constant 1

        scf.for %iv = %lb to %ub step %step {
            %q0_0_bell1000 = mqtopt.allocQubit
            %q1_0_bell1000 = mqtopt.allocQubit

            %q0_1_bell1000 = mqtopt.h() %q0_0_bell1000 : !mqtopt.Qubit
            %q1_1_bell1000, %q0_2_bell1000 = mqtopt.x() %q1_0_bell1000 ctrl %q0_1_bell1000 : !mqtopt.Qubit ctrl !mqtopt.Qubit

            %q0_3_bell1000, %m0_0_bell1000 = "mqtopt.measure"(%q0_2_bell1000) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
            %q1_2_bell1000, %m1_0_bell1000 = "mqtopt.measure"(%q1_1_bell1000) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

            mqtopt.deallocQubit %q0_3_bell1000
            mqtopt.deallocQubit %q1_2_bell1000
        }

        return
    }

    // CHECK-LABEL: func.func @entryGHZ
    func.func @entryGHZ() attributes { entry_point } {

        //
        // GHZ in a loop.
        //
        // This test shows that the routing algorithm can handle
        // loop-carried qubit (and non-qubit) values as well as
        // classical if statements.

        %lb = index.constant 0
        %ub = index.constant 1000
        %step = index.constant 1

        %q0_0_ghz1000 = mqtopt.allocQubit
        %q1_0_ghz1000 = mqtopt.allocQubit
        %q2_0_ghz1000 = mqtopt.allocQubit

        %zero = arith.constant 0 : i32

        %nones, %q0_1_ghz1000, %q1_1_ghz1000, %q2_1_ghz1000 = scf.for %iv = %lb to %ub step %step
            iter_args(%nones0 = %zero, %q0_i_0 = %q0_0_ghz1000, %q1_i_0 = %q1_0_ghz1000, %q2_i_0 = %q2_0_ghz1000) -> (i32, !mqtopt.Qubit, !mqtopt.Qubit, !mqtopt.Qubit) {
            %q0_i_1 = "mqtopt.reset"(%q0_i_0) : (!mqtopt.Qubit) -> !mqtopt.Qubit
            %q1_i_1 = "mqtopt.reset"(%q1_i_0) : (!mqtopt.Qubit) -> !mqtopt.Qubit
            %q2_i_1 = "mqtopt.reset"(%q2_i_0) : (!mqtopt.Qubit) -> !mqtopt.Qubit

            %q0_i_2 = mqtopt.h() %q0_i_1 : !mqtopt.Qubit
            %q1_i_2, %q0_i_3 = mqtopt.x() %q1_i_1 ctrl %q0_i_2 : !mqtopt.Qubit ctrl !mqtopt.Qubit
            %q2_i_2, %q1_i_3 = mqtopt.x() %q2_i_1 ctrl %q1_i_2 : !mqtopt.Qubit ctrl !mqtopt.Qubit

            %q0_i_4, %m0 = "mqtopt.measure"(%q0_i_3) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
            %q1_i_4, %m1 = "mqtopt.measure"(%q1_i_3) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
            %q2_i_3, %m2 = "mqtopt.measure"(%q2_i_2) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

            %nones2 = scf.if %m0 -> i32 {
                %one = arith.constant 1 : i32
                %nones1 = arith.addi %nones0, %one : i32
                scf.yield %nones1 : i32
            } else {
                scf.yield %nones0 : i32
            }

            scf.yield %nones2, %q0_i_4, %q1_i_4, %q2_i_3 : i32, !mqtopt.Qubit, !mqtopt.Qubit, !mqtopt.Qubit
        }

        mqtopt.deallocQubit %q0_1_ghz1000
        mqtopt.deallocQubit %q1_1_ghz1000
        mqtopt.deallocQubit %q2_1_ghz1000

        return
    }

    // CHECK-LABEL: func.func @entryAll
    func.func @entryAll() attributes { entry_point } {

        //
        // All of the above quantum computations in a single entry point.
        // This test shows that the algorithm can handle multiple computations
        // in a single function.

        //
        // SABRE

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

        //
        // The bell state.

        %q0_0_bell = mqtopt.allocQubit
        %q0_1_bell = mqtopt.h() %q0_0_bell : !mqtopt.Qubit

        %q1_0_bell = mqtopt.allocQubit
        %q1_1_bell, %q0_2_bell = mqtopt.x() %q1_0_bell ctrl %q0_1_bell : !mqtopt.Qubit ctrl !mqtopt.Qubit

        %q0_3_bell, %m0_0_bell = "mqtopt.measure"(%q0_2_bell) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
        %q1_2_bell, %m1_0_bell = "mqtopt.measure"(%q1_1_bell) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

        mqtopt.deallocQubit %q0_3_bell
        mqtopt.deallocQubit %q1_2_bell

        //
        // Bell in a loop.

        %lb = index.constant 0
        %ub = index.constant 1000
        %step = index.constant 1

        scf.for %iv = %lb to %ub step %step {
            %q0_0_bell1000 = mqtopt.allocQubit
            %q1_0_bell1000 = mqtopt.allocQubit

            %q0_1_bell1000 = mqtopt.h() %q0_0_bell1000 : !mqtopt.Qubit
            %q1_1_bell1000, %q0_2_bell1000 = mqtopt.x() %q1_0_bell1000 ctrl %q0_1_bell1000 : !mqtopt.Qubit ctrl !mqtopt.Qubit

            %q0_3_bell1000, %m0_0_bell1000 = "mqtopt.measure"(%q0_2_bell1000) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
            %q1_2_bell1000, %m1_0_bell1000 = "mqtopt.measure"(%q1_1_bell1000) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

            mqtopt.deallocQubit %q0_3_bell1000
            mqtopt.deallocQubit %q1_2_bell1000
        }

        //
        // GHZ in a loop.

        %q0_0_ghz1000 = mqtopt.allocQubit
        %q1_0_ghz1000 = mqtopt.allocQubit
        %q2_0_ghz1000 = mqtopt.allocQubit

        %zero = arith.constant 0 : i32

        %nones, %q0_1_ghz1000, %q1_1_ghz1000, %q2_1_ghz1000 = scf.for %iv = %lb to %ub step %step
            iter_args(%nones0 = %zero, %q0_i_0 = %q0_0_ghz1000, %q1_i_0 = %q1_0_ghz1000, %q2_i_0 = %q2_0_ghz1000) -> (i32, !mqtopt.Qubit, !mqtopt.Qubit, !mqtopt.Qubit) {
            %q0_i_1 = "mqtopt.reset"(%q0_i_0) : (!mqtopt.Qubit) -> !mqtopt.Qubit
            %q1_i_1 = "mqtopt.reset"(%q1_i_0) : (!mqtopt.Qubit) -> !mqtopt.Qubit
            %q2_i_1 = "mqtopt.reset"(%q2_i_0) : (!mqtopt.Qubit) -> !mqtopt.Qubit

            %q0_i_2 = mqtopt.h() %q0_i_1 : !mqtopt.Qubit
            %q1_i_2, %q0_i_3 = mqtopt.x() %q1_i_1 ctrl %q0_i_2 : !mqtopt.Qubit ctrl !mqtopt.Qubit
            %q2_i_2, %q1_i_3 = mqtopt.x() %q2_i_1 ctrl %q1_i_2 : !mqtopt.Qubit ctrl !mqtopt.Qubit

            %q0_i_4, %m0 = "mqtopt.measure"(%q0_i_3) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
            %q1_i_4, %m1 = "mqtopt.measure"(%q1_i_3) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)
            %q2_i_3, %m2 = "mqtopt.measure"(%q2_i_2) : (!mqtopt.Qubit) -> (!mqtopt.Qubit, i1)

            %nones2 = scf.if %m0 -> i32 {
                %one = arith.constant 1 : i32
                %nones1 = arith.addi %nones0, %one : i32
                scf.yield %nones1 : i32
            } else {
                scf.yield %nones0 : i32
            }

            scf.yield %nones2, %q0_i_4, %q1_i_4, %q2_i_3 : i32, !mqtopt.Qubit, !mqtopt.Qubit, !mqtopt.Qubit
        }

        mqtopt.deallocQubit %q0_1_ghz1000
        mqtopt.deallocQubit %q1_1_ghz1000
        mqtopt.deallocQubit %q2_1_ghz1000

        return
    }
}
