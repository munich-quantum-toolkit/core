// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

module {
    func.func @main() attributes { passthrough = ["entry_point"] } {
        %q0 = qc.alloc : !qc.qubit
        %q1 = qc.alloc : !qc.qubit

        qc.h %q0 : !qc.qubit
        qc.ctrl(%q0) { qc.x %q1 : !qc.qubit } : !qc.qubit

        %c0 = qc.measure %q0 : !qc.qubit -> i1
        %c1 = qc.measure %q1 : !qc.qubit -> i1

        qc.dealloc %q0 : !qc.qubit
        qc.dealloc %q1 : !qc.qubit

        func.return
    }
}
