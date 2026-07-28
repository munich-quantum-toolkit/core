// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

module {
  func.func @main() -> (i1, i1) attributes {passthrough = ["entry_point"]} {
    %q0 = qco.alloc : !qco.qubit
    %q1 = qco.alloc : !qco.qubit
    %q0_h = qco.h %q0 : !qco.qubit -> !qco.qubit
    %q1_h = qco.h %q1 : !qco.qubit -> !qco.qubit
    %q0_out, %first = qco.measure %q0_h : !qco.qubit
    %q1_out, %second = qco.measure %q1_h : !qco.qubit
    qco.sink %q0_out : !qco.qubit
    qco.sink %q1_out : !qco.qubit
    return %first, %second : i1, i1
  }
}
