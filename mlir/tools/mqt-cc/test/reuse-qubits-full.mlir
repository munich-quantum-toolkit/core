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
    %q0_out, %first = qco.measure %q0 : !qco.qubit
    %q1_out = qco.if %first args(%arg = %q1) -> (!qco.qubit) {
      %result = qco.x %arg : !qco.qubit -> !qco.qubit
      qco.yield %result : !qco.qubit
    } else args(%arg = %q1) {
      qco.yield %arg : !qco.qubit
    }
    %q1_measured, %second = qco.measure %q1_out : !qco.qubit
    qco.sink %q0_out : !qco.qubit
    qco.sink %q1_measured : !qco.qubit
    return %first, %second : i1, i1
  }
}
