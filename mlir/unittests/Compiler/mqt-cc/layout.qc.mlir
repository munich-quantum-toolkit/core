// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

module {
  func.func @main() attributes {mqt.entry_point, mqt.layout = {
  physical_size = 2 : i64, initial = array<i64: 1, 0>,
  output_order = array<i64: 0, 1>, ancillas = array<i64>, registers = []
  }} {
    %q = memref.alloc() : memref<2x!qc.qubit>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %a = memref.load %q[%c0] : memref<2x!qc.qubit>
    %b = memref.load %q[%c1] : memref<2x!qc.qubit>
    qc.h %a : !qc.qubit
    qc.x %b : !qc.qubit
    memref.dealloc %q : memref<2x!qc.qubit>
    return
  }
}
