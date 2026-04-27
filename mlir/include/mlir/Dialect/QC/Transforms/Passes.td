// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_QC_TRANSFORMS_PASSES_TD
#define MLIR_DIALECT_QC_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def ShrinkQubitRegistersPass
    : Pass<"qc-shrink-qubit-registers", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::qc::QCDialect", "mlir::arith::ArithDialect",
                           "mlir::memref::MemRefDialect"];
  let summary =
      "Shrink static `qc::QubitType` MemRef registers to accessed indices.";
  let description = [{
    Shrinks one-dimensional static MemRef registers with element type `!qc.qubit` by removing never-read indices and remapping `memref.load` users accordingly.
  }];
}

#endif // MLIR_DIALECT_QC_TRANSFORMS_PASSES_TD
