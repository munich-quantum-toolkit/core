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
  let summary = "Compact static qubit registers";
  let description = [{
    Compacts one-dimensional static `memref` allocations of `!qc.qubit` with
    identity layout and default memory space. Supported uses are constant-index
    loads and a matching deallocation. Retained indices preserve their relative
    order.

    Run canonicalization first to fold index expressions. Allocations with
    nonconstant indices or unsupported uses are left unchanged.
  }];
}

#endif // MLIR_DIALECT_QC_TRANSFORMS_PASSES_TD
