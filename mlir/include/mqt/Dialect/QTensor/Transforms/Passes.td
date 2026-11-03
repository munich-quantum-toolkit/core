// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_QTENSOR_TRANSFORMS_PASSES_TD
#define MLIR_DIALECT_QTENSOR_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def ShrinkQTensorToFitPass : Pass<"qtensor-shrink-to-fit", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::qtensor::QTensorDialect",
                           "mlir::arith::ArithDialect"];
  let summary = "Compact static quantum tensors";
  let description = [{
    Compacts static quantum tensors whose linear allocation-to-deallocation
    chain consists of constant-index extracts and inserts. Retained indices
    preserve their relative order and source-qubit mapping.

    Run canonicalization first to fold size and index expressions. Tensors with
    nonconstant indices or unsupported uses are left unchanged.
  }];
}

#endif // MLIR_DIALECT_QTENSOR_TRANSFORMS_PASSES_TD
