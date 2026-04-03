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
  let summary = "Shrink static qtensors to their actually accessed indices.";
  let description = [{
    Shrinks one-dimensional static qtensors by tracing linear tensor chains from
    `qtensor.dealloc` to `qtensor.alloc` and rebuilding the chain on a compact
    allocation that only keeps accessed indices.
  }];
}

#endif // MLIR_DIALECT_QTENSOR_TRANSFORMS_PASSES_TD
