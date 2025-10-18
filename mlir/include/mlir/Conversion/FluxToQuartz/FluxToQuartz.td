// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def FluxToQuartz : Pass<"flux-to-quartz"> {
  let summary = "Convert Flux dialect to Quartz dialect.";

  let description = [{
      This pass converts all operations from the Flux dialect to their equivalent
      operations in the Quartz dialect. It handles the transformation of qubit
      values in Flux to qubit references in Quartz, ensuring that the semantics
      of quantum operations are preserved during the conversion process.
   }];

  // Define dependent dialects
  let dependentDialects = [
    "mlir::memref::MemRefDialect",
    "mlir::flux::FluxDialect",
    "mlir::quartz::QuartzDialect"
  ];
}
