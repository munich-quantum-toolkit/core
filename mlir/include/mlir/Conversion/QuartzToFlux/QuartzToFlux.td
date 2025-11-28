// Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
// Copyright (c) 2025 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def QuartzToFlux : Pass<"quartz-to-flux"> {
  let summary = "Convert Quartz dialect to Flux dialect.";

  let description = [{
      This pass converts all operations from the Quartz dialect to their equivalent
      operations in the Flux dialect. It handles the transformation of qubit
      references in Quartz to qubit values in Flux, ensuring that the semantics
      of quantum operations are preserved during the conversion process.
   }];

  // Define dependent dialects
  let dependentDialects = [
    "mlir::quartz::QuartzDialect",
    "mlir::flux::FluxDialect"
  ];
}
