// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef QCO_PASSES
#define QCO_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Transpilation Passes
//===----------------------------------------------------------------------===//

def HeuristicMappingPass : Pass<"map", "mlir::ModuleOp"> {
  let dependentDialects = [ "mlir::qco::QCODialect" ];
  let summary = "This pass ensures that a program meets the connectivity constraints of a given architecture.";
  let description = [{
    This pass inserts SWAP operations to ensure two-qubit gates are executable on a given target architecture.
  }];
}

#endif // QCO_PASSES
