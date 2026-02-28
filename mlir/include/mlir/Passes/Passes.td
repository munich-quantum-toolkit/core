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
// Optimization Passes
//===----------------------------------------------------------------------===//

def GateDecompositionPass : Pass<"gate-decomposition", "mlir::ModuleOp"> {
  let dependentDialects = [ "mlir::arith::ArithDialect", "mlir::qco::QCODialect" ];
  let summary = "This pass performs various gate decompositions to translate quantum gates being used.";
  let description = [{
    Decomposes series of operations that operate on up to two qubits into a sequence of up to three
    two-qubit basis gates and single-qubit operations.
  }];
}

#endif // QCO_PASSES
