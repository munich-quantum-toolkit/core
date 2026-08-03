// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_UTILS_TRANSFORMS_PASSES_TD
#define MLIR_DIALECT_UTILS_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def NormalizeGlobalPhases : Pass<"normalize-global-phases", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::arith::ArithDialect", "mlir::qc::QCDialect",
                           "mlir::qco::QCODialect"];
  let summary = "Normalize scoped QC and QCO global phases";
  let description = [{
    Combines direct `qc.gphase` and `qco.gphase` operations into at most one
    operation per basic block, placed immediately before the block terminator.
    Nested regions are normalized independently.

    A normalized phase is moved out of `inv`, negating its angle, and out of
    `pow` only when the exponent is a finite compile-time integer. A phase in a
    `ctrl` body becomes a relative phase on the control register: `p` for one
    control or a smaller controlled `p` for multiple controls. Phase angles
    defined inside a modifier are moved only when their complete local SSA
    dependency slice is pure and independent of modifier block arguments.

    Function, CFG, structured-control-flow, and unknown region boundaries are
    otherwise preserved. The pass is linear in the number of visited
    operations and does not scan an enclosing module from individual rewrite
    patterns.
  }];
}

#endif // MLIR_DIALECT_UTILS_TRANSFORMS_PASSES_TD
