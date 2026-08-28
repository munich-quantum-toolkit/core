// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_MQT_TRANSFORMS_PASSES_TD
#define MLIR_DIALECT_MQT_TRANSFORMS_PASSES_TD

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

def UnrollModifiers : Pass<"unroll-modifiers", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::qc::QCDialect", "mlir::qco::QCODialect"];
  let summary = "Unroll multi-operation modifiers into single-operation ones";
  let description = [{
    Splits `ctrl` and `inv` modifiers whose body holds more than one unitary
    operation into a sequence of modifiers that each hold a single operation.
    For `inv`, the order of the operations is reversed. Classical operations of
    the body are moved in front of the new modifiers.

    A `pow` modifier is only unrolled if its exponent is a compile-time known
    integer and its operations act on disjoint qubits. Otherwise, `pow(r) { a;
    b }` is generally not equivalent to `pow(r) { a }; pow(r) { b }`: the
    operations do not commute if they share a qubit, and for a non-integer
    exponent, the principal branch of the matrix power does not distribute over
    them. Modifiers nested in a `pow` body are unrolled either way.

    If a modifier cannot be unrolled, the modifier is left untouched. Such
    modifiers are skipped silently. The pass fails before mutation when QCO
    linearity is violated.
  }];
}

#endif // MLIR_DIALECT_MQT_TRANSFORMS_PASSES_TD
