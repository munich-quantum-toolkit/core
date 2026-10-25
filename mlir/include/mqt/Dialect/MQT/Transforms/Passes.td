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

def PromoteClassicalMemory
    : Pass<"promote-classical-memory", "mlir::ModuleOp"> {
  let summary = "Promote local classical memory to SSA values";
  let description = [{
    Folds access indices, splits local allocations with SROA, and promotes
    supported loads and stores with Mem2Reg. Local CBit registers support up to
    256 bits with static accesses. Packed reads support up to 64 bits; wider
    reads are expanded at their snapshot point when all reads of the register
    feed constant comparisons. Escaping registers retain storage.

    Modules without promotable or destructurable allocations are left unchanged.
    Run again after static loop expansion to expose constant-index accesses.
  }];
}

def SimplifyClassicalControl
    : Pass<"simplify-classical-control", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::arith::ArithDialect"];
  let summary = "Propagate known QCO branch conditions";
  let description = [{
    Within each QCO if branch, replaces uses of its condition with the known
    boolean value. A following canonicalizer folds the exposed expressions and
    static branches. Run after inlining and again after static loop unrolling.
  }];
}

def NormalizeGlobalPhases : Pass<"normalize-global-phases", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::math::MathDialect", "mlir::qc::QCDialect",
                           "mlir::qco::QCODialect"];
  let summary = "Normalize scoped QC and QCO global phases";
  let description = [{
    Combines direct `qc.gphase` and `qco.gphase` operations into at most one
    operation per basic block, placed immediately before the block terminator.
    Nested regions are normalized independently, except power subtrees, which
    remain unchanged: phase rounding can be amplified by a large exponent.

    A normalized phase is moved out of `inv`, negating its angle. A phase in a
    `ctrl` body becomes a relative phase on the control register: `p` for one
    control or a smaller controlled `p` for multiple controls. Eager classical
    computations in verified modifier bodies move once, in source order, before
    the outermost enclosing `inv` or `ctrl`.

    Constant angles are reduced modulo 2π. Symbolic sums are balanced and
    reduced with `2 * atan(tan(angle / 2))` to keep runtime phases in range.
    Function, CFG, structured-control-flow, and unknown region boundaries are
    preserved. The pass does not scan an enclosing module from individual
    rewrite patterns.
  }];
}

def UnrollModifiers : Pass<"unroll-modifiers", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::qc::QCDialect", "mlir::qco::QCODialect"];
  let summary = "Unroll multi-operation modifiers into single-operation ones";
  let description = [{
    Expands unitary function calls inside quantum modifiers and normalizes
    exposed global phases before splitting modifiers. Calls outside modifiers
    remain reusable. Functions marked `no_inline` remain calls.

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
    modifiers are skipped silently.
  }];
}

#endif // MLIR_DIALECT_MQT_TRANSFORMS_PASSES_TD
