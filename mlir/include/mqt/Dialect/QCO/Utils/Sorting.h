/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mlir/IR/Block.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLFunctionalExtras.h"

namespace mlir::qco {
/// Check whether CBit or MemRef effects may access aliased storage.
///
/// Direct allocations own distinct storage; other handles may alias them.
bool hasPotentialClassicalAliases(Block& block);

/// Visit SSA, nested-capture, and ordered memory-effect dependencies.
///
/// Calls the visitor with predecessor and successor operations in this block.
/// Edges may repeat. Potentially aliased storage shares one effect chain;
/// unknown and value-less effects are not ordered.
void walkOperationDependencies(
    Block& block, function_ref<void(Operation*, Operation*)> addDependency);

/// Fix SSA dominance issues by reordering operations of the block in-place in
/// topological order.
///
/// Assumes that the block is acyclic. Historically this replaced MLIR's
/// `sortTopologically` due to significant runtime overhead. Preserve
/// deterministic discovery order for ready operations. Keep SSA dependencies
/// and order effects on each SSA value. Preserve CBit and MemRef effect order
/// across handles when aliases may occur. Unknown and value-less effects are
/// not ordered.
void reorderTopologically(Block& block, IRRewriter& rewriter);
} // namespace mlir::qco
