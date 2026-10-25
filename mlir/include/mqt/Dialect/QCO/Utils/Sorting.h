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

namespace mlir::qco {
/// Check whether CBit effects access handles other than direct allocations.
///
/// Allocations own distinct storage; other register handles may alias them.
bool hasPotentialClassicalAliases(Block& block);

/// Fix SSA dominance issues by reordering operations of the block in-place in
/// topological order.
///
/// Assumes that the block is acyclic. Historically this replaced MLIR's
/// `sortTopologically` due to significant runtime overhead. Preserve
/// deterministic discovery order for ready operations. Keep SSA dependencies
/// and order effects on each SSA value. Preserve CBit effect order across
/// registers when aliases may occur. Unknown and value-less effects are not
/// ordered.
void reorderTopologically(Block& block, IRRewriter& rewriter);
} // namespace mlir::qco
