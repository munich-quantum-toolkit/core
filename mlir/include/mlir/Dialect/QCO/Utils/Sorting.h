#pragma once

#include <mlir/IR/Block.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir::qco {
/// Fix SSA dominance issues by reordering operations of the block in-place in
/// topological order. Assumes that the block is acyclic. Historically this
/// replaced MLIR's `sortTopologically` due to significant runtime overhead.
void reorderTopologically(Block& block, IRRewriter& rewriter);
} // namespace mlir::qco