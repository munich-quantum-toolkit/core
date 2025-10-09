/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

namespace mqt::ir::opt {

/**
 * @brief This pattern is responsible applying qubit reuse.
 */
struct ReuseQubitsPattern final : mlir::OpRewritePattern<AllocQubitOp> {

  /**
   * @brief Constructs a ReuseQubitsPattern and registers it with the given MLIR
   * context.
   *
   * @param context MLIR context used to initialize the underlying
   * OpRewritePattern.
   */
  explicit ReuseQubitsPattern(mlir::MLIRContext* context)
      : OpRewritePattern(context) {}

  /**
   * @brief Collects all deallocation operations that can be reached from a
   * qubit value.
   *
   * Traverses uses of the provided qubit value and follows control-flow yields
   * into parent ops (for example, scf.if regions) to discover reachable
   * DeallocQubitOp operations.
   *
   * @param allocQubit The qubit value whose reachable deallocations are to be
   * found.
   * @return llvm::DenseSet<mlir::Operation*> Set of `DeallocQubitOp` operations
   * reachable from `allocQubit`.
   */
  static llvm::DenseSet<mlir::Operation*>
  findAllReachableDeallocs(mlir::Value allocQubit) {
    // Traverse def-use chain using BFS.
    llvm::DenseSet<mlir::Operation*> reachableSinks;
    llvm::SmallVector<mlir::Operation*> toVisit{allocQubit.getUsers().begin(),
                                                allocQubit.getUsers().end()};
    llvm::DenseSet<mlir::Operation*> visited;
    while (!toVisit.empty()) {
      auto* current = toVisit.back();
      toVisit.pop_back();
      visited.insert(current);

      // If we reach the dealloc operation, we add it to the list of sinks.
      if (mlir::isa<DeallocQubitOp>(current)) {
        reachableSinks.insert(current);
      }

      if (auto yieldOp = mlir::dyn_cast<mlir::scf::YieldOp>(current)) {
        // If we reach a yield operation, we continue from the corresponding
        // `parent` (e.g. `scf.if`).
        toVisit.push_back(yieldOp->getParentOp());
        continue;
      }

      // Add all users of the current operation to the visit list.
      for (auto result : current->getResults()) {
        for (auto* user : result.getUsers()) {
          if (visited.contains(user)) {
            continue;
          }
          toVisit.push_back(user);
        }
      }
    }

    return reachableSinks;
  }

  /**
   * @brief Ensures all users reachable from the given operation appear after it
   * in their blocks.
   *
   * Traverses user operations reachable from @p startingOp and reorders them so
   * each user is placed after the operation it depends on, updating block-local
   * operation order via the provided rewriter.
   *
   * @param startingOp Operation from which reachable user operations are
   * reordered.
   * @param rewriter Pattern rewriter used to move operations.
   */
  static void reorderUsers(mlir::Operation* startingOp,
                           mlir::PatternRewriter& rewriter) {
    // Search for operations that need re-ordering using BFS.
    mlir::DenseSet<mlir::Operation*> toVisit{startingOp};
    mlir::DenseSet<mlir::Operation*> visited;

    while (!toVisit.empty()) {
      auto* op = *toVisit.begin();
      toVisit.erase(op);
      visited.insert(op);
      for (auto* user : op->getUsers()) {
        // Move the user operation after the current operation.

        while (op->getBlock() != user->getBlock()) {
          user = user->getParentOp();
        }
        if (op->isBeforeInBlock(user)) {
          continue; // Already in the correct order.
        }
        rewriter.moveOpAfter(user, op);

        if (!visited.contains(user)) {
          toVisit.insert(user);
        }
      }
    }
  }

  /**
   * @brief Replace an allocation with a reset to reuse an existing qubit and
   * remove the corresponding deallocation.
   *
   * Replaces `alloc` with a `ResetOp` that reuses the qubit previously freed by
   * `sink`, erases `sink`, and reorders affected users to preserve block
   * ordering.
   *
   * @param alloc The AllocQubitOp to be replaced.
   * @param sink The DeallocQubitOp whose freed qubit will be reused
   * (replaced/erased).
   * @param rewriter Pattern rewriter used to perform the replacement and
   * reorder operations.
   */
  static void rewriteForReuse(AllocQubitOp alloc, mlir::Operation* sink,
                              mlir::PatternRewriter& rewriter) {
    rewriter.setInsertionPointAfter(sink);
    const mlir::Value originalInput = sink->getOperand(0);
    auto reset = rewriter.replaceOpWithNewOp<ResetOp>(
        alloc, alloc.getQubit().getType(), originalInput);
    rewriter.eraseOp(sink);

    reorderUsers(reset, rewriter);
  }

  /**
   * Attempts to reuse an existing qubit deallocation in the same block for the
   * given AllocQubitOp by replacing the allocation with a reuse pattern.
   *
   * Searches deallocations in the current block and, if a deallocation that is
   * disjoint from the alloc's reachable deallocations is found, rewrites the
   * allocation to reuse that deallocated qubit via the provided rewriter.
   *
   * @param op The AllocQubitOp to match and potentially rewrite.
   * @param rewriter PatternRewriter used to perform the transformation.
   * @return `mlir::success()` if the alloc was rewritten to reuse an existing
   * deallocation, `mlir::failure()` otherwise.
   */
  mlir::LogicalResult
  matchAndRewrite(AllocQubitOp op,
                  mlir::PatternRewriter& rewriter) const override {
    // Find all `DeallocQubitOp` operations in the current block and check
    // if any of them are disjoint from the qubit being allocated, indicating
    // potential for reuse.

    auto* currentBlock = op->getBlock();
    auto deallocs = currentBlock->getOps<DeallocQubitOp>();
    llvm::DenseSet<mlir::Operation*> reachableDeallocs =
        findAllReachableDeallocs(op.getQubit());
    // We search `reverse(deallocs)` rather than `deallocs` because this tends
    // to give more readable results.
    auto reusableDeallocs =
        llvm::find_if(llvm::reverse(deallocs), [&](DeallocQubitOp dealloc) {
          // Check if the qubit to be deallocated is disjoint from the qubit to
          // be allocated.
          return !reachableDeallocs.contains(dealloc);
        });

    if (reusableDeallocs == llvm::reverse(deallocs).end()) {
      return mlir::failure();
      // No reusable dealloc found.
      // We could also check `reset` operations next, which would
      // always result in the optimal solution, but the complexity explodes.
      // Therefore, we only check for deallocs here.
    }

    rewriteForReuse(op, *reusableDeallocs, rewriter);
    return mlir::success();
  }
};

/**
 * @brief Register the ReuseQubitsPattern into a rewrite pattern set.
 *
 * @param patterns Pattern set to which the ReuseQubitsPattern will be added.
 */
void populateReuseQubitsPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add<ReuseQubitsPattern>(patterns.getContext());
}

} // namespace mqt::ir::opt
