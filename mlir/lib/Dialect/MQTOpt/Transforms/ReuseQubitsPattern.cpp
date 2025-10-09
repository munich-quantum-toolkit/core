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

  explicit ReuseQubitsPattern(mlir::MLIRContext* context)
      : OpRewritePattern(context) {}

  /**
   * @brief Finds all reachable `DeallocQubitOp` operation starting from some
   * qubit.
   * @param allocQubit The starting qubit to check (e.g. a newly  allocated
   * qubit).
   * @return A set of all DeallocQubitOp operations reachable from the given
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
   * @brief Reorders the users of the given operation to ensure that they are
   * after it.
   * @param op The operation whose users should be reordered.
   * @param rewriter The pattern rewriter to use for moving operations.
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
   * @brief Rewrites the given `AllocQubitOp` and `DeallocQubitOp` to reuse the
   * qubit instead.
   *
   * @param alloc The allocation that will be replaced by qubit reuse.
   * @param sink The deallocation that will be replaced by a new reset
   * operation.
   * @param rewriter The pattern rewriter to use for the rewrite.
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
 * @brief Populates the given pattern set with the
 * `ReuseQubitsPattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateReuseQubitsPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add<ReuseQubitsPattern>(patterns.getContext());
}

} // namespace mqt::ir::opt
