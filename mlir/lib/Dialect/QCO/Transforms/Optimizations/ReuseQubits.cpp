/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <cassert>
#include <iterator>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_REUSEQUBITS
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

namespace {
/**
 * @brief This is the main qubit reuse pattern.
 */
struct ReuseQubitsPattern final : mlir::OpRewritePattern<AllocOp> {

  explicit ReuseQubitsPattern(mlir::MLIRContext* context)
      : OpRewritePattern(context) {}

  /**
   * @brief Finds all reachable `SinkOp` operation starting from some
   * qubit.
   * @param allocQubit The starting qubit to check (e.g. a newly  allocated
   * qubit).
   * @return A set of all SinkOp operations reachable from the given
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
      if (mlir::isa<SinkOp>(current)) {
        reachableSinks.insert(current);
        continue;
      }

      if (auto yieldOp = mlir::dyn_cast<qco::YieldOp>(current)) {
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
   * @brief Rewrites the given `AllocOp` and `SinkOp` to reuse the
   * qubit instead.
   *
   * @param alloc The allocation that will be replaced by qubit reuse.
   * @param sink The sink that will be replaced by a new reset
   * operation.
   * @param rewriter The pattern rewriter to use for the rewrite.
   */
  static void rewriteForReuse(AllocOp alloc, mlir::Operation* sink,
                              mlir::PatternRewriter& rewriter) {
    rewriter.setInsertionPointAfter(sink);
    const mlir::Value originalInput = sink->getOperand(0);
    auto reset = rewriter.replaceOpWithNewOp<ResetOp>(
        alloc, alloc.getResult().getType(), originalInput);
    rewriter.eraseOp(sink);

    reorderUsers(reset, rewriter);
  }

  mlir::LogicalResult
  matchAndRewrite(AllocOp op, mlir::PatternRewriter& rewriter) const override {
    // Find all `SinkOp` operations in the current block and check
    // if any of them are disjoint from the qubit being allocated, indicating
    // potential for reuse.

    auto* currentBlock = op->getBlock();
    auto deallocs = currentBlock->getOps<SinkOp>();
    llvm::DenseSet<mlir::Operation*> reachableDeallocs =
        findAllReachableDeallocs(op.getResult());
    // We search `reverse(deallocs)` rather than `deallocs` because this tends
    // to give more readable results.
    auto reusableDeallocs =
        llvm::find_if(llvm::reverse(deallocs), [&](SinkOp dealloc) {
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
 * @brief This pass searches for qubits that do not interact with each other
 * directly or indirectly and attempts to reset and reuse one of them for the
 * other.
 */
struct ReuseQubits final : impl::ReuseQubitsBase<ReuseQubits> {
  using ReuseQubitsBase::ReuseQubitsBase;

protected:
  void runOnOperation() override {
    const auto op = getOperation();
    auto* ctx = &getContext();

    // Define the set of patterns to use.
    RewritePatternSet patterns(ctx);
    patterns.add<ReuseQubitsPattern>(patterns.getContext());

    // Apply patterns in an iterative and greedy manner.
    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::qco
