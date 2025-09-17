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
   * @brief Checks if the uses of a qubit allocated by `allocQubit` is disjoint
   * from the uses of a qubit involved in a deallocation.
   * @param allocQubit The allocated qubit to check.
   * @param dealloc The deallocation operation to check against.
   * @return `true` if the qubits are disjoint, `false` otherwise.
   */
  static bool areQubitsDisjoint(mlir::Value allocQubit,
                                DeallocQubitOp dealloc) {
    // If traversing the def-use chain from the "alloc" qubit never reaches the
    // "dealloc" qubit, they are disjoint.
    llvm::SmallVector<mlir::Operation*> toVisit{allocQubit.getUsers().begin(),
                                                allocQubit.getUsers().end()};
    while (!toVisit.empty()) {
      auto* current = toVisit.back();
      toVisit.pop_back();

      // If we reach the dealloc operation, the qubits are not disjoint.
      if (current == dealloc) {
        return false;
      }

      if (auto yieldOp = mlir::dyn_cast<mlir::scf::YieldOp>(current)) {
        // If we reach a yield operation, we continue from the corresponding
        // `parent`.
        toVisit.push_back(yieldOp->getParentOp());
        continue;
      }

      // Add all users of the current operation to the visit list.
      for (auto* user : current->getUsers()) {
        toVisit.push_back(user);
      }
    }

    return true;
  }

  /**
   * @brief Reorders the users of the given operation to ensure that they are
   * after it.
   * @param op The operation whose users should be reordered.
   * @param rewriter The pattern rewriter to use for moving operations.
   */
  static void reorderUsers(mlir::Operation* op,
                           mlir::PatternRewriter& rewriter) {
    for (auto* user : op->getUsers()) {
      // Move the user operation after the allocation operation.

      while (op->getBlock() != user->getBlock()) {
        user = user->getParentOp();
      }
      if (op->isBeforeInBlock(user)) {
        continue; // Already in the correct order.
      }
      rewriter.moveOpAfter(user, op);

      reorderUsers(user, rewriter);
    }
  }

  /**
   * @brief Rewrites the given `AllocQubitOp` and `DeallocQubitOp` to reuse the
   * qubit instead.
   *
   * @param alloc The allocation that will be replaced by qubit reuse.
   * @param dealloc The deallocation that will be replaced by a reset operation.
   * @param rewriter The pattern rewriter to use for the rewrite.
   */
  static void rewriteForReuse(AllocQubitOp alloc, DeallocQubitOp dealloc,
                              mlir::PatternRewriter& rewriter) {
    rewriter.setInsertionPointAfter(dealloc);
    auto reset = rewriter.replaceOpWithNewOp<ResetOp>(
        alloc, alloc.getQubit().getType(), dealloc.getQubit());
    rewriter.eraseOp(dealloc);

    reorderUsers(reset, rewriter);
  }

  mlir::LogicalResult
  matchAndRewrite(AllocQubitOp op,
                  mlir::PatternRewriter& rewriter) const override {
    auto* currentBlock = op->getBlock();
    auto deallocs = currentBlock->getOps<DeallocQubitOp>();
    auto reusable = llvm::find_if(deallocs, [&](DeallocQubitOp dealloc) {
      // Check if the qubit to be deallocated is disjoint from the qubit to be
      // allocated.
      return areQubitsDisjoint(op.getQubit(), dealloc);
    });

    if (reusable == deallocs.end()) {
      // No reusable qubit found, nothing to do.
      return mlir::failure();
    }

    rewriteForReuse(op, *reusable, rewriter);
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
