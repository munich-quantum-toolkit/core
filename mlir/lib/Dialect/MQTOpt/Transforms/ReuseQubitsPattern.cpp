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
   * from the uses of a qubit involved in a sink (dealloc or reset) operation.
   * @param allocQubit The allocated qubit to check.
   * @param sink The sink operation that trashes a qubit.
   * @return `true` if the qubits are disjoint, `false` otherwise.
   */
  static llvm::DenseSet<mlir::Operation*>
  areQubitsDisjoint(mlir::Value allocQubit, mlir::Operation* sink,
                    llvm::DenseSet<mlir::Operation*>& nonDisjointGates) {
    // If traversing the def-use chain from the "alloc" qubit never reaches the
    // "dealloc" qubit, they are disjoint.
    llvm::DenseSet<mlir::Operation*> reachableSinks;
    llvm::SmallVector<mlir::Operation*> toVisit{allocQubit.getUsers().begin(),
                                                allocQubit.getUsers().end()};
    llvm::DenseSet<mlir::Operation*> visited;
    while (!toVisit.empty()) {
      auto* current = toVisit.back();
      toVisit.pop_back();
      visited.insert(current);

      // If we reach the dealloc operation, the qubits are not disjoint.
      if (mlir::isa<DeallocQubitOp>(current)) {
        reachableSinks.insert(current);
      }

      if (auto yieldOp = mlir::dyn_cast<mlir::scf::YieldOp>(current)) {
        // If we reach a yield operation, we continue from the corresponding
        // `parent`.
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

  static DeallocQubitOp findDealloc(mlir::Value qubit) {
    mlir::Value currentQubit = qubit;
    mlir::Operation* current = *qubit.getUsers().begin();
    unsigned int operandIndex = 0;
    while (!mlir::isa<DeallocQubitOp>(current)) {
      if (auto unitaryOp = mlir::dyn_cast<UnitaryInterface>(current)) {
        currentQubit = unitaryOp.getCorrespondingOutput(currentQubit);
        auto& use = *currentQubit.getUses().begin();
        current = use.getOwner();
        operandIndex = use.getOperandNumber();
        continue;
      }
      if (auto yieldOp = mlir::dyn_cast<mlir::scf::YieldOp>(current)) {
        // current becomes the corresponding output of the parent `scf.if`
        auto* parent = yieldOp->getParentOp();
        currentQubit = parent->getResult(operandIndex);
        auto& use = *currentQubit.getUses().begin();
        current = use.getOwner();
        operandIndex = use.getOperandNumber();
      }
      currentQubit =
          *llvm::find_if(current->getResults(), [&](mlir::Value result) {
            return mlir::isa<QubitType>(result.getType());
          });
      auto& use = *currentQubit.getUses().begin();
      current = use.getOwner();
      operandIndex = use.getOperandNumber();
    }
    return mlir::dyn_cast<DeallocQubitOp>(current);
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
      // llvm::outs() << "Moving operation before " <<
      // user->getName().getStringRef().str() << "(" << user << ")\n";

      reorderUsers(user, rewriter);
    }
  }

  /**
   * @brief Rewrites the given `AllocQubitOp` and `DeallocQubitOp` to reuse the
   * qubit instead.
   *
   * @param alloc The allocation that will be replaced by qubit reuse.
   * @param sink The deallocation/reset that will be replaced by a new reset
   * operation.
   * @param rewriter The pattern rewriter to use for the rewrite.
   */
  static void rewriteForReuse(AllocQubitOp alloc, mlir::Operation* sink,
                              mlir::PatternRewriter& rewriter) {
    // alloc.print(llvm::outs());
    // llvm::outs() << "\n";
    //     sink->print(llvm::outs());
    //     llvm::outs() << "\n";
    rewriter.setInsertionPointAfter(sink);
    const mlir::Value originalInput = sink->getOperand(0);
    auto reset = rewriter.replaceOpWithNewOp<ResetOp>(
        alloc, alloc.getQubit().getType(), originalInput);
    if (auto originalReset = mlir::dyn_cast<ResetOp>(sink)) {
      // The replaced operation is a Reset.
      // This means it has an output which should be placed after
      // the `dealloc` of the qubit use chain it is replaced by.
      DeallocQubitOp dealloc = findDealloc(reset.getOutQubit());
      rewriter.setInsertionPoint(dealloc);
      auto newReset = rewriter.replaceOpWithNewOp<ResetOp>(
          dealloc, originalReset.getOutQubit().getType(), dealloc.getQubit());
      rewriter.replaceAllUsesWith(originalReset.getOutQubit(),
                                  newReset.getOutQubit());
      reorderUsers(newReset, rewriter);
    } else {
      rewriter.eraseOp(sink);
    }
    // TODO this probably shouldn't be recursive
    reorderUsers(reset, rewriter);
  }

  mlir::LogicalResult
  matchAndRewrite(AllocQubitOp op,
                  mlir::PatternRewriter& rewriter) const override {
    auto* currentBlock = op->getBlock();
    auto deallocs = currentBlock->getOps<DeallocQubitOp>();
    llvm::DenseSet<mlir::Operation*> nonDisjointGates;
    llvm::DenseSet<mlir::Operation*> reachableSinks =
        areQubitsDisjoint(op.getQubit(), nullptr, nonDisjointGates);
    auto reusableDeallocs =
        llvm::find_if(llvm::reverse(deallocs), [&](DeallocQubitOp dealloc) {
          // Check if the qubit to be deallocated is disjoint from the qubit to
          // be allocated.
          // return areQubitsDisjoint(op.getQubit(), dealloc, nonDisjointGates);
          return !reachableSinks.contains(dealloc);
        });

    mlir::Operation* reusable = nullptr;

    if (reusableDeallocs == llvm::reverse(deallocs).end()) {
      return mlir::failure();
      // No reusable dealloc found, check resets next.
      /*auto reset = currentBlock->getOps<ResetOp>();
      auto reusableResets = llvm::find_if(reset, [&](ResetOp reset) {
        // Check if the qubit to be reset is disjoint from the qubit to be
        // allocated.
        return areQubitsDisjoint(op.getQubit(), reset, nonDisjointGates);
      });
      if (reusableResets == reset.end()) {
        // No reusable qubit found.
        return mlir::failure();
      }
      reusable = *reusableResets;*/
    } else {
      reusable = *reusableDeallocs;
    }

    rewriteForReuse(op, reusable, rewriter);
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
