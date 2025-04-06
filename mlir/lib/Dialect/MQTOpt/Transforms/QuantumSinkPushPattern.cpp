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

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <unordered_set>
#include <vector>

namespace mqt::ir::opt {

/**
 * @brief This pattern is responsible to push Unitary operations through branch
 * operations into new blocks.
 *
 * This can be done in some circumstances, if a branch operation uses the result
 * of a unitary operation as a block argument.
 */
struct QuantumSinkPushPattern final
    : mlir::OpInterfaceRewritePattern<UnitaryInterface> {

  explicit QuantumSinkPushPattern(mlir::MLIRContext* context)
      : OpInterfaceRewritePattern(context) {}

  /**
   * @brief Returns the block distance between a block and an operation that
   * precedes it.
   *
   * @param toCheck The block to check.
   * @param op The operation to find the distance to.
   * @return The distance between the block and the operation or -1 if the
   * operation is not in a predecessor block.
   */
  int getDepth(mlir::Block& toCheck, const UnitaryInterface& op) const {
    auto* originalBlock = op->getBlock();
    if (&toCheck == originalBlock) {
      return 0;
    }
    int depth = -1;
    for (auto* pred : toCheck.getPredecessors()) {
      const auto result = getDepth(*pred, op);
      if (result == -1) {
        continue;
      }
      if (depth == -1) {
        depth = result + 1;
      } else {
        depth = std::min(depth, result + 1);
      }
    }
    return depth;
  }

  /**
   * @brief Out of a set of users, returns the closest one.
   *
   * In this case, "closest" denotes the operation that is in the closest
   * successor block of the original operation.
   *
   * @param users The set of users to consider.
   * @param op The operation to find the closest user for.
   * @return The closest user operation.
   */
  [[nodiscard]] mlir::Operation*
  getNext(const mlir::ResultRange::user_range& users,
          const UnitaryInterface& op) const {
    mlir::Operation* next = nullptr;
    int minDepth = 0;
    for (auto* user : users) {
      const auto depth = getDepth(*user->getBlock(), op);
      if (depth == -1) {
        continue;
      }
      if (next == nullptr || depth < minDepth) {
        next = user;
        minDepth = depth;
      }
    }
    return next;
  }

  /**
   * @brief Returns the next branch operation that uses the given operation.
   *
   * In this case, "next" denotes the operation that is in the closest successor
   * block of the original operation.
   *
   * @param op The operation to find the next branch operation for.
   * @return The next branch operation that uses the given operation.
   */
  [[nodiscard]] mlir::Operation*
  getNextBranchOpUser(const UnitaryInterface& op) const {
    auto allUsers = op->getUsers();
    std::vector<mlir::Operation*> output;
    std::copy_if(allUsers.begin(), allUsers.end(), std::back_inserter(output),
                 [&](auto* user) {
                   return mlir::isa<mlir::cf::BranchOp>(user) ||
                          mlir::isa<mlir::cf::CondBranchOp>(user);
                 });
    auto* nextBranch = getNext(allUsers, op);
    return nextBranch;
  }

  mlir::LogicalResult match(UnitaryInterface op) const override {
    // We only consider 1-qubit gates.
    if (op.getOutQubits().size() != 1) {
      return mlir::failure();
    }

    const auto& users = op->getUsers();
    if (users.empty()) {
      return mlir::failure();
    }

    auto* nextBranch = getNextBranchOpUser(op);
    if (nextBranch == nullptr) {
      return mlir::failure();
    }

    // This pattern only applies to operations in the same block.
    if (nextBranch->getBlock() != op->getBlock()) {
      return mlir::failure();
    }

    // The pattern can always be applied if the user is a conditional branch
    // operation.
    if (mlir::isa<mlir::cf::CondBranchOp>(nextBranch)) {
      return mlir::success();
    }

    // For normal branch operations, we can only apply the pattern if the
    // successor block only has one predecessor.
    if (auto branchOp = mlir::dyn_cast<mlir::cf::BranchOp>(nextBranch)) {
      auto* successor = branchOp.getDest();
      if (std::distance(successor->getPredecessors().begin(),
                        successor->getPredecessors().end()) == 1) {
        return mlir::success();
      }
    }
    return mlir::failure();
  }

  /**
   * @brief Replaces all uses of a given operation with the results of a clone,
   * but only in the current block and all successor blocks.
   *
   * This differs from MLIR's `Operation::replaceAllUsesWith` in that it does
   * not replace uses in parallel blocks.
   *
   * @param original The original operation to replace.
   * @param clone The clone to replace with.
   * @param block The block to start replacing in.
   * @param visited A set of blocks that have already been visited to prevent
   * endless loops.
   */
  void
  replaceAllChildUsesWith(mlir::Operation& original, mlir::Operation& clone,
                          mlir::Block& block,
                          std::unordered_set<mlir::Block*>& visited) const {
    if (visited.find(&block) != visited.end()) {
      return;
    }
    visited.insert(&block);

    for (auto& op : block.getOperations()) {
      if (&op == &original) {
        continue;
      }
      for (size_t i = 0; i < original.getResults().size(); i++) {
        op.replaceUsesOfWith(original.getResult(i), clone.getResult(i));
      }
    }

    for (auto* successor : block.getSuccessors()) {
      replaceAllChildUsesWith(original, clone, *successor, visited);
    }
  }

  /**
   * @brief Pushes the given Unitary operation through a branch operation into
   * the given block.
   *
   * @param op The Unitary operation to push.
   * @param block The block to push the operation into.
   * @param blockArgs The arguments used to call the block.
   * @param rewriter The pattern rewriter to use.
   * @return The new arguments to call the block with.
   */
  std::vector<mlir::Value>
  pushIntoBlock(const UnitaryInterface& op, mlir::Block& block,
                const mlir::OperandRange blockArgs,
                mlir::PatternRewriter& rewriter) const {
    std::vector<mlir::Value> newBlockArgs;
    // We start by inserting the operation at the beginning of the block.
    rewriter.setInsertionPointToStart(&block);
    auto* clone = rewriter.clone(*op);

    // We iterate over all block args: If any of them involved the result of the
    // pushed operation, they can be removed, as the operation is now already in
    // the block.
    for (int i = static_cast<int>(blockArgs.size()) - 1; i >= 0; i--) {
      auto arg = blockArgs[i];
      auto found =
          std::find(op->getResults().begin(), op->getResults().end(), arg);
      if (found == op->getResults().end()) {
        newBlockArgs.emplace_back(arg);
        continue;
      }
      auto idx = std::distance(op->getResults().begin(), found);
      block.getArgument(i).replaceAllUsesWith(clone->getResult(idx));
      block.eraseArgument(i);
    }

    std::unordered_set<mlir::Block*> visited;
    replaceAllChildUsesWith(*op, *clone, block, visited);

    // Now, the block instead needs to be passed the inputs of the pushed
    // operation so we extend the block with new arguments.
    for (size_t i = 0; i < clone->getOperands().size(); i++) {
      auto in = clone->getOperand(i);
      auto newArg = block.addArgument(in.getType(), clone->getLoc());
      rewriter.modifyOpInPlace(clone, [&]() { clone->setOperand(i, newArg); });
      newBlockArgs.emplace_back(in);
    }

    return newBlockArgs;
  }

  /**
   * @brief Breaks a critical edge through the given branch operation by adding
   * a new block.
   *
   * @param oldTarget The target block to which the critical edge went.
   * @param branchOp The branch operation that caused the critical edge.
   * @param rewriter The pattern rewriter to use.
   * @return The new block that was created.
   */
  static mlir::Block* breakCriticalEdge(mlir::Block& oldTarget,
                                        mlir::Operation& branchOp,
                                        mlir::PatternRewriter& rewriter) {
    auto* newBlock = rewriter.createBlock(oldTarget.getParent());
    std::vector<mlir::Value> newBlockOutputs;
    newBlockOutputs.reserve(oldTarget.getNumArguments());
    for (auto arg : oldTarget.getArguments()) {
      auto newArg = newBlock->addArgument(arg.getType(), branchOp.getLoc());
      newBlockOutputs.emplace_back(newArg);
    }
    rewriter.setInsertionPointToEnd(newBlock);
    rewriter.create<mlir::cf::BranchOp>(branchOp.getLoc(), &oldTarget,
                                        newBlockOutputs);
    return newBlock;
  }

  /**
   * @brief Pushes a Unitary operation through a single conditional branch
   * operation.
   *
   * @param op The Unitary operation to push.
   * @param condBranchOp The conditional branch operation to push through.
   * @param rewriter The pattern rewriter to use.
   */
  void rewriteCondBranch(const UnitaryInterface& op,
                         mlir::cf::CondBranchOp condBranchOp,
                         mlir::PatternRewriter& rewriter) const {
    auto* targetBlockTrue =
        std::distance(condBranchOp.getTrueDest()->getPredecessors().begin(),
                      condBranchOp.getTrueDest()->getPredecessors().end()) == 1
            ? condBranchOp.getTrueDest()
            : breakCriticalEdge(*condBranchOp.getTrueDest(), *condBranchOp,
                                rewriter);
    auto newBlockArgsTrue = pushIntoBlock(
        op, *targetBlockTrue, condBranchOp.getTrueOperands(), rewriter);

    auto* targetBlockFalse =
        std::distance(condBranchOp.getFalseDest()->getPredecessors().begin(),
                      condBranchOp.getFalseDest()->getPredecessors().end()) == 1
            ? condBranchOp.getFalseDest()
            : breakCriticalEdge(*condBranchOp.getFalseDest(), *condBranchOp,
                                rewriter);
    auto newBlockArgsFalse = pushIntoBlock(
        op, *targetBlockFalse, condBranchOp.getFalseOperands(), rewriter);

    rewriter.setInsertionPoint(condBranchOp);
    rewriter.replaceOpWithNewOp<mlir::cf::CondBranchOp>(
        condBranchOp, condBranchOp.getCondition(), targetBlockTrue,
        newBlockArgsTrue, targetBlockFalse, newBlockArgsFalse);

    rewriter.eraseOp(op);
  }

  /**
   * @brief Pushes a Unitary operation through a single branch operation.
   *
   * The successor of this branch will always have exactly one predecessor as
   * per the match function. Therefore, we do not have to worry about critical
   * edges.
   *
   * @param op The Unitary operation to push.
   * @param branchOp The branch operation to push through.
   * @param rewriter The pattern rewriter to use.
   */
  void rewriteBranch(const UnitaryInterface& op, mlir::cf::BranchOp& branchOp,
                     mlir::PatternRewriter& rewriter) const {
    auto newBlockArgs = pushIntoBlock(op, *branchOp.getDest(),
                                      branchOp.getOperands(), rewriter);
    rewriter.modifyOpInPlace(branchOp, [&]() {
      branchOp.getDestOperandsMutable().assign(newBlockArgs);
    });
    rewriter.eraseOp(op);
  }

  void rewrite(UnitaryInterface op,
               mlir::PatternRewriter& rewriter) const override {
    auto* user = getNextBranchOpUser(op);
    if (auto condBranchOp = mlir::dyn_cast<mlir::cf::CondBranchOp>(user)) {
      rewriteCondBranch(op, condBranchOp, rewriter);
    } else {
      auto branchOp = mlir::dyn_cast<mlir::cf::BranchOp>(user);
      rewriteBranch(op, branchOp, rewriter);
    }
  }
};

/**
 * @brief Populates the given pattern set with the
 * `QuantumSinkPushPattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateQuantumSinkPushPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add<QuantumSinkPushPattern>(patterns.getContext());
}

} // namespace mqt::ir::opt
