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
 * @brief This pattern is responsible to shift Unitary operations into the block
 * their results are used in.
 */
struct QuantumSinkShiftPattern final
    : mlir::OpInterfaceRewritePattern<UnitaryInterface> {

  explicit QuantumSinkShiftPattern(mlir::MLIRContext* context)
      : OpInterfaceRewritePattern(context) {}

  /**
   * @brief Recursively checks if a block is a successor of another block or the
   * same block.
   *
   * @param successor The block that might be after the other block.
   * @param predecessor The block to check against.
   */
  bool isAfterOrEqual(mlir::Block& successor, mlir::Block& predecessor,
                      std::unordered_set<mlir::Block*>& visited) const {
    if (visited.contains(&successor)) {
      return false;
    }
    visited.insert(&successor);
    if (&successor == &predecessor) {
      return true;
    }
    auto parents = successor.getPredecessors();
    return std::any_of(parents.begin(), parents.end(), [&](auto* parent) {
      return isAfterOrEqual(*parent, predecessor, visited);
    });
  }

  /**
   * @brief Returns the earliest users of a given operation.
   *
   * In this case, "earliest" denotes all operations that are not preceded
   * another operation in the list of users.
   *
   * @param users The users to find the earliest users for.
   * @return A vector of all earliest users of the given operation.
   */
  [[nodiscard]] std::vector<mlir::Operation*>
  getEarliestUsers(mlir::ResultRange::user_range users) const {
    std::vector<mlir::Operation*> earliestUsers;
    std::copy_if(
        users.begin(), users.end(), std::back_inserter(earliestUsers),
        [&](auto* user) {
          return std::none_of(users.begin(), users.end(), [&](auto* other) {
            std::unordered_set<mlir::Block*> visited;
            return user != other && isAfterOrEqual(*user->getBlock(),
                                                   *other->getBlock(), visited);
          });
        });
    return earliestUsers;
  }

  /**
   * @brief Replaces all uses of a given operation with the results of a clone,
   *
   * @param rewriter The pattern rewriter to use.
   * @param original The original operation to replace.
   * @param clone The clone to replace with.
   * @param user The user operation to replace the inputs in.
   */
  static void replaceInputsWithClone(mlir::PatternRewriter& rewriter,
                                     mlir::Operation& original,
                                     mlir::Operation& clone,
                                     mlir::Operation& user) {
    for (size_t i = 0; i < user.getOperands().size(); i++) {
      const auto& operand = user.getOperand(i);
      const auto found = std::find(original.getResults().begin(),
                                   original.getResults().end(), operand);
      if (found == original.getResults().end()) {
        continue;
      }
      const auto idx = std::distance(original.getResults().begin(), found);
      rewriter.modifyOpInPlace(
          &user, [&] { user.setOperand(i, clone.getResults()[idx]); });
    }
  }

  mlir::LogicalResult
  matchAndRewrite(UnitaryInterface op,
                  mlir::PatternRewriter& rewriter) const override {
    // We only consider 1-qubit gates.
    if (op.getAllOutQubits().size() != 1) {
      return mlir::failure();
    }

    // Ensure that there is at least one user.
    const auto& users = op->getUsers();
    if (users.empty()) {
      return mlir::failure();
    }

    // There may be multiple users if canonicalization has deemed a branch
    // operand as pass-through and therefore removed it. If more than one user
    // is in the same block, then the `QuantumSinkShiftPattern` cannot be
    // applied.
    if (!std::all_of(users.begin(), users.end(), [&](auto* user) {
          return user->getBlock() != op->getBlock();
        })) {
      return mlir::failure();
    }

    // --- Begin Rewrite Phase ---
    auto earliestUsers = getEarliestUsers(users);

    for (auto* user : earliestUsers) {
      auto* block = user->getBlock();
      rewriter.setInsertionPoint(&block->front());
      auto* clone = rewriter.clone(*op);
      for (size_t i = 0; i < user->getOperands().size(); i++) {
        const auto& operand = user->getOperand(i);
        const auto found = std::find(op->getResults().begin(),
                                     op->getResults().end(), operand);
        if (found == op->getResults().end()) {
          continue;
        }
        const auto idx = std::distance(op->getResults().begin(), found);
        rewriter.modifyOpInPlace(
            user, [&] { user->setOperand(i, clone->getResult(idx)); });
      }
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

/**
 * @brief Populates the given pattern set with the
 * `QuantumSinkShiftPattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateQuantumSinkShiftPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add<QuantumSinkShiftPattern>(patterns.getContext());
}

} // namespace mqt::ir::opt
