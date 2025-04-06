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
#include <map>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <string>

namespace mqt::ir::opt {

/**
 * @brief This pattern attempts to cancel consecutive self-inverse operations.
 */
struct CancelConsecutiveInversesPattern final
    : mlir::OpInterfaceRewritePattern<UnitaryInterface> {

  explicit CancelConsecutiveInversesPattern(mlir::MLIRContext* context)
      : OpInterfaceRewritePattern(context) {}

  /**
   * @brief Checks if two gates are inverse to each other.
   *
   * @param a The first gate.
   * @param b The second gate.
   * @return True if the gates are inverse to each other, false otherwise.
   */
  [[nodiscard]] static bool areGatesInverse(mlir::Operation& a,
                                            mlir::Operation& b) {
    static const std::map<std::string, std::string> INVERSE_PAIRS = {
        {"x", "x"},           {"y", "y"},   {"z", "z"},
        {"h", "h"},           {"i", "i"},   {"swap", "swap"},
        {"ecr", "ecr"},       {"t", "tdg"}, {"s", "sdg"},
        {"sx", "sxdg"},       {"v", "vdg"}, {"iswap", "iswapdg"},
        {"peres", "peresdg"},
    };

    const auto aName = a.getName().stripDialect().str();
    const auto bName = b.getName().stripDialect().str();
    const auto foundA = INVERSE_PAIRS.find(aName);
    const auto foundB = INVERSE_PAIRS.find(bName);

    return (foundA != INVERSE_PAIRS.end() && foundA->second == bName) ||
           (foundB != INVERSE_PAIRS.end() && foundB->second == aName);
  }

  /**
   * @brief Checks if all users of an operation are the same.
   *
   * @param users The users to check.
   * @return True if all users are the same, false otherwise.
   */
  [[nodiscard]] static bool
  areUsersUnique(const mlir::ResultRange::user_range& users) {
    return std::none_of(users.begin(), users.end(),
                        [&](auto* user) { return user != *users.begin(); });
  }

  mlir::LogicalResult match(UnitaryInterface op) const override {
    const auto& users = op->getUsers();
    if (!areUsersUnique(users)) {
      return mlir::failure();
    }
    auto* user = *users.begin();
    if (!areGatesInverse(*op, *user)) {
      return mlir::failure();
    }
    auto unitaryUser = mlir::dyn_cast<UnitaryInterface>(user);
    if (op.getOutQubits() != unitaryUser.getAllInQubits()) {
      return mlir::failure();
    }
    if (op.getPosCtrlQubits().size() != unitaryUser.getPosCtrlQubits().size() ||
        op.getNegCtrlQubits().size() != unitaryUser.getNegCtrlQubits().size()) {
      // We only need to check the sizes, because the order of the controls was
      // already checked by the previous condition.
      return mlir::failure();
    }
    return mlir::success();
  }

  void rewrite(UnitaryInterface op,
               mlir::PatternRewriter& rewriter) const override {
    auto user = mlir::dyn_cast<UnitaryInterface>(
        *op->getUsers().begin()); // We always have exactly one user.
    const auto& childUsers = user->getUsers();

    for (const auto& childUser : childUsers) {
      for (size_t i = 0; i < childUser->getOperands().size(); i++) {
        const auto& operand = childUser->getOperand(i);
        const auto found = std::find(user.getOutQubits().begin(),
                                     user.getOutQubits().end(), operand);
        if (found == user.getOutQubits().end()) {
          continue;
        }
        const auto idx = std::distance(user.getOutQubits().begin(), found);
        rewriter.modifyOpInPlace(childUser, [&] {
          childUser->setOperand(i, op.getAllInQubits()[idx]);
        });
      }
    }

    rewriter.eraseOp(user);
    rewriter.eraseOp(op);
  }
};

/**
 * @brief Populates the given pattern set with the
 * `CancelConsecutiveInversePattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateCancelInversesPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add<CancelConsecutiveInversesPattern>(patterns.getContext());
}

} // namespace mqt::ir::opt
