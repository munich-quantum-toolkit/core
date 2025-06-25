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

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <map>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <string>

namespace mqt::ir::opt {

/**
 * @brief This pattern attempts to merge consecutive rotation gates.
 */
struct MergeRotationGatesPattern final
    : mlir::OpInterfaceRewritePattern<UnitaryInterface> {

  explicit MergeRotationGatesPattern(mlir::MLIRContext* context)
      : OpInterfaceRewritePattern(context) {}

  /**
   * @brief Checks if two gates can be merged.
   *
   * @param a The first gate.
   * @param b The second gate.
   * @return True if the gates can be merged, false otherwise.
   */
  [[nodiscard]] static bool areGatesMergeable(mlir::Operation& a,
                                              mlir::Operation& b) {
    static const std::unordered_set<std::string> MERGEABLE_GATES = {"rx"};

    const auto aName = a.getName().stripDialect().str();
    const auto bName = b.getName().stripDialect().str();

    return ((aName == bName) && (MERGEABLE_GATES.count(aName) > 0));
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
    if (!areGatesMergeable(*op, *user)) {
      return mlir::failure();
    }
    auto unitaryUser = mlir::dyn_cast<UnitaryInterface>(user);
    if (op.getAllOutQubits() != unitaryUser.getAllInQubits()) {
      return mlir::failure();
    }
    if (op.getPosCtrlInQubits().size() !=
            unitaryUser.getPosCtrlInQubits().size() ||
        op.getNegCtrlInQubits().size() !=
            unitaryUser.getNegCtrlInQubits().size()) {
      // We only need to check the sizes, because the order of the controls was
      // already checked by the previous condition.
      return mlir::failure();
    }
    return mlir::success();
  }

  void rewrite(UnitaryInterface op,
               mlir::PatternRewriter& rewriter) const override {
    auto user = mlir::dyn_cast<UnitaryInterface>(*op->getUsers().begin());

    // Compute newParam
    auto opParam = op.getParams()[0];
    auto userParam = user.getParams()[0];
    auto addParams =
        rewriter.create<mlir::arith::AddFOp>(user.getLoc(), opParam, userParam);
    auto newParam = addParams.getResult();

    // Create newUser
    auto userInQubit = user.getInQubits()[0];
    auto userControlQubitsPositive = user.getPosCtrlInQubits();
    auto userControlQubitsNegative = user.getNegCtrlInQubits();
    auto newUser = rewriter.create<RXOp>(
        user.getLoc(), userInQubit.getType(),
        userControlQubitsPositive.getType(),
        userControlQubitsNegative.getType(), mlir::DenseF64ArrayAttr{},
        mlir::DenseBoolArrayAttr{}, mlir::ValueRange{newParam},
        mlir::ValueRange{userInQubit}, userControlQubitsPositive,
        userControlQubitsNegative);

    // Prepare erasure of op
    const auto& opInQubits = op.getAllInQubits();
    const auto& newUserInQubits = user.getAllInQubits();
    for (size_t i = 0; i < newUser->getOperands().size(); i++) {
      const auto& operand = newUser->getOperand(i);
      const auto found =
          std::find(newUserInQubits.begin(), newUserInQubits.end(), operand);
      if (found == newUserInQubits.end()) {
        continue;
      }
      const auto idx = std::distance(newUserInQubits.begin(), found);
      rewriter.modifyOpInPlace(
          newUser, [&] { newUser->setOperand(i, opInQubits[idx]); });
    }

    // Eraise op
    rewriter.eraseOp(op);

    // Replace user with newUser
    rewriter.replaceOp(user, newUser);
  }
};

/**
 * @brief Populates the given pattern set with the `MergeRotationGatesPattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateMergeRotationGatesPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add<MergeRotationGatesPattern>(patterns.getContext());
}

} // namespace mqt::ir::opt
