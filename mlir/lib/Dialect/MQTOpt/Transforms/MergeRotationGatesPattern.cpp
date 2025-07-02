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
#include "mlir/IR/BuiltinAttributes.h"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace mqt::ir::opt {

static const std::unordered_set<std::string> MERGEABLE_GATES = {"gphase", "rx", "ry", "rz", "rxx", "ryy", "rzz", "rzx"};

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
    const auto aName = a.getName().stripDialect().str();
    const auto bName = b.getName().stripDialect().str();

    return ((aName == bName) && (MERGEABLE_GATES.count(aName) == 1));
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

  /**
   * @brief Merges two consecutive rotation gates into a single gate.
   *
   * The function supports gphase, rx, ry, rz, rxx, ryy, rzz, and rzx.
   * The gates are merged by adding their angles.
   * The merged gate is not removed if the angles add up to zero.
   *
   * @param op The first instance of the rotation gate.
   * @param rewriter The pattern rewriter.
   */
  void static rewriteAdditiveAngle(UnitaryInterface op,
                                   mlir::PatternRewriter& rewriter) {
    auto const type = op->getName().stripDialect().str();

    auto user = mlir::dyn_cast<UnitaryInterface>(*op->getUsers().begin());
    auto loc = user->getLoc();

    auto opParam = op.getParams()[0];
    auto userParam = user.getParams()[0];

    auto add = rewriter.create<mlir::arith::AddFOp>(loc, opParam, userParam);
    const llvm::SmallVector<mlir::Value, 1> newParamsVec{add.getResult()};
    const mlir::ValueRange newParams(newParamsVec);

    auto userInQubits = user.getInQubits();
    auto userPosCtrlInQubits = user.getPosCtrlInQubits();
    auto userNegCtrlInQubits = user.getNegCtrlInQubits();

    UnitaryInterface newUser;
    if (type == "gphase") {
      newUser = rewriter.create<GPhaseOp>(
          loc, userInQubits.getType(), userPosCtrlInQubits.getType(),
          userNegCtrlInQubits.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, newParams, userInQubits,
          userPosCtrlInQubits, userNegCtrlInQubits);
    } else if (type == "rx") {
      newUser = rewriter.create<RXOp>(
          loc, userInQubits.getType(), userPosCtrlInQubits.getType(),
          userNegCtrlInQubits.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, newParams, userInQubits,
          userPosCtrlInQubits, userNegCtrlInQubits);
    } else if (type == "ry") {
      newUser = rewriter.create<RYOp>(
          loc, userInQubits.getType(), userPosCtrlInQubits.getType(),
          userNegCtrlInQubits.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, newParams, userInQubits,
          userPosCtrlInQubits, userNegCtrlInQubits);
    } else if (type == "rz") {
      newUser = rewriter.create<RZOp>(
          loc, userInQubits.getType(), userPosCtrlInQubits.getType(),
          userNegCtrlInQubits.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, newParams, userInQubits,
          userPosCtrlInQubits, userNegCtrlInQubits);
    } else if (type == "rxx") {
      newUser = rewriter.create<RXXOp>(
          loc, userInQubits.getType(), userPosCtrlInQubits.getType(),
          userNegCtrlInQubits.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, newParams, userInQubits,
          userPosCtrlInQubits, userNegCtrlInQubits);
    } else if (type == "ryy") {
      newUser = rewriter.create<RYYOp>(
          loc, userInQubits.getType(), userPosCtrlInQubits.getType(),
          userNegCtrlInQubits.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, newParams, userInQubits,
          userPosCtrlInQubits, userNegCtrlInQubits);
    } else if (type == "rzz") {
      newUser = rewriter.create<RZZOp>(
          loc, userInQubits.getType(), userPosCtrlInQubits.getType(),
          userNegCtrlInQubits.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, newParams, userInQubits,
          userPosCtrlInQubits, userNegCtrlInQubits);
    } else if (type == "rzx") {
      newUser = rewriter.create<RZXOp>(
          loc, userInQubits.getType(), userPosCtrlInQubits.getType(),
          userNegCtrlInQubits.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, newParams, userInQubits,
          userPosCtrlInQubits, userNegCtrlInQubits);
    } else {
      throw std::runtime_error("Unsupported operation type: " + type);
    }

    // Prepare erasure of op
    const auto& opAllInQubits = op.getAllInQubits();
    const auto& newUserAllInQubits = newUser.getAllInQubits();
    for (size_t i = 0; i < newUser->getOperands().size(); i++) {
      const auto& operand = newUser->getOperand(i);
      const auto found = std::find(newUserAllInQubits.begin(),
                                   newUserAllInQubits.end(), operand);
      if (found == newUserAllInQubits.end()) {
        continue;
      }
      const auto idx = std::distance(newUserAllInQubits.begin(), found);
      rewriter.modifyOpInPlace(
          newUser, [&] { newUser->setOperand(i, opAllInQubits[idx]); });
    }

    // Replace user with newUser
    rewriter.replaceOp(user, newUser);

    // Erase op
    rewriter.eraseOp(op);
  }

  void rewrite(UnitaryInterface op,
               mlir::PatternRewriter& rewriter) const override {
    auto const type = op->getName().stripDialect().str();

    if (MERGEABLE_GATES.count(type) == 1) {
      rewriteAdditiveAngle(op, rewriter);
    } else {
      throw std::runtime_error("Unsupported operation type: " + type);
    }
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
