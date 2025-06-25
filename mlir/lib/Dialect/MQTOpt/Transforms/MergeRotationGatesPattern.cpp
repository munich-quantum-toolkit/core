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
    static const std::unordered_set<std::string> MERGEABLE_GATES = {
        "rx",  "ry",        "rz",       "gphase", "rxx", "ryy",
        "rzz", "xxminusyy", "xxplusyy", "u",      "u2"};

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

  static UnitaryInterface createNewUser(
      const mlir::Location loc, const std::string& type,
      const mlir::ValueRange inQubits, mlir::ValueRange controlQubitsPositive,
      mlir::ValueRange controlQubitsNegative, mlir::ValueRange opParams,
      mlir::ValueRange userParams, mlir::PatternRewriter& rewriter) {
    if (type == "gphase") {
      auto add =
          rewriter.create<mlir::arith::AddFOp>(loc, opParams[0], userParams[0]);
      const llvm::SmallVector<mlir::Value, 1> newParamsVec{add.getResult()};
      const mlir::ValueRange newParams(newParamsVec);
      return rewriter.create<GPhaseOp>(
          loc, inQubits.getType(), controlQubitsPositive.getType(),
          controlQubitsNegative.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, newParams, inQubits,
          controlQubitsPositive, controlQubitsNegative);
    }
    if (type == "rx") {
      auto add =
          rewriter.create<mlir::arith::AddFOp>(loc, opParams[0], userParams[0]);
      const llvm::SmallVector<mlir::Value, 1> newParamsVec{add.getResult()};
      const mlir::ValueRange newParams(newParamsVec);
      return rewriter.create<RXOp>(
          loc, inQubits.getType(), controlQubitsPositive.getType(),
          controlQubitsNegative.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, newParams, inQubits,
          controlQubitsPositive, controlQubitsNegative);
    }
    if (type == "ry") {
      auto add =
          rewriter.create<mlir::arith::AddFOp>(loc, opParams[0], userParams[0]);
      const llvm::SmallVector<mlir::Value, 1> newParamsVec{add.getResult()};
      const mlir::ValueRange newParams(newParamsVec);
      return rewriter.create<RYOp>(
          loc, inQubits.getType(), controlQubitsPositive.getType(),
          controlQubitsNegative.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, newParams, inQubits,
          controlQubitsPositive, controlQubitsNegative);
    }
    if (type == "rz") {
      auto add =
          rewriter.create<mlir::arith::AddFOp>(loc, opParams[0], userParams[0]);
      const llvm::SmallVector<mlir::Value, 1> newParamsVec{add.getResult()};
      const mlir::ValueRange newParams(newParamsVec);
      return rewriter.create<RZOp>(
          loc, inQubits.getType(), controlQubitsPositive.getType(),
          controlQubitsNegative.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, newParams, inQubits,
          controlQubitsPositive, controlQubitsNegative);
    }
    if (type == "rxx") {
      auto add =
          rewriter.create<mlir::arith::AddFOp>(loc, opParams[0], userParams[0]);
      const llvm::SmallVector<mlir::Value, 1> newParamsVec{add.getResult()};
      const mlir::ValueRange newParams(newParamsVec);
      return rewriter.create<RXXOp>(
          loc, inQubits.getType(), controlQubitsPositive.getType(),
          controlQubitsNegative.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, newParams, inQubits,
          controlQubitsPositive, controlQubitsNegative);
    }
    if (type == "ryy") {
      auto add =
          rewriter.create<mlir::arith::AddFOp>(loc, opParams[0], userParams[0]);
      const llvm::SmallVector<mlir::Value, 1> newParamsVec{add.getResult()};
      const mlir::ValueRange newParams(newParamsVec);
      return rewriter.create<RYYOp>(
          loc, inQubits.getType(), controlQubitsPositive.getType(),
          controlQubitsNegative.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, newParams, inQubits,
          controlQubitsPositive, controlQubitsNegative);
    }
    if (type == "rzz") {
      auto add =
          rewriter.create<mlir::arith::AddFOp>(loc, opParams[0], userParams[0]);
      const llvm::SmallVector<mlir::Value, 1> newParamsVec{add.getResult()};
      const mlir::ValueRange newParams(newParamsVec);
      return rewriter.create<RZZOp>(
          loc, inQubits.getType(), controlQubitsPositive.getType(),
          controlQubitsNegative.getType(), mlir::DenseF64ArrayAttr{},
          mlir::DenseBoolArrayAttr{}, newParams, inQubits,
          controlQubitsPositive, controlQubitsNegative);
    }
    if (type == "xxminusyy" || type == "xxplusyy" || type == "u" ||
        type == "u2") {
      throw std::runtime_error("Not implemented yet");
    }
    throw std::runtime_error("Unsupported operation type");
  }

  void rewrite(UnitaryInterface op,
               mlir::PatternRewriter& rewriter) const override {
    auto user = mlir::dyn_cast<UnitaryInterface>(*op->getUsers().begin());

    // Create newUser
    auto newUser = createNewUser(
        user.getLoc(), user->getName().stripDialect().str(), user.getInQubits(),
        user.getPosCtrlInQubits(), user.getNegCtrlInQubits(), op.getParams(),
        user.getParams(), rewriter);

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
