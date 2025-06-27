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
        "gphase", "rx",  "ry",        "rz",       "rxx", "ryy",
        "rzz",    "rzx", "xxminusyy", "xxplusyy", "u",   "u2"};

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

  static double getDoubleFromValue(mlir::Value value) {
    auto constOp = value.getDefiningOp<mlir::arith::ConstantOp>();
    auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(constOp.getValue());
    return floatAttr.getValueAsDouble();
  }

  static mlir::Value getValueFromDouble(double value,
                                        mlir::PatternRewriter& rewriter,
                                        mlir::Location loc) {
    auto floatAttr = rewriter.getFloatAttr(rewriter.getF64Type(), value);
    return rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getF64Type(),
                                                    floatAttr);
  }

  void rewriteSingleAdditiveParam(UnitaryInterface op, const std::string& type,
                                  mlir::PatternRewriter& rewriter) const {
    auto user = mlir::dyn_cast<UnitaryInterface>(*op->getUsers().begin());

    auto loc = user->getLoc();

    auto opParamDouble = getDoubleFromValue(op.getParams()[0]);
    auto userParamDouble = getDoubleFromValue(user.getParams()[0]);

    double newParamValue = opParamDouble + userParamDouble;

    auto newParam = getValueFromDouble(newParamValue, rewriter, loc);
    const llvm::SmallVector<mlir::Value, 1> newParamsVec{newParam};
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

    // Eraise op
    rewriter.eraseOp(op);

    // Replace user with newUser
    rewriter.replaceOp(user, newUser);
  }

  void rewrite(UnitaryInterface op,
               mlir::PatternRewriter& rewriter) const override {
    auto const type = op->getName().stripDialect().str();

    if (type == "gphase" || type == "rx" || type == "ry" || type == "rz" ||
        type == "rxx" || type == "ryy" || type == "rzz" || type == "rzx") {
      rewriteSingleAdditiveParam(op, type, rewriter);
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
