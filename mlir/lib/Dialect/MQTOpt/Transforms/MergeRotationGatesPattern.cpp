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

static constexpr auto PI = static_cast<double>(
    3.141592653589793238462643383279502884197169399375105820974L);

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

    if (aName != bName) {
      return false;
    }

    static const std::unordered_set<std::string> MERGEABLE_GATES = {
        "gphase", "rx", "ry", "rz", "rxx", "ryy", "rzz", "rzx"};
    if (MERGEABLE_GATES.count(aName) == 1) {
      return true;
    }

    if (aName == "xxminusyy" || aName == "xxplusyy") {
      return areGatesMergeableXxMinusPlusYy(a, b);
    }

    if (aName == "u") {
      return areGatesMergeableU(a, b);
    }

    if (aName == "u2") {
      return areGatesMergeableU2(a, b);
    }

    return false;
  }

  [[nodiscard]] static bool areGatesMergeableXxMinusPlusYy(mlir::Operation& a,
                                                           mlir::Operation& b) {
    auto unitaryA = mlir::dyn_cast<UnitaryInterface>(a);
    auto unitaryB = mlir::dyn_cast<UnitaryInterface>(b);

    auto aThetaDouble = getDoubleFromValue(unitaryA.getParams()[0]);
    auto bThetaDouble = getDoubleFromValue(unitaryB.getParams()[0]);

    return (aThetaDouble == -bThetaDouble);
  }

  [[nodiscard]] static bool areGatesMergeableU(mlir::Operation& a,
                                               mlir::Operation& b) {
    auto unitaryA = mlir::dyn_cast<UnitaryInterface>(a);
    auto unitaryB = mlir::dyn_cast<UnitaryInterface>(b);

    auto aThetaDouble = getDoubleFromValue(unitaryA.getParams()[0]);
    auto aPhiDouble = getDoubleFromValue(unitaryA.getParams()[1]);
    auto aLambdaDouble = getDoubleFromValue(unitaryA.getParams()[2]);

    auto bThetaDouble = getDoubleFromValue(unitaryB.getParams()[0]);
    auto bPhiDouble = getDoubleFromValue(unitaryB.getParams()[1]);
    auto bLambdaDouble = getDoubleFromValue(unitaryB.getParams()[2]);

    return ((aThetaDouble == -bThetaDouble) && (aPhiDouble == -bLambdaDouble) &&
            (aLambdaDouble == -bPhiDouble));
  }

  [[nodiscard]] static bool areGatesMergeableU2(mlir::Operation& a,
                                                mlir::Operation& b) {
    auto unitaryA = mlir::dyn_cast<UnitaryInterface>(a);
    auto unitaryB = mlir::dyn_cast<UnitaryInterface>(b);

    auto aPhiDouble = getDoubleFromValue(unitaryA.getParams()[0]);
    auto aLambdaDouble = getDoubleFromValue(unitaryA.getParams()[1]);

    auto bPhiDouble = getDoubleFromValue(unitaryB.getParams()[0]);
    auto bLambdaDouble = getDoubleFromValue(unitaryB.getParams()[1]);

    return ((aPhiDouble == -bLambdaDouble - PI) &&
            (aLambdaDouble == -bPhiDouble + PI));
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
    const auto f64Type = rewriter.getF64Type();
    auto floatAttr = rewriter.getF64FloatAttr(value);
    return rewriter.create<mlir::arith::ConstantOp>(loc, f64Type, floatAttr);
  }

  void static cancelGates(UnitaryInterface op, UnitaryInterface user,
                          mlir::PatternRewriter& rewriter) {
    // Prepare erasures of op and user
    const auto& userUsers = user->getUsers();
    const auto& opAllInQubits = op.getAllInQubits();
    const auto& userAllOutQubits = user.getAllOutQubits();
    for (const auto& userUser : userUsers) {
      for (size_t i = 0; i < userUser->getOperands().size(); i++) {
        const auto& operand = userUser->getOperand(i);
        const auto found = std::find(userAllOutQubits.begin(),
                                     userAllOutQubits.end(), operand);
        if (found == userAllOutQubits.end()) {
          continue;
        }
        const auto idx = std::distance(userAllOutQubits.begin(), found);
        rewriter.modifyOpInPlace(
            userUser, [&] { userUser->setOperand(i, opAllInQubits[idx]); });
      }
    }

    // Erase op
    rewriter.eraseOp(op);

    // Erase user
    rewriter.eraseOp(user);
  }

  void static rewriteSingleAdditiveParam(UnitaryInterface op,
                                         mlir::PatternRewriter& rewriter) {
    auto const type = op->getName().stripDialect().str();

    auto user = mlir::dyn_cast<UnitaryInterface>(*op->getUsers().begin());

    auto opParamDouble = getDoubleFromValue(op.getParams()[0]);
    auto userParamDouble = getDoubleFromValue(user.getParams()[0]);

    double const newParamValue = opParamDouble + userParamDouble;

    if (newParamValue == 0.0) {
      cancelGates(op, user, rewriter);
      return;
    }

    auto loc = user->getLoc();
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

    // Erase op
    rewriter.eraseOp(op);

    // Replace user with newUser
    rewriter.replaceOp(user, newUser);
  }

  void static rewriteXxMinusPlusYy(UnitaryInterface op,
                                   mlir::PatternRewriter& rewriter) {
    auto user = mlir::dyn_cast<UnitaryInterface>(*op->getUsers().begin());
    cancelGates(op, user, rewriter);
  }

  void static rewriteU(UnitaryInterface op, mlir::PatternRewriter& rewriter) {
    auto user = mlir::dyn_cast<UnitaryInterface>(*op->getUsers().begin());
    cancelGates(op, user, rewriter);
  }

  void static rewriteU2(UnitaryInterface op, mlir::PatternRewriter& rewriter) {
    auto user = mlir::dyn_cast<UnitaryInterface>(*op->getUsers().begin());
    cancelGates(op, user, rewriter);
  }

  void rewrite(UnitaryInterface op,
               mlir::PatternRewriter& rewriter) const override {
    auto const type = op->getName().stripDialect().str();

    if (type == "gphase" || type == "rx" || type == "ry" || type == "rz" ||
        type == "rxx" || type == "ryy" || type == "rzz" || type == "rzx") {
      rewriteSingleAdditiveParam(op, rewriter);
    } else if (type == "xxminusyy" || type == "xxplusyy") {
      rewriteXxMinusPlusYy(op, rewriter);
    } else if (type == "u") {
      rewriteU(op, rewriter);
    } else if (type == "u2") {
      rewriteU2(op, rewriter);
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
