/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <iostream>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <unordered_set>

namespace mlir::qco {

struct Quaternion {
  mlir::Value w;
  mlir::Value x;
  mlir::Value y;
  mlir::Value z;
};

static const std::unordered_set<std::string> MERGEABLE_GATES = {
    "gphase", "p", "rx", "ry", "rz", "rxx", "ryy", "rzz", "rzx"};

static const std::unordered_set<std::string> QUATERNION_GATES = {"u", "rx",
                                                                 "ry", "rz"};

/**
 * @brief This pattern attempts to merge consecutive rotation gates.
 */
struct MergeRotationGatesPattern final
    : mlir::OpInterfaceRewritePattern<UnitaryOpInterface> {
  explicit MergeRotationGatesPattern(mlir::MLIRContext* context,
                                     bool quaternionFolding)
      : OpInterfaceRewritePattern(context),
        quaternionFolding(quaternionFolding) {}

  const bool quaternionFolding = false;

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
   * @brief Checks if two gates can and should be merged with quaternions.
   *
   * Merging with quaternions should be used when merging gates of different
   * types, or when merging u-gates (which require quaternion multiplication).
   *
   * @param a The first gate.
   * @param b The second gate.
   * @return True if the gates should be merged using quaternions, false
   * otherwise.
   */
  [[nodiscard]] static bool areQuaternionMergeable(mlir::Operation& a,
                                                   mlir::Operation& b) {
    const auto aName = a.getName().stripDialect().str();
    const auto bName = b.getName().stripDialect().str();

    if (!(QUATERNION_GATES.contains(aName) &&
          QUATERNION_GATES.contains(bName))) {
      return false;
    }
    return (aName != bName) || (aName == "u" && bName == "u");
  }

  /**
   * @brief Checks if all users of an operation are the same.
   *
   * @param users The users to check.
   * @return True if all users are the same, false otherwise.
   */
  [[nodiscard]] static bool
  areUsersUnique(const mlir::ResultRange::user_range& users) {
    return llvm::none_of(users,
                         [&](auto* user) { return user != *users.begin(); });
  }

  mlir::LogicalResult
  matchAndRewrite(UnitaryOpInterface op,
                  mlir::PatternRewriter& rewriter) const override {
    const auto& users = op->getUsers();
    if (users.empty()) {
      return mlir::failure();
    }
    if (!areUsersUnique(users)) {
      return mlir::failure();
    }
    auto* user = *users.begin();

    if (!(areGatesMergeable(*op, *user) ||
          (quaternionFolding && areQuaternionMergeable(*op, *user)))) {
      return mlir::failure();
    }
    auto unitaryUser = mlir::dyn_cast<UnitaryOpInterface>(user);
    if (op.getAllOutQubits() != unitaryUser.getAllInQubits()) {
      return mlir::failure();
    }
    if (op.getPosCtrlInQubits().size() !=
            unitaryUser.getPosCtrlInQubits().size() ||
        op.getNegCtrlInQubits().size() !=
            unitaryUser.getNegCtrlInQubits().size()) {
      // We only need to check the sizes, because the order of the
      // controls was already checked by the previous condition.
      return mlir::failure();
    }
    rewriteAdditiveAngle(op, rewriter, quaternionFolding);
    return mlir::success();
  }

  /**
   * @brief Creates a new rotation gate.
   *
   * The new rotation gate is created by adding the angles of two compatible
   * rotation gates.
   *
   * @tparam OpType The type of the operation to create.
   * @param op The first instance of the rotation gate.
   * @param user The second instance of the rotation gate.
   * @param rewriter The pattern rewriter.
   * @return A new rotation gate.
   */
  template <typename OpType>
  static UnitaryOpInterface
  createOpAdditiveAngle(UnitaryOpInterface op, UnitaryOpInterface user,
                        mlir::PatternRewriter& rewriter) {
    auto loc = user->getLoc();

    auto userInQubits = user.getInQubits();
    auto userPosCtrlInQubits = user.getPosCtrlInQubits();
    auto userNegCtrlInQubits = user.getNegCtrlInQubits();

    auto opParam = op.getParams()[0];
    auto userParam = user.getParams()[0];
    auto add = rewriter.create<mlir::arith::AddFOp>(loc, opParam, userParam);
    const llvm::SmallVector<mlir::Value, 1> newParamsVec{add.getResult()};
    const mlir::ValueRange newParams(newParamsVec);

    return rewriter.create<OpType>(
        loc, userInQubits.getType(), userPosCtrlInQubits.getType(),
        userNegCtrlInQubits.getType(), mlir::DenseF64ArrayAttr{},
        mlir::DenseBoolArrayAttr{}, newParams, userInQubits,
        userPosCtrlInQubits, userNegCtrlInQubits);
  }

  static Quaternion createAxisQuaternion(mlir::Value angle, char axis,
                                         mlir::Location loc,
                                         mlir::PatternRewriter& rewriter) {
    auto floatType = angle.getType();

    // constant 0.0
    auto zeroAttr = rewriter.getFloatAttr(floatType, 0.0);
    auto zero = rewriter.create<mlir::arith::ConstantOp>(loc, zeroAttr);

    // constant 2.0
    auto twoAttr = rewriter.getFloatAttr(floatType, 2.0);
    auto two = rewriter.create<mlir::arith::ConstantOp>(loc, twoAttr);

    auto half = rewriter.create<mlir::arith::DivFOp>(loc, angle, two);
    // cos(angle/2)
    auto cos = rewriter.create<mlir::math::CosOp>(loc, floatType, half);
    // sin(angle/2)
    auto sin = rewriter.create<mlir::math::SinOp>(loc, floatType, half);

    switch (axis) {
    case 'x':
      return {.w = cos, .x = sin, .y = zero, .z = zero};
    case 'y':
      return {.w = cos, .x = zero, .y = sin, .z = zero};
    case 'z':
      return {.w = cos, .x = zero, .y = zero, .z = sin};
    default:
      throw std::runtime_error("Invalid rotation axis");
    }
  }

  static Quaternion quaternionFromRotation(UnitaryOpInterface op,
                                           mlir::PatternRewriter& rewriter) {
    auto const type = op->getName().stripDialect().str();

    if (type == "u") {
      return quaternionFromUGate(op, rewriter);
    }

    auto loc = op->getLoc();
    auto angle = op.getParams()[0];

    if (type == "rx") {
      return createAxisQuaternion(angle, 'x', loc, rewriter);
      // return {.w = cos, .x = sin, .y = zero, .z = zero};
    }
    if (type == "ry") {
      return createAxisQuaternion(angle, 'y', loc, rewriter);
      // return {.w = cos, .x = zero, .y = sin, .z = zero};
    }
    if (type == "rz") {
      return createAxisQuaternion(angle, 'z', loc, rewriter);
      // return {.w = cos, .x = zero, .y = zero, .z = sin};
    }
    throw std::runtime_error("Unsupported operation type: " + type);
  }

  static Quaternion hamiltonProduct(Quaternion q1, Quaternion q2,
                                    UnitaryOpInterface op,
                                    mlir::PatternRewriter& rewriter) {
    auto loc = op->getLoc();

    // wRes = w1w2 - x1x2 - y1y2 - z1z2
    auto w1w2 = rewriter.create<mlir::arith::MulFOp>(loc, q1.w, q2.w);
    auto x1x2 = rewriter.create<mlir::arith::MulFOp>(loc, q1.x, q2.x);
    auto y1y2 = rewriter.create<mlir::arith::MulFOp>(loc, q1.y, q2.y);
    auto z1z2 = rewriter.create<mlir::arith::MulFOp>(loc, q1.z, q2.z);
    auto wTemp1 = rewriter.create<mlir::arith::SubFOp>(loc, w1w2, x1x2);
    auto wTemp2 = rewriter.create<mlir::arith::SubFOp>(loc, wTemp1, y1y2);
    auto wRes = rewriter.create<mlir::arith::SubFOp>(loc, wTemp2, z1z2);

    // xRes = w1x2 + x1w2 + y1z2 - z1y2
    auto w1x2 = rewriter.create<mlir::arith::MulFOp>(loc, q1.w, q2.x);
    auto x1w2 = rewriter.create<mlir::arith::MulFOp>(loc, q1.x, q2.w);
    auto y1z2 = rewriter.create<mlir::arith::MulFOp>(loc, q1.y, q2.z);
    auto z1y2 = rewriter.create<mlir::arith::MulFOp>(loc, q1.z, q2.y);
    auto xTemp1 = rewriter.create<mlir::arith::AddFOp>(loc, w1x2, x1w2);
    auto xTemp2 = rewriter.create<mlir::arith::AddFOp>(loc, xTemp1, y1z2);
    auto xRes = rewriter.create<mlir::arith::SubFOp>(loc, xTemp2, z1y2);

    // yRes = w1y2 - x1z2 + y1w2 + z1x2
    auto w1y2 = rewriter.create<mlir::arith::MulFOp>(loc, q1.w, q2.y);
    auto x1z2 = rewriter.create<mlir::arith::MulFOp>(loc, q1.x, q2.z);
    auto y1w2 = rewriter.create<mlir::arith::MulFOp>(loc, q1.y, q2.w);
    auto z1x2 = rewriter.create<mlir::arith::MulFOp>(loc, q1.z, q2.x);
    auto yTemp1 = rewriter.create<mlir::arith::SubFOp>(loc, w1y2, x1z2);
    auto yTemp2 = rewriter.create<mlir::arith::AddFOp>(loc, yTemp1, y1w2);
    auto yRes = rewriter.create<mlir::arith::AddFOp>(loc, yTemp2, z1x2);

    // zRes = w1z2 + x1y2 - y1x2 + z1w2
    auto w1z2 = rewriter.create<mlir::arith::MulFOp>(loc, q1.w, q2.z);
    auto x1y2 = rewriter.create<mlir::arith::MulFOp>(loc, q1.x, q2.y);
    auto y1x2 = rewriter.create<mlir::arith::MulFOp>(loc, q1.y, q2.x);
    auto z1w2 = rewriter.create<mlir::arith::MulFOp>(loc, q1.z, q2.w);
    auto zTemp1 = rewriter.create<mlir::arith::AddFOp>(loc, w1z2, x1y2);
    auto zTemp2 = rewriter.create<mlir::arith::SubFOp>(loc, zTemp1, y1x2);
    auto zRes = rewriter.create<mlir::arith::AddFOp>(loc, zTemp2, z1w2);

    return {.w = wRes, .x = xRes, .y = yRes, .z = zRes};
  }

  static Quaternion quaternionFromUGate(UnitaryOpInterface op,
                                        mlir::PatternRewriter& rewriter) {
    auto loc = op->getLoc();
    auto params = op.getParams();

    // U gate uses ZYZ decomposition:
    // U(alpha, beta, gamma) = Rz(alpha) * Ry(beta) * Rz(gamma)
    auto qAlpha = createAxisQuaternion(params[0], 'z', loc, rewriter);
    auto qBeta = createAxisQuaternion(params[1], 'y', loc, rewriter);
    auto qGamma = createAxisQuaternion(params[2], 'z', loc, rewriter);

    auto temp = hamiltonProduct(qAlpha, qBeta, op, rewriter);
    return hamiltonProduct(temp, qGamma, op, rewriter);
  }

  static UnitaryOpInterface
  uGateFromQuaternion(Quaternion q, UnitaryOpInterface op,
                      mlir::PatternRewriter& rewriter) {
    auto loc = op->getLoc();
    auto user = mlir::dyn_cast<UnitaryOpInterface>(*op->getUsers().begin());

    auto userInQubits = user.getInQubits();
    auto userPosCtrlInQubits = user.getPosCtrlInQubits();
    auto userNegCtrlInQubits = user.getNegCtrlInQubits();

    // convert back to zyz euler angles
    auto floatType = op.getParams()[0].getType();
    // constant 1.0
    auto oneAttr = rewriter.getFloatAttr(floatType, 1.0);
    auto one = rewriter.create<mlir::arith::ConstantOp>(loc, oneAttr);
    // constant 2.0
    auto twoAttr = rewriter.getFloatAttr(floatType, 2.0);
    auto two = rewriter.create<mlir::arith::ConstantOp>(loc, twoAttr);

    // calculate angle beta (for y-rotation)
    // beta = acos(2 * (w**2 + z**2)-1)
    auto ww = rewriter.create<mlir::arith::MulFOp>(loc, q.w, q.w);
    auto zz = rewriter.create<mlir::arith::MulFOp>(loc, q.z, q.z);
    auto bTemp1 = rewriter.create<mlir::arith::AddFOp>(loc, ww, zz);
    auto bTemp2 = rewriter.create<mlir::arith::MulFOp>(loc, two, bTemp1);
    auto bTemp3 = rewriter.create<mlir::arith::SubFOp>(loc, bTemp2, one);
    auto beta = rewriter.create<mlir::math::AcosOp>(loc, bTemp3);

    // intermediate angles for z-rotations alpha and gamma
    // theta+ = atan2(z, w)
    // theta- = atan2(-x, y)
    auto xMinus = rewriter.create<mlir::arith::NegFOp>(loc, q.x);
    auto thetaPlus = rewriter.create<mlir::math::Atan2Op>(loc, q.z, q.w);
    auto thetaMinus = rewriter.create<mlir::math::Atan2Op>(loc, xMinus, q.y);

    // z-rotations alpha and gamma
    // alpha = theta+ - theta-
    // gamma = theta+ + theta-
    auto alpha =
        rewriter.create<mlir::arith::SubFOp>(loc, thetaPlus, thetaMinus);
    auto gamma =
        rewriter.create<mlir::arith::AddFOp>(loc, thetaPlus, thetaMinus);

    const llvm::SmallVector<mlir::Value, 3> newParamsVec{
        alpha.getResult(), beta.getResult(), gamma.getResult()};
    const mlir::ValueRange newParams(newParamsVec);

    return rewriter.create<UOp>(
        loc, userInQubits.getType(), userPosCtrlInQubits.getType(),
        userNegCtrlInQubits.getType(), mlir::DenseF64ArrayAttr{},
        mlir::DenseBoolArrayAttr{}, newParams, userInQubits,
        userPosCtrlInQubits, userNegCtrlInQubits);
  }

  /**
   * @brief Creates a u-gate by merging two rotation gates.
   *
   * The new  u-gate is created by converting the two
   * rotation gates into quaternions. These quaternions are then
   * multiplied/merged together by using the hamilton product. The order of
   * multiplication needs to be the reverse order in which the gates appear in
   * the circuit.
   *
   * @param op The first instance of the rotation gate.
   * @param user The second instance of the rotation gate.
   * @param rewriter The pattern rewriter.
   * @return A new rotation gate.
   */
  static UnitaryOpInterface
  createOpQuaternionMergedAngle(UnitaryOpInterface op, UnitaryOpInterface user,
                                mlir::PatternRewriter& rewriter) {
    auto q1 = quaternionFromRotation(op, rewriter);
    auto q2 = quaternionFromRotation(user, rewriter);
    auto qHam = hamiltonProduct(q2, q1, op, rewriter);
    auto newUser = uGateFromQuaternion(qHam, op, rewriter);

    return newUser;
  }

  /**
   * @brief Merges two consecutive rotation gates into a single gate.
   *
   * The function supports gphase, p, rx, ry, rz, rxx, ryy, rzz, and rzx.
   * The gates are merged by adding their angles.
   * The merged gate is not removed if the angles add up to zero.
   *
   * @param op The first instance of the rotation gate.
   * @param rewriter The pattern rewriter.
   */
  void static rewriteAdditiveAngle(UnitaryOpInterface op,
                                   mlir::PatternRewriter& rewriter,
                                   bool quaternionFolding) {
    auto const type = op->getName().stripDialect().str();

    auto user = mlir::dyn_cast<UnitaryOpInterface>(*op->getUsers().begin());

    UnitaryOpInterface newUser;

    if (quaternionFolding && areQuaternionMergeable(*op, *user)) {
      newUser = createOpQuaternionMergedAngle(op, user, rewriter);
    } else if (type == "gphase") {
      newUser = createOpAdditiveAngle<GPhaseOp>(op, user, rewriter);
    } else if (type == "p") {
      newUser = createOpAdditiveAngle<POp>(op, user, rewriter);
    } else if (type == "rx") {
      newUser = createOpAdditiveAngle<RXOp>(op, user, rewriter);
    } else if (type == "ry") {
      newUser = createOpAdditiveAngle<RYOp>(op, user, rewriter);
    } else if (type == "rz") {
      newUser = createOpAdditiveAngle<RZOp>(op, user, rewriter);
    } else if (type == "rxx") {
      newUser = createOpAdditiveAngle<RXXOp>(op, user, rewriter);
    } else if (type == "ryy") {
      newUser = createOpAdditiveAngle<RYYOp>(op, user, rewriter);
    } else if (type == "rzz") {
      newUser = createOpAdditiveAngle<RZZOp>(op, user, rewriter);
    } else if (type == "rzx") {
      newUser = createOpAdditiveAngle<RZXOp>(op, user, rewriter);
    } else {
      throw std::runtime_error("Unsupported operation type: " + type);
    }

    // Prepare erasure of op
    const auto& opAllInQubits = op.getAllInQubits();
    const auto& newUserAllInQubits = newUser.getAllInQubits();
    for (size_t i = 0; i < newUser->getOperands().size(); i++) {
      const auto& operand = newUser->getOperand(i);
      const auto found = llvm::find(newUserAllInQubits, operand);
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
};

/**
 * @brief Populates the given pattern set with the `MergeRotationGatesPattern`.
 *
 * @param patterns The pattern set to populate.
 */
void populateMergeRotationGatesPatterns(mlir::RewritePatternSet& patterns,
                                        bool quaternionFolding) {
  patterns.add<MergeRotationGatesPattern>(patterns.getContext(),
                                          quaternionFolding);
}

#define GEN_PASS_DEF_MERGEROTATIONGATES
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

/**
 * @brief This pattern attempts to merge consecutive rotation gates by using
 * quaternions
 */
struct MergeRotationGates final
    : impl::MergeRotationGatesBase<MergeRotationGates> {
  using impl::MergeRotationGatesBase<
      MergeRotationGates>::MergeRotationGatesBase;

  // TODO add flag for pass
  void runOnOperation() override {
    std::cout << "Hello from MergeRotationGates" << std::endl;
    // TODO implement pass here
  }
};

} // namespace mlir::qco
