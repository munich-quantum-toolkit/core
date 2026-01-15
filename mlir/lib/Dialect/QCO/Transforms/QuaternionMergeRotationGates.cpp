/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <algorithm>
#include <array>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_MERGEROTATIONGATES
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

/**
 * @brief This pattern attempts to merge consecutive rotation gates.
 */
struct MergeRotationGatesPattern final
    : mlir::OpInterfaceRewritePattern<UnitaryOpInterface> {
  explicit MergeRotationGatesPattern(mlir::MLIRContext* context)
      : OpInterfaceRewritePattern(context) {}

  struct Quaternion {
    mlir::Value w;
    mlir::Value x;
    mlir::Value y;
    mlir::Value z;
  };

  static constexpr std::array<std::string_view, 4> MERGEABLE_GATES = {
      "u", "rx", "ry", "rz"};

  static bool isMergeable(std::string_view name) {
    return std::ranges::find(MERGEABLE_GATES, name) != MERGEABLE_GATES.end();
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

    if (!(isMergeable(aName) && isMergeable(bName))) {
      return false;
    }
    return (aName != bName) || (aName == "u" && bName == "u");
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
    auto angle = op.getParameter(0);

    if (type == "rx") {
      return createAxisQuaternion(angle, 'x', loc, rewriter);
    }
    if (type == "ry") {
      return createAxisQuaternion(angle, 'y', loc, rewriter);
    }
    if (type == "rz") {
      return createAxisQuaternion(angle, 'z', loc, rewriter);
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

    // U gate uses ZYZ decomposition:
    // U(alpha, beta, gamma) = Rz(alpha) * Ry(beta) * Rz(gamma)
    auto qAlpha = createAxisQuaternion(op.getParameter(0), 'z', loc, rewriter);
    auto qBeta = createAxisQuaternion(op.getParameter(1), 'y', loc, rewriter);
    auto qGamma = createAxisQuaternion(op.getParameter(2), 'z', loc, rewriter);

    auto temp = hamiltonProduct(qAlpha, qBeta, op, rewriter);
    return hamiltonProduct(temp, qGamma, op, rewriter);
  }

  static UnitaryOpInterface
  uGateFromQuaternion(Quaternion q, UnitaryOpInterface op,
                      mlir::PatternRewriter& rewriter) {
    auto loc = op->getLoc();

    // convert back to zyz euler angles
    auto floatType = op.getParameter(0).getType();
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

    return rewriter.create<UOp>(loc, op.getInputQubit(0), alpha.getResult(),
                                beta.getResult(), gamma.getResult());
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

  mlir::LogicalResult
  matchAndRewrite(UnitaryOpInterface op,
                  mlir::PatternRewriter& rewriter) const override {
    // QCO operations cannot contain control qubits so we dont need to check for
    // these
    if (!op->hasOneUse()) {
      return mlir::failure();
    }

    const auto& users = op->getUsers();
    auto* userOP = *users.begin();

    if (!areQuaternionMergeable(*op, *userOP)) {
      return mlir::failure();
    }
    auto user = mlir::dyn_cast<UnitaryOpInterface>(userOP);

    // TODO: merge createOpQuaternionMergedAngle into here?
    UnitaryOpInterface newUser =
        createOpQuaternionMergedAngle(op, user, rewriter);

    // Replace user with newUser
    rewriter.replaceOp(user, newUser->getResults());

    // Erase op
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

/**
 * @brief Populates the given pattern set with the `MergeRotationGatesPattern`.
 *
 * @param patterns The pattern set to populate.
 */
static void
populateMergeRotationGatesPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add<MergeRotationGatesPattern>(patterns.getContext());
}

/**
 * @brief This pattern attempts to merge consecutive rotation gates by using
 * quaternions
 */
struct MergeRotationGates final
    : impl::MergeRotationGatesBase<MergeRotationGates> {
  using impl::MergeRotationGatesBase<
      MergeRotationGates>::MergeRotationGatesBase;

  void runOnOperation() override {
    // Get the current operation being operated on.
    auto op = getOperation();
    auto* ctx = &getContext();

    // Define the set of patterns to use.
    mlir::RewritePatternSet patterns(ctx);
    populateMergeRotationGatesPatterns(patterns);

    // Apply patterns in an iterative and greedy manner.
    if (mlir::failed(mlir::applyPatternsGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::qco
