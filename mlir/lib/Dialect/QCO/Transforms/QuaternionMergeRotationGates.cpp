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
#include <string_view>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_MERGEROTATIONGATES
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

/**
 * @brief Pattern that merges consecutive rotation gates using quaternion
 * multiplication.
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

  enum class RotationAxis { X, Y, Z };

  static constexpr std::array<std::string_view, 4> MERGEABLE_GATES = {
      "u", "rx", "ry", "rz"};

  /**
   * @brief Checks if an operation is a mergeable rotation gate (rx, ry, rz, u).
   *
   * @param name Name of the operation to check
   * @return True if mergeable, false otherwise
   */
  static bool isMergeable(std::string_view name) {
    return std::ranges::find(MERGEABLE_GATES, name) != MERGEABLE_GATES.end();
  }

  /**
   * @brief Checks if two gates require quaternion-based merging.
   *
   * Returns true for different gate types (e.g., RX+RY) or two U-gates.
   * Same-axis rotations (e.g., RX+RX) use angle addition and aren't handled
   * here.
   *
   * @param a The first gate
   * @param b The second gate
   * @return True if quaternion-based merging should be used, false otherwise
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

  /**
   * @brief Converts a single-axis rotation to quaternion representation.
   *
   * Uses half-angle formulas:
   *   RX(a) = Q(cos(a/2), sin(a/2), 0, 0)
   *   RY(a) = Q(cos(a/2), 0, sin(a/2), 0)
   *   RZ(a) = Q(cos(a/2), 0, 0, sin(a/2))
   *
   * @see
   * https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
   * @param angle The rotation angle
   * @param axis The rotation axis (X, Y, or Z)
   * @param loc Location in the IR
   * @param rewriter Pattern rewriter for creating new operations
   * @return Quaternion representing the rotation
   */
  static Quaternion createAxisQuaternion(mlir::Value angle, RotationAxis axis,
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
    case RotationAxis::X:
      return {.w = cos, .x = sin, .y = zero, .z = zero};
    case RotationAxis::Y:
      return {.w = cos, .x = zero, .y = sin, .z = zero};
    case RotationAxis::Z:
      return {.w = cos, .x = zero, .y = zero, .z = sin};
    }
  }

  /**
   * @brief Converts a rotation gate (RX, RY, RZ, or U) to quaternion
   * representation.
   *
   * @param op The rotation gate to convert
   * @param rewriter Pattern rewriter for creating new operations
   * @return Quaternion representing the rotation gate
   */
  static Quaternion quaternionFromRotation(UnitaryOpInterface op,
                                           mlir::PatternRewriter& rewriter) {
    auto const type = op->getName().stripDialect().str();

    if (type == "u") {
      return quaternionFromUGate(op, rewriter);
    }

    auto loc = op->getLoc();
    auto angle = op.getParameter(0);

    if (type == "rx") {
      return createAxisQuaternion(angle, RotationAxis::X, loc, rewriter);
    }
    if (type == "ry") {
      return createAxisQuaternion(angle, RotationAxis::Y, loc, rewriter);
    }
    if (type == "rz") {
      return createAxisQuaternion(angle, RotationAxis::Z, loc, rewriter);
    }
    llvm_unreachable("Unsupported operation type");
  }

  /**
   * @brief Computes the Hamilton product of two quaternions (q1 * q2).
   *
   * For q1 = w1 + x1*i + y1*j + z1*k and q2 = w2 + x2*i + y2*j + z2*k:
   *
   * q1 * q2 = (w1w2 - x1x2 - y1y2 - z1z2)
   *         + (w1x2 + x1w2 + y1z2 - z1y2) * i
   *         + (w1y2 - x1z2 + y1w2 + z1x2) * j
   *         + (w1z2 + x1y2 - y1x2 + z1w2) * k
   *
   * @see https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
   * @param q1 First quaternion
   * @param q2 Second quaternion
   * @param op Current operation (used for location)
   * @param rewriter Pattern rewriter for creating arithmetic operations
   * @return The product quaternion
   */
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

  /**
   * @brief Converts a u-gate to quaternion representation.
   *
   * U(alpha, beta, gamma) uses ZYZ decomposition: RZ(alpha) -> RY(beta) ->
   * RZ(gamma).
   *
   * When composing rotations, quaternion multiplication follows matrix
   * multiplication order (right-to-left), which is the reverse of the
   * application sequence:
   *   Sequential application: RZ(alpha), then RY(beta), then RZ(gamma)
   *   Quaternion product:     Qgamma * Qbeta * Qalpha
   *
   * @param op The u-gate operation to convert
   * @param rewriter Pattern rewriter for creating new operations
   * @return Quaternion representing the u-gate
   */
  static Quaternion quaternionFromUGate(UnitaryOpInterface op,
                                        mlir::PatternRewriter& rewriter) {
    auto loc = op->getLoc();

    // U gate uses ZYZ decomposition:
    // U(alpha, beta, gamma) = Rz(alpha) -> Ry(beta) -> Rz(gamma)
    auto qAlpha = createAxisQuaternion(op.getParameter(0), RotationAxis::Z, loc,
                                       rewriter);
    auto qBeta = createAxisQuaternion(op.getParameter(1), RotationAxis::Y, loc,
                                      rewriter);
    auto qGamma = createAxisQuaternion(op.getParameter(2), RotationAxis::Z, loc,
                                       rewriter);

    // qGamma * qBeta * qAlpha (multiplication in reverse order!)
    auto temp = hamiltonProduct(qGamma, qBeta, op, rewriter);
    return hamiltonProduct(temp, qAlpha, op, rewriter);
  }

  /**
   * @brief Converts a quaternion to a u-gate using ZYZ Euler angle extraction.
   *
   * For unit quaternion q = w + x*i + y*j + z*k, extracts u-gate parameters:
   *   alpha = atan2(z, w) - atan2(-x, y)
   *   beta  = acos(2 * (w^2 + z^2) - 1)
   *   gamma = atan2(z, w) + atan2(-x, y)
   *
   * Based on Bernardes & Viollet (2022), simplified for unit quaternions and
   * proper ZYZ Euler angles (Chapter 3.3):
   * https://doi.org/10.1371/journal.pone.0276302
   *
   * Reference implementation:
   * https://github.com/evbernardes/quaternion_to_euler
   * SymPy also implements this paper:
   * https://docs.sympy.org/latest/modules/algebras.html#sympy.algebras.Quaternion.to_euler
   *
   * @note Floating-point errors may accumulate when merging many gates.
   * @param q The quaternion to convert
   * @param op The current operation (used for location and type information)
   * @param rewriter Pattern rewriter for creating new operations
   * @return U-gate equivalent to the quaternion rotation
   */
  static UnitaryOpInterface
  uGateFromQuaternion(Quaternion q, UnitaryOpInterface op,
                      mlir::PatternRewriter& rewriter) {
    auto loc = op->getLoc();

    auto floatType = op.getParameter(0).getType();
    // constant 1.0
    auto oneAttr = rewriter.getFloatAttr(floatType, 1.0);
    auto one = rewriter.create<mlir::arith::ConstantOp>(loc, oneAttr);
    // constant 2.0
    auto twoAttr = rewriter.getFloatAttr(floatType, 2.0);
    auto two = rewriter.create<mlir::arith::ConstantOp>(loc, twoAttr);

    // calculate angle beta (for y-rotation)
    // beta = acos(2 * (w^2 + z^2) - 1)
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
   * Converts both gates to quaternions, multiplies them using the Hamilton
   * product (in reverse circuit order), and converts back to a u-gate.
   *
   * @param op The first rotation gate
   * @param user The second rotation gate
   * @param rewriter Pattern rewriter for creating the merged gate
   * @return A u-gate representing the merged rotation
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
   * @brief Matches and merges consecutive rotation gates on the same qubit.
   *
   * Merges two gates using quaternion multiplication when the first gate has
   * exactly one use, replacing both with an equivalent u-gate.
   *
   * @param op The rotation gate to match
   * @param rewriter Pattern rewriter for applying transformations
   * @return success() if gates were merged, failure() otherwise
   */
  mlir::LogicalResult
  matchAndRewrite(UnitaryOpInterface op,
                  mlir::PatternRewriter& rewriter) const override {
    // QCO operations cannot contain control qubits, so no need to check for
    // them
    if (!op->hasOneUse()) {
      return mlir::failure();
    }

    const auto& users = op->getUsers();
    auto* userOP = *users.begin();

    if (!areQuaternionMergeable(*op, *userOP)) {
      return mlir::failure();
    }
    auto user = mlir::dyn_cast<UnitaryOpInterface>(userOP);
    if (!user) {
      return mlir::failure();
    }

    UnitaryOpInterface newUser =
        createOpQuaternionMergedAngle(op, user, rewriter);

    // Replace user with newUser
    rewriter.replaceOp(user, newUser);

    // Erase op
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

/**
 * @brief Populates the given pattern set with the `MergeRotationGatesPattern`.
 *
 * @param patterns The pattern set to populate
 */
static void
populateMergeRotationGatesPatterns(mlir::RewritePatternSet& patterns) {
  patterns.add<MergeRotationGatesPattern>(patterns.getContext());
}

/**
 * @brief Pass that merges consecutive rotation gates using quaternion
 * multiplication.
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
