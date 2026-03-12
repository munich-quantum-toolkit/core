/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <optional>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_MERGEROTATIONGATES
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

/**
 * @brief Pattern that merges consecutive rotation gates using quaternion
 * multiplication.
 */
struct MergeRotationGatesPattern final
    : OpInterfaceRewritePattern<UnitaryOpInterface> {
  explicit MergeRotationGatesPattern(MLIRContext* context)
      : OpInterfaceRewritePattern(context) {}

  struct Quaternion {
    Value w;
    Value x;
    Value y;
    Value z;
  };

  enum class RotationAxis : std::uint8_t { X, Y, Z };

  /**
   * @brief Checks if an operation is a mergeable rotation gate (rx, ry, rz, u).
   *
   * @param op The operation to check
   * @return True if mergeable, false otherwise
   */
  static bool isMergeable(Operation* op) {
    return isa<RXOp, RYOp, RZOp, UOp>(op);
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
  [[nodiscard]] static bool areQuaternionMergeable(Operation& a, Operation& b) {
    if (!isMergeable(&a) || !isMergeable(&b)) {
      return false;
    }

    // Different gate types OR both are U gates
    return (a.getName() != b.getName()) || (isa<UOp>(a) && isa<UOp>(b));
  }

  /**
   * @brief Returns the rotation axis for a single-axis rotation gate.
   *
   * @param op The operation to query
   * @return The rotation axis, or std::nullopt if the operation is not a
   *         single-axis rotation gate (RX, RY, RZ)
   */
  static std::optional<RotationAxis> getRotationAxis(Operation* op) {
    return llvm::TypeSwitch<Operation*, std::optional<RotationAxis>>(op)
        .Case<RXOp>([](auto) { return RotationAxis::X; })
        .Case<RYOp>([](auto) { return RotationAxis::Y; })
        .Case<RZOp>([](auto) { return RotationAxis::Z; })
        .Default([](auto) { return std::nullopt; });
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
  static Quaternion createAxisQuaternion(Value angle, RotationAxis axis,
                                         Location loc,
                                         PatternRewriter& rewriter) {
    auto floatType = angle.getType();

    // constant 0.0
    auto zeroAttr = rewriter.getFloatAttr(floatType, 0.0);
    auto zero = arith::ConstantOp::create(rewriter, loc, zeroAttr);

    // constant 2.0
    auto twoAttr = rewriter.getFloatAttr(floatType, 2.0);
    auto two = arith::ConstantOp::create(rewriter, loc, twoAttr);

    auto half = arith::DivFOp::create(rewriter, loc, angle, two);
    // cos(angle/2)
    auto cos = math::CosOp::create(rewriter, loc, floatType, half);
    // sin(angle/2)
    auto sin = math::SinOp::create(rewriter, loc, floatType, half);

    switch (axis) {
    case RotationAxis::X:
      return {.w = cos, .x = sin, .y = zero, .z = zero};
    case RotationAxis::Y:
      return {.w = cos, .x = zero, .y = sin, .z = zero};
    case RotationAxis::Z:
      return {.w = cos, .x = zero, .y = zero, .z = sin};
    } // NOLINT(bugprone-branch-clone): false positive, branches differ

    llvm_unreachable("Invalid rotation axis");
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
                                           PatternRewriter& rewriter) {
    if (isa<UOp>(op)) {
      return quaternionFromUGate(op, rewriter);
    }

    if (auto axis = getRotationAxis(op.getOperation())) {
      return createAxisQuaternion(op.getParameter(0), *axis, op->getLoc(),
                                  rewriter);
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
                                    PatternRewriter& rewriter) {
    auto loc = op->getLoc();

    // wRes = w1w2 - x1x2 - y1y2 - z1z2
    auto w1w2 = arith::MulFOp::create(rewriter, loc, q1.w, q2.w);
    auto x1x2 = arith::MulFOp::create(rewriter, loc, q1.x, q2.x);
    auto y1y2 = arith::MulFOp::create(rewriter, loc, q1.y, q2.y);
    auto z1z2 = arith::MulFOp::create(rewriter, loc, q1.z, q2.z);
    auto wTemp1 = arith::SubFOp::create(rewriter, loc, w1w2, x1x2);
    auto wTemp2 = arith::SubFOp::create(rewriter, loc, wTemp1, y1y2);
    auto wRes = arith::SubFOp::create(rewriter, loc, wTemp2, z1z2);

    // xRes = w1x2 + x1w2 + y1z2 - z1y2
    auto w1x2 = arith::MulFOp::create(rewriter, loc, q1.w, q2.x);
    auto x1w2 = arith::MulFOp::create(rewriter, loc, q1.x, q2.w);
    auto y1z2 = arith::MulFOp::create(rewriter, loc, q1.y, q2.z);
    auto z1y2 = arith::MulFOp::create(rewriter, loc, q1.z, q2.y);
    auto xTemp1 = arith::AddFOp::create(rewriter, loc, w1x2, x1w2);
    auto xTemp2 = arith::AddFOp::create(rewriter, loc, xTemp1, y1z2);
    auto xRes = arith::SubFOp::create(rewriter, loc, xTemp2, z1y2);

    // yRes = w1y2 - x1z2 + y1w2 + z1x2
    auto w1y2 = arith::MulFOp::create(rewriter, loc, q1.w, q2.y);
    auto x1z2 = arith::MulFOp::create(rewriter, loc, q1.x, q2.z);
    auto y1w2 = arith::MulFOp::create(rewriter, loc, q1.y, q2.w);
    auto z1x2 = arith::MulFOp::create(rewriter, loc, q1.z, q2.x);
    auto yTemp1 = arith::SubFOp::create(rewriter, loc, w1y2, x1z2);
    auto yTemp2 = arith::AddFOp::create(rewriter, loc, yTemp1, y1w2);
    auto yRes = arith::AddFOp::create(rewriter, loc, yTemp2, z1x2);

    // zRes = w1z2 + x1y2 - y1x2 + z1w2
    auto w1z2 = arith::MulFOp::create(rewriter, loc, q1.w, q2.z);
    auto x1y2 = arith::MulFOp::create(rewriter, loc, q1.x, q2.y);
    auto y1x2 = arith::MulFOp::create(rewriter, loc, q1.y, q2.x);
    auto z1w2 = arith::MulFOp::create(rewriter, loc, q1.z, q2.w);
    auto zTemp1 = arith::AddFOp::create(rewriter, loc, w1z2, x1y2);
    auto zTemp2 = arith::SubFOp::create(rewriter, loc, zTemp1, y1x2);
    auto zRes = arith::AddFOp::create(rewriter, loc, zTemp2, z1w2);

    return {.w = wRes, .x = xRes, .y = yRes, .z = zRes};
  }

  /**
   * @brief Converts a u-gate to quaternion representation.
   *
   * U(theta, phi, lambda) uses ZYZ decomposition: RZ(lambda) -> RY(theta) ->
   * RZ(phi).
   *
   * When composing rotations, quaternion multiplication follows matrix
   * multiplication order (right-to-left), which is the reverse of the
   * application sequence:
   *   Sequential application: RZ(lambda), then RY(theta), then RZ(phi)
   *   Quaternion product:     qPhi * qTheta * qLambda
   *
   * @param op The u-gate operation to convert
   * @param rewriter Pattern rewriter for creating new operations
   * @return Quaternion representing the u-gate
   */
  static Quaternion quaternionFromUGate(UnitaryOpInterface op,
                                        PatternRewriter& rewriter) {
    auto loc = op->getLoc();

    // U gate uses ZYZ decomposition:
    // U(theta, phi, lambda) uses ZYZ decomposition: RZ(lambda) -> RY(theta) ->
    // RZ(phi)
    auto qTheta = createAxisQuaternion(op.getParameter(0), RotationAxis::Y, loc,
                                       rewriter);
    auto qPhi = createAxisQuaternion(op.getParameter(1), RotationAxis::Z, loc,
                                     rewriter);
    auto qLambda = createAxisQuaternion(op.getParameter(2), RotationAxis::Z,
                                        loc, rewriter);

    // qPhi * qTheta * qLambda (multiplication in reverse order!)
    auto temp = hamiltonProduct(qPhi, qTheta, op, rewriter);
    return hamiltonProduct(temp, qLambda, op, rewriter);
  }

  /**
   * @brief Converts a quaternion to a u-gate using ZYZ Euler angle extraction.
   *
   * For unit quaternion q = w + x*i + y*j + z*k, extracts u-gate parameters:
   *   alpha = atan2(z, w) + atan2(-x, y)
   *   beta  = acos(2 * (w^2 + z^2) - 1)
   *   gamma = atan2(z, w) - atan2(-x, y)
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
  static UnitaryOpInterface uGateFromQuaternion(Quaternion q,
                                                UnitaryOpInterface op,
                                                PatternRewriter& rewriter) {
    auto loc = op->getLoc();

    auto floatType = op.getParameter(0).getType();
    // constant -1.0
    auto negOneAttr = rewriter.getFloatAttr(floatType, -1.0);
    auto negOne = arith::ConstantOp::create(rewriter, loc, negOneAttr);
    // constant 0.0
    auto zeroAttr = rewriter.getFloatAttr(floatType, 0.0);
    auto zero = arith::ConstantOp::create(rewriter, loc, zeroAttr);
    // constant 1.0
    auto oneAttr = rewriter.getFloatAttr(floatType, 1.0);
    auto one = arith::ConstantOp::create(rewriter, loc, oneAttr);
    // constant 2.0
    auto twoAttr = rewriter.getFloatAttr(floatType, 2.0);
    auto two = arith::ConstantOp::create(rewriter, loc, twoAttr);
    // constant epsilon (boundary around gimbal lock positions)
    auto epsAttr = rewriter.getFloatAttr(floatType, 1e-7);
    auto eps = arith::ConstantOp::create(rewriter, loc, epsAttr);
    // constant PI
    auto piAttr = rewriter.getFloatAttr(floatType, std::numbers::pi);
    auto pi = arith::ConstantOp::create(rewriter, loc, piAttr);

    // calculate angle beta (for y-rotation)
    // beta = acos(2 * (w^2 + z^2) - 1)
    // NOTE: the term (2 * (w^2 + z^2) - 1) is clamped to [-1, 1],
    // otherwise acos could produce NaN.
    auto ww = arith::MulFOp::create(rewriter, loc, q.w, q.w);
    auto zz = arith::MulFOp::create(rewriter, loc, q.z, q.z);
    auto bTemp1 = arith::AddFOp::create(rewriter, loc, ww, zz);
    auto bTemp2 = arith::MulFOp::create(rewriter, loc, two, bTemp1);
    auto bTemp3 = arith::SubFOp::create(rewriter, loc, bTemp2, one);
    auto clampedLow = arith::MaximumFOp::create(rewriter, loc, bTemp3, negOne);
    auto clamped = arith::MinimumFOp::create(rewriter, loc, clampedLow, one);
    auto beta = math::AcosOp::create(rewriter, loc, clamped);

    // intermediates to check for gimbal lock (|beta| and |beta - PI|)
    auto absBeta = math::AbsFOp::create(rewriter, loc, beta);
    auto betaMinusPi = arith::SubFOp::create(rewriter, loc, beta, pi);
    auto absBetaMinusPi = math::AbsFOp::create(rewriter, loc, betaMinusPi);

    // safe1 = beta not within boundary eps around 0:
    // |beta| >= eps
    auto safe1 = arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OGE,
                                       absBeta, eps);
    // safe2 = beta not within boundary eps around PI: |beta-pi| >= eps
    auto safe2 = arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OGE,
                                       absBetaMinusPi, eps);
    // is safe (not in gimbal lock) when both hold (safe1 AND safe2)
    auto safe = arith::AndIOp::create(rewriter, loc, safe1, safe2);

    // intermediate angles for z-rotations alpha and gamma
    // theta+ = atan2(z, w)
    // theta- = atan2(-x, y)
    auto xMinus = arith::NegFOp::create(rewriter, loc, q.x);
    auto thetaPlus = math::Atan2Op::create(rewriter, loc, q.z, q.w);
    auto thetaMinus = math::Atan2Op::create(rewriter, loc, xMinus, q.y);

    // intermediate angles for gimbal lock cases
    // twoTheta+ = 2 * theta+
    // twoTheta- = 2 * theta-
    auto twoThetaPlus = arith::MulFOp::create(rewriter, loc, two, thetaPlus);
    auto twoThetaMinus = arith::MulFOp::create(rewriter, loc, two, thetaMinus);

    // Safe Case (no gimbal lock):
    // alphaSafe = theta+ + theta-
    // gammaSafe = theta+ - theta-
    auto alphaSafe =
        arith::AddFOp::create(rewriter, loc, thetaPlus, thetaMinus);
    auto gammaSafe =
        arith::SubFOp::create(rewriter, loc, thetaPlus, thetaMinus);

    // Unsafe Case (gimbal lock):
    // when b = 0  then alpha = 2 * (atan2(z,w))
    // when b = PI then alpha = 2 * (atan2(-x, y))
    // gamma is set to zero in both cases
    auto alphaUnsafe = arith::SelectOp::create(rewriter, loc, safe1,
                                               twoThetaMinus, twoThetaPlus);

    // TODO: could add some normalization here for alpha and gamma otherwise
    // they can be outside of [-PI, PI].

    // choose correct alpha and gamma whether safe or not
    auto alpha =
        arith::SelectOp::create(rewriter, loc, safe, alphaSafe, alphaUnsafe);
    auto gamma = arith::SelectOp::create(rewriter, loc, safe, gammaSafe, zero);

    return UOp::create(rewriter, loc, op.getInputQubit(0), beta.getResult(),
                       alpha.getResult(), gamma.getResult());
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
                                PatternRewriter& rewriter) {
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
  LogicalResult matchAndRewrite(UnitaryOpInterface op,
                                PatternRewriter& rewriter) const override {
    // QCO operations cannot contain control qubits, so no need to check for
    // them
    if (!op->hasOneUse()) {
      return failure();
    }

    const auto& users = op->getUsers();
    auto* userOP = *users.begin();

    if (!areQuaternionMergeable(*op, *userOP)) {
      return failure();
    }
    auto user = dyn_cast<UnitaryOpInterface>(userOP);
    assert(user && "Cannot cast to UnitaryOpInterface, mergeable gates must "
                   "implement UnitaryOpInterface");

    rewriter.setInsertionPoint(user);
    const UnitaryOpInterface newUser =
        createOpQuaternionMergedAngle(op, user, rewriter);

    // Replace user with newUser
    rewriter.replaceOp(user, newUser);

    // Erase op
    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Populates the given pattern set with the `MergeRotationGatesPattern`.
 *
 * @param patterns The pattern set to populate
 */
static void populateMergeRotationGatesPatterns(RewritePatternSet& patterns) {
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
    RewritePatternSet patterns(ctx);
    populateMergeRotationGatesPatterns(patterns);

    // Apply patterns in an iterative and greedy manner.
    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mlir::qco
