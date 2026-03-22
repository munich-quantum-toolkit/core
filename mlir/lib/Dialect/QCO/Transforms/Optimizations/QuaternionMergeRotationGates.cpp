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

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <cassert>
#include <cstdint>
#include <numbers>
#include <optional>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_MERGEROTATIONGATES
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

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

  struct Constants {
    Value negOne, zero, one, two, eps, pi;
  };

  /**
   * @brief Checks if an operation is a mergeable rotation gate.
   *
   * @param op The operation to check
   * @return True if mergeable, false otherwise
   */
  static bool isMergeable(Operation* op) {
    return isa<RXOp, RYOp, RZOp, POp, ROp, U2Op, UOp>(op);
  }

  /**
   * @brief Checks if two gates require quaternion-based merging.
   *
   * Returns true for different gate types (e.g., RXOp+RYOp) or same-type
   * multi-parameter gates (UOp, U2Op, ROp). Same-type single-parameter gates
   * (e.g., RXOp+RXOp, POp+POp) use angle addition and aren't handled here.
   *
   * @param a The first gate
   * @param b The second gate
   * @return True if quaternion-based merging should be used, false otherwise
   */
  [[nodiscard]] static bool areQuaternionMergeable(Operation& a, Operation& b) {
    if (!isMergeable(&a) || !isMergeable(&b)) {
      return false;
    }

    // Different gate types always require quaternion merging.
    // Same-type multi-parameter gates (UOp, U2Op, ROp) also require it,
    // since they cannot be merged by simple angle addition.
    return (a.getName() != b.getName()) || isa<UOp, U2Op, ROp>(a);
  }

  /**
   * @brief Returns the rotation axis for an RXOp, RYOp, or RZOp.
   *
   * @param op The operation to query
   * @return The rotation axis, or std::nullopt if the operation is not
   *         RXOp, RYOp, or RZOp.
   */
  static std::optional<RotationAxis> getRotationAxis(Operation* op) {
    return llvm::TypeSwitch<Operation*, std::optional<RotationAxis>>(op)
        .Case<RXOp>([](auto) { return RotationAxis::X; })
        .Case<RYOp>([](auto) { return RotationAxis::Y; })
        .Case<RZOp, POp>([](auto) { return RotationAxis::Z; })
        .Default([](auto) { return std::nullopt; });
  }

  /**
   * @brief Creates shared f64 arithmetic constants used throughout the pass.
   *
   * These constants are created once and reused across quaternion construction,
   * Hamilton product, and Euler angle extraction to avoid redundant ops in the
   * generated IR.
   *
   * @param loc Source location for the created operations
   * @param rewriter Pattern rewriter for creating new operations
   * @return A Constants struct with all pre-built constant ops
   */
  static Constants createConstants(Location loc, PatternRewriter& rewriter) {
    // MLIR types are pointer-sized wrappers;
    // slicing FloatType to Type is safe and intentional.
    // NOLINTNEXTLINE(cppcoreguidelines-slicing)
    Type f64 = rewriter.getF64Type();
    return {
        .negOne = arith::ConstantOp::create(rewriter, loc,
                                            rewriter.getFloatAttr(f64, -1.0)),
        .zero = arith::ConstantOp::create(rewriter, loc,
                                          rewriter.getFloatAttr(f64, 0.0)),
        .one = arith::ConstantOp::create(rewriter, loc,
                                         rewriter.getFloatAttr(f64, 1.0)),
        .two = arith::ConstantOp::create(rewriter, loc,
                                         rewriter.getFloatAttr(f64, 2.0)),
        // Tolerance for gimbal-lock detection in quaternion-to-Euler
        // conversion. Value from reference implementation:
        // https://github.com/evbernardes/quaternion_to_euler/blob/main/euler_from_quat.py
        .eps = arith::ConstantOp::create(rewriter, loc,
                                         rewriter.getFloatAttr(f64, 1e-12)),
        .pi = arith::ConstantOp::create(
            rewriter, loc, rewriter.getFloatAttr(f64, std::numbers::pi)),
    };
  }

  /**
   * @brief Normalizes an angle to the range [-PI, PI].
   *
   * Uses the identity atan2(sin(a), cos(a)) which projects the angle onto the
   * unit circle and recovers the canonical representative in [-PI, PI].
   *
   * @param angle The angle value to normalize
   * @param loc Source location for the created operations
   * @param rewriter Pattern rewriter for creating new operations
   * @return The normalized angle value
   */
  static Value normalizeAngle(Value angle, Location loc,
                              PatternRewriter& rewriter) {
    auto sinA = math::SinOp::create(rewriter, loc, angle);
    auto cosA = math::CosOp::create(rewriter, loc, angle);
    return math::Atan2Op::create(rewriter, loc, sinA, cosA);
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
   * @param constants Pre-created arithmetic constants
   * @param rewriter Pattern rewriter for creating new operations
   * @return Quaternion representing the rotation
   */
  static Quaternion createAxisQuaternion(Value angle, RotationAxis axis,
                                         Location loc,
                                         const Constants& constants,
                                         PatternRewriter& rewriter) {
    // NOLINTNEXTLINE(cppcoreguidelines-slicing)
    Type f64 = rewriter.getF64Type();
    auto half = arith::DivFOp::create(rewriter, loc, angle, constants.two);
    // cos(angle/2)
    auto cos = math::CosOp::create(rewriter, loc, f64, half);
    // sin(angle/2)
    auto sin = math::SinOp::create(rewriter, loc, f64, half);

    switch (axis) {
    case RotationAxis::X:
      return {.w = cos, .x = sin, .y = constants.zero, .z = constants.zero};
    case RotationAxis::Y:
      return {.w = cos, .x = constants.zero, .y = sin, .z = constants.zero};
    case RotationAxis::Z:
      return {.w = cos, .x = constants.zero, .y = constants.zero, .z = sin};
    }
  }

  /**
   * @brief Converts a ZYZ Euler angle decomposition to quaternion.
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
   * @param theta The Y-rotation angle
   * @param phi The first Z-rotation angle
   * @param lambda The second Z-rotation angle
   * @param loc Location in the IR
   * @param constants Pre-created arithmetic constants
   * @param rewriter Pattern rewriter for creating new operations
   * @return Quaternion representing the ZYZ rotation
   */
  static Quaternion quaternionFromZYZ(Value theta, Value phi, Value lambda,
                                      Location loc, const Constants& constants,
                                      PatternRewriter& rewriter) {
    auto qTheta =
        createAxisQuaternion(theta, RotationAxis::Y, loc, constants, rewriter);
    auto qPhi =
        createAxisQuaternion(phi, RotationAxis::Z, loc, constants, rewriter);
    auto qLambda =
        createAxisQuaternion(lambda, RotationAxis::Z, loc, constants, rewriter);

    // qPhi * qTheta * qLambda (multiplication in reverse order!)
    auto temp = hamiltonProduct(qPhi, qTheta, loc, rewriter);
    return hamiltonProduct(temp, qLambda, loc, rewriter);
  }

  /**
   * @brief Converts a UOp to quaternion representation.
   *
   * U(theta, phi, lambda) is decomposed via ZYZ Euler angles.
   *
   * @param op The UOp to convert
   * @param constants Pre-created arithmetic constants
   * @param rewriter Pattern rewriter for creating new operations
   * @return Quaternion representing the UOp
   */
  static Quaternion quaternionFromUOp(UnitaryOpInterface op,
                                      const Constants& constants,
                                      PatternRewriter& rewriter) {
    return quaternionFromZYZ(op.getParameter(0), op.getParameter(1),
                             op.getParameter(2), op->getLoc(), constants,
                             rewriter);
  }

  /**
   * @brief Converts a U2Op to quaternion representation.
   *
   * U2(phi, lambda) = U(pi/2, phi, lambda), using ZYZ decomposition with
   * theta fixed to pi/2.
   *
   * @param op The U2Op to convert
   * @param constants Pre-created arithmetic constants
   * @param rewriter Pattern rewriter for creating new operations
   * @return Quaternion representing the U2Op
   */
  static Quaternion quaternionFromU2Op(UnitaryOpInterface op,
                                       const Constants& constants,
                                       PatternRewriter& rewriter) {
    auto loc = op->getLoc();
    auto piHalf =
        arith::DivFOp::create(rewriter, loc, constants.pi, constants.two);
    return quaternionFromZYZ(piHalf, op.getParameter(0), op.getParameter(1),
                             loc, constants, rewriter);
  }

  /**
   * @brief Converts an ROp to quaternion representation.
   *
   * R(theta, phi) represents a rotation by theta around axis
   * (cos(phi), sin(phi), 0) in the XY plane:
   * Q(cos(theta/2), sin(theta/2)*cos(phi), sin(theta/2)*sin(phi), 0)
   *
   * @param op The ROp to convert
   * @param constants Pre-created arithmetic constants
   * @param rewriter Pattern rewriter for creating new operations
   * @return Quaternion representing the ROp
   */
  static Quaternion quaternionFromROp(UnitaryOpInterface op,
                                      const Constants& constants,
                                      PatternRewriter& rewriter) {
    auto loc = op->getLoc();
    // NOLINTNEXTLINE(cppcoreguidelines-slicing)
    Type f64 = rewriter.getF64Type();
    auto theta = op.getParameter(0);
    auto phi = op.getParameter(1);

    auto halfTheta = arith::DivFOp::create(rewriter, loc, theta, constants.two);
    auto cosHalf = math::CosOp::create(rewriter, loc, f64, halfTheta);
    auto sinHalf = math::SinOp::create(rewriter, loc, f64, halfTheta);
    auto cosPhi = math::CosOp::create(rewriter, loc, f64, phi);
    auto sinPhi = math::SinOp::create(rewriter, loc, f64, phi);

    auto x = arith::MulFOp::create(rewriter, loc, sinHalf, cosPhi);
    auto y = arith::MulFOp::create(rewriter, loc, sinHalf, sinPhi);

    return {.w = cosHalf, .x = x, .y = y, .z = constants.zero};
  }

  /**
   * @brief Converts a rotation gate to quaternion representation.
   *
   * @param op The rotation gate to convert (RXOp, RYOp, RZOp, POp, ROp, U2Op,
   *        UOp)
   * @param constants Pre-created arithmetic constants
   * @param rewriter Pattern rewriter for creating new operations
   * @return Quaternion representing the rotation gate
   */
  static Quaternion quaternionFromRotation(UnitaryOpInterface op,
                                           const Constants& constants,
                                           PatternRewriter& rewriter) {
    // Single-axis rotations (RX, RY, RZ, P) share the same conversion pattern
    if (auto axis = getRotationAxis(op.getOperation())) {
      return createAxisQuaternion(op.getParameter(0), *axis, op->getLoc(),
                                  constants, rewriter);
    }

    // Multi-parameter gates each need their own conversion
    return llvm::TypeSwitch<Operation*, Quaternion>(op.getOperation())
        .Case<ROp>(
            [&](auto) { return quaternionFromROp(op, constants, rewriter); })
        .Case<U2Op>(
            [&](auto) { return quaternionFromU2Op(op, constants, rewriter); })
        .Case<UOp>(
            [&](auto) { return quaternionFromUOp(op, constants, rewriter); })
        .Default([](auto) -> Quaternion {
          llvm_unreachable("Unsupported operation type");
        });
  }

  /**
   * @brief Checks if this op is the start of a mergeable chain.
   *
   * A chain start is a mergeable op whose qubit input does NOT come from
   * a chain-compatible predecessor. This ensures the greedy rewriter only
   * triggers the rewrite at chain heads, building the maximal chain in one
   * shot regardless of worklist order.
   *
   * @param op The operation to check
   * @return True if this op is the start of a chain
   */
  static bool isChainStart(UnitaryOpInterface op) {
    if (!isMergeable(op.getOperation())) {
      return false;
    }
    auto input = op.getInputQubit(0);
    auto* defOp = input.getDefiningOp();
    return defOp == nullptr ||
           !areQuaternionMergeable(*defOp, *op.getOperation());
  }

  /**
   * @brief Collects a chain of consecutive mergeable gates.
   *
   * Walks forward via single-use SSA edges. Breaks when the next operation is
   * not mergeable or would form a same-type single-parameter pair with the
   * current tail (leaving those for canonicalization).
   *
   * @param start The chain head (must satisfy isChainStart)
   * @return The chain of operations in circuit order (first applied to last)
   */
  static SmallVector<UnitaryOpInterface>
  collectChain(UnitaryOpInterface start) {
    SmallVector<UnitaryOpInterface> chain = {start};
    auto current = start;
    while (!current->use_empty()) {
      auto* userOp = *current->getUsers().begin();
      if (!areQuaternionMergeable(*current.getOperation(), *userOp)) {
        break;
      }
      chain.push_back(cast<UnitaryOpInterface>(userOp));
      current = chain.back();
    }
    return chain;
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
   * @param q1 The first quaternion
   * @param q2 The second quaternion
   * @param loc Location in the IR
   * @param rewriter Pattern rewriter for creating new operations
   * @return Product quaternion
   */
  static Quaternion hamiltonProduct(Quaternion q1, Quaternion q2, Location loc,
                                    PatternRewriter& rewriter) {
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
   * @brief Converts a quaternion to a UOp using ZYZ Euler angle extraction.
   *
   * For unit quaternion q = w + x*i + y*j + z*k, extracts UOp parameters:
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
   * @param constants Pre-created arithmetic constants
   * @param rewriter Pattern rewriter for creating new operations
   * @return UOp equivalent to the quaternion rotation
   */
  static UnitaryOpInterface uOpFromQuaternion(Quaternion q,
                                              UnitaryOpInterface op,
                                              const Constants& constants,
                                              PatternRewriter& rewriter) {
    auto loc = op->getLoc();

    // calculate angle beta (for y-rotation)
    // beta = acos(2 * (w^2 + z^2) - 1)
    // NOTE: the term (2 * (w^2 + z^2) - 1) is clamped to [-1, 1],
    // otherwise acos could produce NaN.
    auto ww = arith::MulFOp::create(rewriter, loc, q.w, q.w);
    auto zz = arith::MulFOp::create(rewriter, loc, q.z, q.z);
    auto bTemp1 = arith::AddFOp::create(rewriter, loc, ww, zz);
    auto bTemp2 = arith::MulFOp::create(rewriter, loc, constants.two, bTemp1);
    auto bTemp3 = arith::SubFOp::create(rewriter, loc, bTemp2, constants.one);
    auto clampedLow =
        arith::MaximumFOp::create(rewriter, loc, bTemp3, constants.negOne);
    auto clamped =
        arith::MinimumFOp::create(rewriter, loc, clampedLow, constants.one);
    auto beta = math::AcosOp::create(rewriter, loc, clamped);

    // intermediates to check for gimbal lock (|beta| and |beta - PI|)
    auto absBeta = math::AbsFOp::create(rewriter, loc, beta);
    auto betaMinusPi = arith::SubFOp::create(rewriter, loc, beta, constants.pi);
    auto absBetaMinusPi = math::AbsFOp::create(rewriter, loc, betaMinusPi);

    // safe1 = beta not within boundary eps around 0:
    // |beta| >= eps
    auto safe1 = arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OGE,
                                       absBeta, constants.eps);
    // safe2 = beta not within boundary eps around PI: |beta - PI| >= eps
    auto safe2 = arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OGE,
                                       absBetaMinusPi, constants.eps);
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
    auto twoThetaPlus =
        arith::MulFOp::create(rewriter, loc, constants.two, thetaPlus);
    auto twoThetaMinus =
        arith::MulFOp::create(rewriter, loc, constants.two, thetaMinus);

    // Safe Case (no gimbal lock):
    // alphaSafe = theta+ + theta-
    // gammaSafe = theta+ - theta-
    auto alphaSafe =
        arith::AddFOp::create(rewriter, loc, thetaPlus, thetaMinus);
    auto gammaSafe =
        arith::SubFOp::create(rewriter, loc, thetaPlus, thetaMinus);

    // Unsafe Case (gimbal lock):
    // when beta = 0  then alpha = 2 * (atan2(z, w))
    // when beta = PI then alpha = 2 * (atan2(-x, y))
    // gamma is set to zero in both cases
    auto alphaUnsafe = arith::SelectOp::create(rewriter, loc, safe1,
                                               twoThetaMinus, twoThetaPlus);

    // choose correct alpha and gamma whether safe or not
    auto alpha =
        arith::SelectOp::create(rewriter, loc, safe, alphaSafe, alphaUnsafe);
    auto gamma =
        arith::SelectOp::create(rewriter, loc, safe, gammaSafe, constants.zero);

    // normalize alpha and gamma to [-PI, PI] since they are sums/differences
    // of atan2 results and can exceed that range
    auto alphaNorm = normalizeAngle(alpha, loc, rewriter);
    auto gammaNorm = normalizeAngle(gamma, loc, rewriter);

    return UOp::create(rewriter, loc, op.getInputQubit(0), beta.getResult(),
                       alphaNorm, gammaNorm);
  }

  /**
   * @brief Matches and merges a chain of consecutive rotation gates.
   *
   * Detects the full chain of mergeable operations, folds their quaternions
   * via Hamilton product, and emits a single UOp.
   *
   * @param op The operation to match (only chain heads trigger the rewrite)
   * @param rewriter Pattern rewriter for applying transformations
   * @return success() if operations were merged, failure() otherwise
   */
  LogicalResult matchAndRewrite(UnitaryOpInterface op,
                                PatternRewriter& rewriter) const override {
    if (!isChainStart(op)) {
      return failure();
    }

    auto chain = collectChain(op);
    if (chain.size() < 2) {
      return failure();
    }

    // Emit all helper ops at the chain tail so the merged UOp is placed
    // adjacent to the last gate it replaces.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(chain.back().getOperation());

    auto loc = op->getLoc();
    auto constants = createConstants(loc, rewriter);

    // Initialize quaternion accumulator from the first operation
    auto qAccum = quaternionFromRotation(chain.front(), constants, rewriter);

    // Fold remaining operations via Hamilton product
    for (auto chainOp : llvm::drop_begin(chain)) {
      auto qi = quaternionFromRotation(chainOp, constants, rewriter);
      qAccum = hamiltonProduct(qi, qAccum, loc, rewriter);
    }

    // Convert merged quaternion back to UOp
    auto newOp = uOpFromQuaternion(qAccum, op, constants, rewriter);

    // Bypass and erase each tail op, then replace the head with the merged UOp
    for (auto chainOp : llvm::drop_begin(chain)) {
      rewriter.replaceOp(chainOp, chainOp.getInputQubit(0));
    }
    rewriter.replaceOp(chain.front(), newOp);

    return success();
  }
};

/**
 * @brief Pass that merges consecutive rotation gates using quaternion
 * multiplication.
 */
struct MergeRotationGates final
    : impl::MergeRotationGatesBase<MergeRotationGates> {
  using impl::MergeRotationGatesBase<
      MergeRotationGates>::MergeRotationGatesBase;

protected:
  void runOnOperation() override {
    // Get the current operation being operated on.
    auto op = getOperation();
    auto* ctx = &getContext();

    // Define the set of patterns to use.
    RewritePatternSet patterns(ctx);
    patterns.add<MergeRotationGatesPattern>(patterns.getContext());

    // Apply patterns in an iterative and greedy manner.
    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::qco
