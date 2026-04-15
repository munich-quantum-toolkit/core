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
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <cassert>
#include <cstdint>
#include <numbers>
#include <optional>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_MERGESINGLEQUBITROTATIONGATES
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

/**
 * @brief Pattern that merges consecutive rotation gates using quaternion
 * multiplication.
 */
struct MergeSingleQubitRotationGatesPattern final
    : OpInterfaceRewritePattern<UnitaryOpInterface> {
  explicit MergeSingleQubitRotationGatesPattern(MLIRContext* context)
      : OpInterfaceRewritePattern(context) {}

  /// Quaternion representation (w + xi + yj + zk) using MLIR Values.
  struct Quaternion {
    Value w;
    Value x;
    Value y;
    Value z;
  };

  /// Axis of a single-axis rotation gate.
  enum class RotationAxis : std::uint8_t { X, Y, Z };

  /// Cached frequently-used constant Values.
  struct Constants {
    Value negOne;
    Value zero;
    Value one;
    Value two;
    Value eps;
    Value pi;
  };

  /// Euler-angle triple for a U gate (theta, phi, lambda).
  struct UOpAngles {
    Value theta;
    Value phi;
    Value lambda;
  };

  /// Returns whether an operation is considered mergeable
  static bool isMergeable(Operation* op) {
    return isa<RXOp, RYOp, RZOp, POp, ROp, U2Op, UOp>(op);
  }

  /// Checks if two gates a and b are mergeable via quaternion-based merging.
  [[nodiscard]] static bool areQuaternionMergeable(Operation& a, Operation& b) {
    return isMergeable(&a) && isMergeable(&b);
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
    return {
        .negOne = arith::ConstantFloatOp::create(
            rewriter, loc, rewriter.getF64Type(), APFloat(-1.0)),
        .zero = arith::ConstantFloatOp::create(
            rewriter, loc, rewriter.getF64Type(), APFloat(0.0)),
        .one = arith::ConstantFloatOp::create(
            rewriter, loc, rewriter.getF64Type(), APFloat(1.0)),
        .two = arith::ConstantFloatOp::create(
            rewriter, loc, rewriter.getF64Type(), APFloat(2.0)),
        // Tolerance for gimbal-lock detection in quaternion-to-Euler
        // conversion. Value from reference implementation:
        // https://github.com/evbernardes/quaternion_to_euler/blob/main/euler_from_quat.py
        .eps = arith::ConstantFloatOp::create(
            rewriter, loc, rewriter.getF64Type(), APFloat(1e-12)),
        .pi = arith::ConstantFloatOp::create(
            rewriter, loc, rewriter.getF64Type(), APFloat(std::numbers::pi)),
    };
  }

  /**
   * @brief Normalizes an angle to the range [-PI, PI].
   *
   * Uses floor-based modular arithmetic:
   *   normalize(a) = a - floor((a + π) / 2π) * 2π
   *
   * @param angle The angle value to normalize
   * @param loc Source location for the created operations
   * @param constants Pre-created arithmetic constants
   * @param rewriter Pattern rewriter for creating new operations
   * @return The normalized angle value
   */
  static Value normalizeAngle(Value angle, Location loc,
                              const Constants& constants,
                              PatternRewriter& rewriter) {
    auto twoPi =
        arith::MulFOp::create(rewriter, loc, constants.two, constants.pi);
    auto shifted = arith::AddFOp::create(rewriter, loc, angle, constants.pi);
    auto divided = arith::DivFOp::create(rewriter, loc, shifted, twoPi);
    auto floored = math::FloorOp::create(rewriter, loc, divided);
    auto multiple = arith::MulFOp::create(rewriter, loc, floored, twoPi);
    return arith::SubFOp::create(rewriter, loc, angle, multiple);
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
    auto half = arith::DivFOp::create(rewriter, loc, angle, constants.two);
    // cos(angle/2)
    auto cos = math::CosOp::create(rewriter, loc, half);
    // sin(angle/2)
    auto sin = math::SinOp::create(rewriter, loc, half);

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
   * @note U is defined as P(phi)*RY(theta)*P(lambda), which equals
   * e^{i*(phi+lambda)/2} * RZ(phi)*RY(theta)*RZ(lambda).
   * Since quaternions represent SU(2), this pass works with the SU(2) part
   * RZ(phi)*RY(theta)*RZ(lambda) and tracks the factored-out global phase
   * (phi+lambda)/2 separately via globalPhaseOf.
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
   * @note Global phase is discarded; see quaternionFromZYZ for details.
   *
   * @param op The UOp to convert
   * @param constants Pre-created arithmetic constants
   * @param rewriter Pattern rewriter for creating new operations
   * @return Quaternion representing the UOp
   */
  static Quaternion quaternionFromUOp(UOp op, const Constants& constants,
                                      PatternRewriter& rewriter) {
    return quaternionFromZYZ(op.getParameter(0), op.getParameter(1),
                             op.getParameter(2), op->getLoc(), constants,
                             rewriter);
  }

  /**
   * @brief Converts a U2Op to quaternion representation.
   *
   * U2(phi, lambda) = U(pi / 2, phi, lambda), using ZYZ decomposition with
   * theta = pi / 2.
   *
   * @note Global phase is discarded; see quaternionFromZYZ for details.
   *
   * @param op The U2Op to convert
   * @param constants Pre-created arithmetic constants
   * @param rewriter Pattern rewriter for creating new operations
   * @return Quaternion representing the U2Op
   */
  static Quaternion quaternionFromU2Op(U2Op op, const Constants& constants,
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
   * Q(cos(theta / 2), sin(theta / 2) * cos(phi), sin(theta / 2) * sin(phi), 0)
   *
   * @param op The ROp to convert
   * @param constants Pre-created arithmetic constants
   * @param rewriter Pattern rewriter for creating new operations
   * @return Quaternion representing the ROp
   */
  static Quaternion quaternionFromROp(ROp op, const Constants& constants,
                                      PatternRewriter& rewriter) {
    auto loc = op->getLoc();
    auto theta = op.getParameter(0);
    auto phi = op.getParameter(1);

    auto halfTheta = arith::DivFOp::create(rewriter, loc, theta, constants.two);
    auto cosHalf = math::CosOp::create(rewriter, loc, halfTheta);
    auto sinHalf = math::SinOp::create(rewriter, loc, halfTheta);
    auto cosPhi = math::CosOp::create(rewriter, loc, phi);
    auto sinPhi = math::SinOp::create(rewriter, loc, phi);

    auto x = arith::MulFOp::create(rewriter, loc, sinHalf, cosPhi);
    auto y = arith::MulFOp::create(rewriter, loc, sinHalf, sinPhi);

    return {.w = cosHalf, .x = x, .y = y, .z = constants.zero};
  }

  /**
   * @brief Converts a rotation gate to quaternion representation.
   *
   * @note Global phase is discarded; see quaternionFromZYZ for details.
   *
   * @param op The rotation gate to convert (RXOp, RYOp, RZOp, POp, ROp, U2Op,
   * UOp)
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
            [&](ROp o) { return quaternionFromROp(o, constants, rewriter); })
        .Case<U2Op>(
            [&](U2Op o) { return quaternionFromU2Op(o, constants, rewriter); })
        .Case<UOp>(
            [&](UOp o) { return quaternionFromUOp(o, constants, rewriter); })
        .Default([](auto) -> Quaternion {
          llvm_unreachable("Unsupported operation type");
        });
  }

  /**
   * @brief Returns the global phase contribution of a rotation gate.
   *
   * Rotation gates can be factored as U = e^{i * phase} * SU(2), where SU(2)
   * is the quaternion-representable part and phase is the global phase. This
   * function returns the global phase for each gate type:
   *
   * - RX, RY, RZ, R         -> none (already SU(2), no global phase)
   * - P(theta)              -> theta / 2 (P = e^{i * theta / 2} * RZ(theta))
   * - U(theta, phi, lambda) -> (phi + lambda) / 2
   * - U2(phi, lambda)       -> (phi + lambda) / 2
   *
   * @param op The rotation gate to query
   * @param constants Pre-created arithmetic constants
   * @param loc Source location for created operations
   * @param rewriter Pattern rewriter for creating new operations
   * @return The global phase as a Value, or std::nullopt for SU(2) gates
   */
  static std::optional<Value> globalPhaseOf(UnitaryOpInterface op,
                                            const Constants& constants,
                                            Location loc,
                                            PatternRewriter& rewriter) {
    return llvm::TypeSwitch<Operation*, std::optional<Value>>(op.getOperation())
        .Case<RXOp, RYOp, RZOp, ROp>(
            [&](auto) -> std::optional<Value> { return std::nullopt; })
        .Case<POp>([&](auto) -> std::optional<Value> {
          return arith::DivFOp::create(rewriter, loc, op.getParameter(0),
                                       constants.two);
        })
        .Case<UOp, U2Op>([&](auto) -> std::optional<Value> {
          // phi is at different indexes for UOp and U2Op
          auto phiIdx = isa<UOp>(op.getOperation()) ? 1U : 0U;
          auto sum =
              arith::AddFOp::create(rewriter, loc, op.getParameter(phiIdx),
                                    op.getParameter(phiIdx + 1));
          return arith::DivFOp::create(rewriter, loc, sum, constants.two);
        })
        .Default([](auto) -> std::optional<Value> {
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
   * not considered as mergeable.
   *
   * @param start The chain head (must satisfy isChainStart)
   * @return The chain of operations in circuit order (first applied to last)
   */
  static SmallVector<UnitaryOpInterface>
  collectChain(UnitaryOpInterface start) {
    SmallVector<UnitaryOpInterface> chain = {start};
    auto current = start;
    while (true) {
      auto* userOp = *current->getUsers().begin();
      if (!areQuaternionMergeable(*current.getOperation(), *userOp)) {
        break;
      }
      current = chain.emplace_back(cast<UnitaryOpInterface>(userOp));
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
   * @brief Extracts ZYZ Euler angles from a unit quaternion.
   *
   * For unit quaternion q = w + x * i + y * j + z * k, extracts UOp parameters:
   *
   * - alpha = atan2(z, w) + atan2(-x, y)
   * - beta  = acos(2 * (w^2 + z^2) - 1)
   * - gamma = atan2(z, w) - atan2(-x, y)
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
   * @param loc Source location for the created operations
   * @param constants Pre-created arithmetic constants
   * @param rewriter Pattern rewriter for creating new operations
   * @return UOpAngles {theta, phi, lambda} suitable for UOp::create
   */
  static UOpAngles anglesFromQuaternion(Quaternion q, Location loc,
                                        const Constants& constants,
                                        PatternRewriter& rewriter) {
    // Calculate angle beta (for y-rotation)
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
    auto alphaNorm = normalizeAngle(alpha, loc, constants, rewriter);
    auto gammaNorm = normalizeAngle(gamma, loc, constants, rewriter);

    return {.theta = beta.getResult(), .phi = alphaNorm, .lambda = gammaNorm};
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

    // Initialize accumulators from the first operation
    auto qAccum = quaternionFromRotation(chain.front(), constants, rewriter);
    auto phaseAccum = globalPhaseOf(chain.front(), constants, loc, rewriter);

    // Fold remaining operations via Hamilton product
    for (auto chainOp : llvm::drop_begin(chain)) {
      auto qi = quaternionFromRotation(chainOp, constants, rewriter);
      qAccum = hamiltonProduct(qi, qAccum, loc, rewriter);

      if (auto phase = globalPhaseOf(chainOp, constants, loc, rewriter)) {
        phaseAccum = phaseAccum ? Value(arith::AddFOp::create(
                                      rewriter, loc, *phaseAccum, *phase))
                                : phase;
      }
    }

    // Extract Euler angles from merged quaternion
    auto [theta, phi, lambda] =
        anglesFromQuaternion(qAccum, loc, constants, rewriter);

    // Emit global phase correction:
    //   The synthesized UOp carries an intrinsic phase
    //   outPhase = (phi + lambda) / 2 that must always be compensated.
    //   correction = totalInputPhase - outPhase
    auto phiPlusLambda = arith::AddFOp::create(rewriter, loc, phi, lambda);
    auto outPhase =
        arith::DivFOp::create(rewriter, loc, phiPlusLambda, constants.two);
    Value inputPhase = phaseAccum.value_or(constants.zero);
    auto correction =
        arith::SubFOp::create(rewriter, loc, inputPhase, outPhase);
    GPhaseOp::create(rewriter, loc, correction.getResult());

    // Replace the tail with the merged UOp;
    // the rest of the chain is now unused and will be deleted by DCE
    rewriter.replaceOpWithNewOp<UOp>(
        chain.back(), chain.front().getInputQubit(0), theta, phi, lambda);

    return success();
  }
};

/**
 * @brief Pass that merges consecutive rotation gates using quaternion
 * multiplication.
 */
struct MergeSingleQubitRotationGates final
    : impl::MergeSingleQubitRotationGatesBase<MergeSingleQubitRotationGates> {
  using impl::MergeSingleQubitRotationGatesBase<
      MergeSingleQubitRotationGates>::MergeSingleQubitRotationGatesBase;

protected:
  void runOnOperation() override {
    auto op = getOperation();
    auto* ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<MergeSingleQubitRotationGatesPattern>(patterns.getContext());

    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::qco
