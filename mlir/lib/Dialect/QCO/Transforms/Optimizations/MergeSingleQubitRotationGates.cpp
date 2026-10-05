/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/Transforms/GlobalPhaseNormalization.h"
#include "mqt/Dialect/MQT/Utils/ConstantFolding.h"
#include "mqt/Dialect/MQT/Utils/Parameters.h"
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mqt/Dialect/QCO/Transforms/NativeSynthesis/SingleQubitFusion.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"
#include "mqt/Dialect/QCO/Utils/Matrix.h"
#include "mqt/Dialect/QCO/Utils/WireIterator.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numbers>
#include <optional>
#include <type_traits>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_MERGESINGLEQUBITROTATIONGATES
#include "mqt/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

using Val = mqt::FloatExpression;

enum class RotationAxis : uint8_t { X, Y, Z };

/// Unit quaternion w + x i + y j + z k over runtime scalars.
struct Quat {
  Val w;
  Val x;
  Val y;
  Val z;
};

/// Shared numeric constants used by quaternion construction and Euler extract.
struct ScalarConsts {
  Val zero;
  Val one;
  Val two;
  Val eps;
  Val pi;
};

struct RuntimeEulerAngles {
  Val theta;
  Val phi;
  Val lambda;
  Val phase;
};

} // namespace

/// Creates shared f64 constants for the merge algorithm.
///
/// `eps` (1e-12) is the gimbal-lock tolerance from the reference
/// implementation:
/// https://github.com/evbernardes/quaternion_to_euler/blob/main/euler_from_quat.py
static ScalarConsts makeConsts(RewriterBase& rewriter, Location loc) {
  auto c = [&](double x) { return Val::constant(rewriter, loc, x); };
  return {
      .zero = c(0.0),
      .one = c(1.0),
      .two = c(2.0),
      .eps = c(1e-12),
      .pi = c(std::numbers::pi),
  };
}

/// Normalizes an angle to the range [-π, π].
///
/// Uses floor-based modular arithmetic:
///   normalize(a) = a - floor((a + π) / 2π) * 2π
static Val wrapToPi(Val angle, const ScalarConsts& c) {
  const auto twoPi = c.two * c.pi;
  const auto shifted = angle + c.pi;
  const auto turns = shifted / twoPi;
  const auto floored = turns.floor();
  return angle - floored * twoPi;
}

/// Computes the Hamilton product of two quaternions (q1 * q2).
///
/// For q1 = w1 + x1*i + y1*j + z1*k and q2 = w2 + x2*i + y2*j + z2*k:
///
/// q1 * q2 = (w1w2 - x1x2 - y1y2 - z1z2)
///         + (w1x2 + x1w2 + y1z2 - z1y2) * i
///         + (w1y2 - x1z2 + y1w2 + z1x2) * j
///         + (w1z2 + x1y2 - y1x2 + z1w2) * k
///
/// @see https://en.wikipedia.org/wiki/Quaternion#Hamilton_product
static Quat hamiltonProduct(const Quat& q1, const Quat& q2) {
  return {
      .w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
      .x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
      .y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
      .z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
  };
}

/// Converts a single-axis rotation to quaternion representation.
///
/// Uses half-angle formulas:
///   RX(a) = Q(cos(a/2), sin(a/2), 0, 0)
///   RY(a) = Q(cos(a/2), 0, sin(a/2), 0)
///   RZ(a) = Q(cos(a/2), 0, 0, sin(a/2))
///
/// @see
/// https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
static Quat axisQuaternion(Val angle, RotationAxis axis,
                           const ScalarConsts& c) {
  const auto half = angle / c.two;
  const auto cos = half.cos();
  const auto sin = half.sin();
  switch (axis) {
  case RotationAxis::X:
    return {.w = cos, .x = sin, .y = c.zero, .z = c.zero};
  case RotationAxis::Y:
    return {.w = cos, .x = c.zero, .y = sin, .z = c.zero};
  case RotationAxis::Z:
    return {.w = cos, .x = c.zero, .y = c.zero, .z = sin};
  }
  llvm_unreachable("invalid rotation axis");
}

/// Converts a ZYZ Euler angle decomposition to quaternion.
///
/// U(θ, φ, λ) uses ZYZ decomposition: RZ(λ) → RY(θ) → RZ(φ).
///
/// When composing rotations, quaternion multiplication follows matrix
/// multiplication order (right-to-left), which is the reverse of the
/// application sequence:
///   Sequential application: RZ(λ), then RY(θ), then RZ(φ)
///   Quaternion product:     qPhi * qTheta * qLambda
///
/// @note U is defined as P(φ)*RY(θ)*P(λ), which equals e^{i*(φ+λ)/2} *
/// RZ(φ)*RY(θ)*RZ(λ). Since quaternions represent SU(2), this pass works with
/// the SU(2) part RZ(φ)*RY(θ)*RZ(λ) and tracks the factored-out global phase
/// (φ+λ)/2 separately.
static Quat quaternionFromZYZ(Val theta, Val phi, Val lambda,
                              const ScalarConsts& c) {
  const auto qTheta = axisQuaternion(theta, RotationAxis::Y, c);
  const auto qPhi = axisQuaternion(phi, RotationAxis::Z, c);
  const auto qLambda = axisQuaternion(lambda, RotationAxis::Z, c);
  /// Expand the sparse axis products without combining the input angles.
  const auto w = qPhi.w * qTheta.w;
  const auto yz = qPhi.z * qTheta.y;
  const auto x = -yz;
  const auto y = qPhi.w * qTheta.y;
  const auto z = qPhi.z * qTheta.w;
  return {
      .w = w * qLambda.w - z * qLambda.z,
      .x = x * qLambda.w + y * qLambda.z,
      .y = y * qLambda.w - x * qLambda.z,
      .z = w * qLambda.z + z * qLambda.w,
  };
}

/// Returns the rotation axis for an RXOp, RYOp, RZOp, or POp.
static std::optional<RotationAxis> getRotationAxis(Operation* op) {
  return TypeSwitch<Operation*, std::optional<RotationAxis>>(op)
      .Case([](RXOp) { return RotationAxis::X; })
      .Case([](RYOp) { return RotationAxis::Y; })
      .Case<RZOp, POp>([](auto) { return RotationAxis::Z; })
      .Default([](auto) { return std::nullopt; });
}

/// Normalize gate angles with the synthesis SU(2) contract.
static Val normalizeGateAngle(Val angle) {
  const auto normalized = decomposition::normalizeRotationParameter(
      angle.getBuilder(), angle.getLoc(), angle.getValue());
  return {angle.getBuilder(), angle.getLoc(),
          mqt::variantToValue(angle.getBuilder(), angle.getLoc(), normalized)};
}

static Val gateParam(UnitaryOpInterface op, unsigned i, RewriterBase& rewriter,
                     Location loc) {
  Value p = op.getParameter(i);
  return normalizeGateAngle(Val(rewriter, loc, p));
}

/// Constant gates share the matrix contract. Only runtime gates need symbolic
/// formulas; return their phase with the quaternion so each parameter is read
/// once.
static std::pair<Quat, Val> quaternionFromGate(UnitaryOpInterface op,
                                               const ScalarConsts& c,
                                               RewriterBase& rewriter) {
  const Location loc = op->getLoc();
  const auto constant = [&](double value) {
    return Val::constant(rewriter, loc, value);
  };
  if (const auto matrix = op.getUnitaryMatrix<Matrix2x2>()) {
    const double phase = std::arg(matrix->determinant()) / 2.;
    const auto su = *matrix * std::polar(1., -phase);
    return {{
                .w = constant(su(0, 0).real()),
                .x = constant(-su(1, 0).imag()),
                .y = constant(su(1, 0).real()),
                .z = constant(-su(0, 0).imag()),
            },
            constant(phase)};
  }
  const auto param = [&](unsigned i) {
    return gateParam(op, i, rewriter, loc);
  };
  if (const auto axis = getRotationAxis(op.getOperation())) {
    const auto angle = param(0);
    return {axisQuaternion(angle, *axis, c),
            isa<POp>(op) ? angle / c.two : c.zero};
  }
  if (isa<ROp>(op)) {
    const auto theta = param(0) / c.two;
    const auto phi = param(1);
    const auto sine = theta.sin();
    return {{
                .w = theta.cos(),
                .x = sine * phi.cos(),
                .y = sine * phi.sin(),
                .z = c.zero,
            },
            c.zero};
  }
  assert((isa<UOp, U2Op>(op)) && "unsupported quaternion gate");
  const bool u2 = isa<U2Op>(op);
  const auto theta = u2 ? c.pi / c.two : param(0);
  const auto phi = param(u2 ? 0 : 1);
  const auto lambda = param(u2 ? 1 : 2);
  return {quaternionFromZYZ(theta, phi, lambda, c), (phi + lambda) / c.two};
}

/// Extracts ZYZ Euler angles from a unit quaternion.
///
/// For unit quaternion q = w + x * i + y * j + z * k, extracts UOp parameters:
///
/// - α = atan2(z, w) + atan2(-x, y)
/// - β = 2 * atan2(sqrt(x^2 + y^2), sqrt(w^2 + z^2))
/// - γ = atan2(z, w) - atan2(-x, y)
///
/// Based on Bernardes & Viollet (2022), simplified for unit quaternions and
/// proper ZYZ Euler angles (Chapter 3.3):
/// https://doi.org/10.1371/journal.pone.0276302
///
/// Reference implementation:
/// https://github.com/evbernardes/quaternion_to_euler
/// SymPy also implements this paper:
/// https://docs.sympy.org/latest/modules/algebras.html#sympy.algebras.Quaternion.to_euler
///
/// Pure-Z / XY-aligned quaternions (|x|,|y| < eps) take the β≈0 gimbal form so
/// tiny β drift cannot split the Z angle across φ/λ. Sanitize the atan2
/// operand at the pole so folding never evaluates atan2(0,0).
///
/// @note Floating-point errors may accumulate when merging many gates.
/// Normalizing either Z angle by 2*π flips the corresponding SU(2) quaternion
/// sign. The returned phase correction accounts for those flips.
///
/// @return `{theta, phi, lambda, phaseCorrection}` suitable for UOp
static std::array<Val, 4> anglesFromQuaternion(const Quat& q,
                                               const ScalarConsts& c) {
  OpBuilder& rewriter = q.w.getBuilder();
  const Location loc = q.w.getLoc();

  const auto land = [&](Value a, Value b) {
    return rewriter.createOrFold<arith::AndIOp>(loc, a, b);
  };
  const auto xyNearZero = land(q.x.abs().olt(c.eps), q.y.abs().olt(c.eps));

  /// The half-angle norms retain small rotations when cos(beta) rounds to one.
  /// Force beta=0 when (x,y)≈0, retaining the pure-Z shortcut.
  const auto sinHalfBetaSquared = q.x * q.x + q.y * q.y;
  const auto cosHalfBetaSquared = q.w * q.w + q.z * q.z;
  const auto sinHalfBeta = sinHalfBetaSquared.sqrt();
  const auto cosHalfBeta = cosHalfBetaSquared.sqrt();
  const auto betaRaw = sinHalfBeta.atan2(cosHalfBeta) * c.two;
  const auto beta = Val::select(xyNearZero, c.zero, betaRaw);

  /// safe1 = |beta| >= eps; safe2 = |beta - π| >= eps
  const auto safe1 = beta.abs().oge(c.eps);
  const auto betaMinusPi = beta - c.pi;
  const auto safe2 = betaMinusPi.abs().oge(c.eps);
  const auto notXy = rewriter.createOrFold<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, xyNearZero,
      mqt::constantFromScalar(rewriter, loc, false));
  const auto safe = land(land(safe1, safe2), notXy);
  const auto usePiGimbal = land(safe1, notXy);

  /// theta+ = atan2(z, w); theta- = atan2(-x, y)
  /// Sanitize y when (x,y)≈0 for the constant folder.
  const auto yForAtan2 = Val::select(xyNearZero, c.one, q.y);
  const auto thetaPlus = q.z.atan2(q.w);
  const auto minusX = -q.x;
  const auto thetaMinus = minusX.atan2(yForAtan2);
  const auto twoThetaPlus = thetaPlus * c.two;
  const auto twoThetaMinus = thetaMinus * c.two;

  /// Safe: alpha = theta+ + theta-, gamma = theta+ - theta-
  /// Gimbal: beta≈0 → alpha = 2*theta+; beta≈π → alpha = 2*theta-; gamma = 0
  const auto alphaSafe = thetaPlus + thetaMinus;
  const auto gammaSafe = thetaPlus - thetaMinus;
  const auto alphaUnsafe =
      Val::select(usePiGimbal, twoThetaMinus, twoThetaPlus);
  const auto alpha = Val::select(safe, alphaSafe, alphaUnsafe);
  const auto gamma = Val::select(safe, gammaSafe, c.zero);

  const auto phi = wrapToPi(alpha, c);
  const auto lambda = wrapToPi(gamma, c);
  /// Each removed 2*π Z rotation flips the SU(2) representative. Half of the
  /// total removed angle restores the original matrix as a global phase.
  const auto removedAlpha = alpha - phi;
  const auto removedGamma = gamma - lambda;
  const auto removedAngle = removedAlpha + removedGamma;
  return {beta, phi, lambda, removedAngle / c.two};
}

/// Conjugates q by Hadamard, mapping X to Z, Y to -Y, and Z to X.
static Quat hadamardConjugate(const Quat& q) {
  return {.w = q.w, .x = q.z, .y = -q.y, .z = q.x};
}

static bool isZeroAngle(Val angle) {
  const auto value = mqt::valueToConstantDouble(angle.getValue());
  return value && std::abs(*value) <= mqt::PARAMETER_COMPARISON_TOLERANCE;
}

static Val sumAngles(Val lhs, Val rhs) {
  if (isZeroAngle(lhs)) {
    return rhs;
  }
  if (isZeroAngle(rhs)) {
    return lhs;
  }
  return lhs + rhs;
}

/// Compose a maximal run without introducing quaternion expressions.
template <typename RotationOp>
static LogicalResult mergeParameterizedRotations(RotationOp op,
                                                 PatternRewriter& rewriter) {
  const auto matches = [&](RotationOp other) {
    if (!other || other->getBlock() != op->getBlock()) {
      return false;
    }
    if constexpr (std::is_same_v<RotationOp, ROp>) {
      return valuesMatchWithinTolerance(op.getPhi(), other.getPhi());
    }
    return true;
  };
  if (matches(op.getQubitIn().template getDefiningOp<RotationOp>())) {
    return failure();
  }
  SmallVector<RotationOp> chain;
  SmallVector<Value> angles;
  for (auto next = op; matches(next);
       next = dyn_cast<RotationOp>(*next.getQubitOut().user_begin())) {
    chain.push_back(next);
    angles.push_back(next.getTheta());
  }
  if (chain.size() < 2 || llvm::all_of(angles, [](Value angle) {
        return mqt::valueToConstantDouble(angle).has_value();
      })) {
    return failure();
  }
  auto last = chain.back();
  rewriter.setInsertionPoint(last);
  auto sum = decomposition::sumRotationAngles(rewriter, last.getLoc(), angles);
  if (mqt::valueToConstantDouble(sum) == 0.) {
    rewriter.replaceOp(last, op.getQubitIn());
  } else {
    /// Reuse the first gate so the common equatorial axis is unchanged.
    rewriter.moveOpBefore(op, last);
    rewriter.modifyOpInPlace(op, [&] { op.getThetaMutable().assign(sum); });
    rewriter.replaceOp(last, op.getQubitOut());
  }
  for (auto gate : llvm::reverse(llvm::drop_end(chain))) {
    if (gate != op || op->use_empty()) {
      rewriter.eraseOp(gate);
    }
  }
  return success();
}

/// Two positive equatorial half-turns are a Z rotation and phase pi.
static LogicalResult mergeEquatorialHalfTurns(ROp op,
                                              PatternRewriter& rewriter) {
  auto next = dyn_cast<ROp>(*op.getQubitOut().user_begin());
  if (!next || next->getBlock() != op->getBlock() ||
      op->getParentOfType<CtrlOp>() ||
      mqt::valueToConstantDouble(op.getTheta()) != std::numbers::pi ||
      mqt::valueToConstantDouble(next.getTheta()) != std::numbers::pi) {
    return failure();
  }
  rewriter.setInsertionPoint(next);
  const auto normalized = [&](Value angle) {
    return mqt::variantToValue(rewriter, next.getLoc(),
                               decomposition::normalizeRotationParameter(
                                   rewriter, next.getLoc(), angle));
  };
  auto difference = rewriter.createOrFold<arith::SubFOp>(
      next.getLoc(), normalized(next.getPhi()), normalized(op.getPhi()));
  auto angle = rewriter.createOrFold<arith::MulFOp>(
      next.getLoc(), difference,
      mqt::constantFromScalar(rewriter, next.getLoc(), 2.));
  decomposition::emitGPhaseIfNeeded(rewriter, next.getLoc(), std::numbers::pi);
  rewriter.replaceOpWithNewOp<RZOp>(next, op.getQubitIn(), angle);
  rewriter.eraseOp(op);
  return success();
}

/// Quaternion fusion produces basis Euler angles; synthesis owns emission.
static Value emitRuntimeEulerAngles(RewriterBase& rewriter, Location loc,
                                    Value qubit, RuntimeEulerAngles angles,
                                    const CompilerTarget::SynthesisBasis& basis,
                                    const ScalarConsts& consts) {
  auto [theta, phi, lambda, phase] = angles;
  if (basis.singleQubit == decomposition::SingleQubitBasis::ZXZ) {
    phi = phi + consts.pi / consts.two;
    lambda = lambda - consts.pi / consts.two;
  } else if (basis.singleQubit == decomposition::SingleQubitBasis::U) {
    phase = phase - sumAngles(phi, lambda) / consts.two;
  }
  return decomposition::emitParameterizedEulerAngles(
      rewriter, loc, qubit,
      {theta.getValue(), phi.getValue(), lambda.getValue(), phase.getValue()},
      basis);
}

static bool isMergeable(Operation* op) {
  return decomposition::canSynthesizeParameterizedUnitary1Q(op) ||
         isa<XOp, YOp, ZOp, HOp, SOp, SdgOp, TOp, TdgOp, SXOp, SXdgOp, IdOp>(
             op);
}

namespace {

/// Pattern that merges consecutive rotation gates using quaternion
/// multiplication.
struct MergeSingleQubitRotationGatesPattern final
    : OpInterfaceRewritePattern<UnitaryOpInterface> {
  explicit MergeSingleQubitRotationGatesPattern(
      MLIRContext* context,
      std::optional<CompilerTarget::SynthesisBasis> fusionBasis = std::nullopt,
      const CompilerTarget* target = nullptr)
      : OpInterfaceRewritePattern(context), fusionBasis(fusionBasis),
        target(target) {}

  std::optional<CompilerTarget::SynthesisBasis> fusionBasis;
  const CompilerTarget* target;

  /// Checks if this op is the start of a mergeable chain.
  ///
  /// A chain start is a mergeable op whose qubit input does NOT come from
  /// a chain-compatible predecessor. This ensures the greedy rewriter only
  /// triggers the rewrite at chain heads, building the maximal chain in one
  /// shot regardless of worklist order.
  static bool isChainStart(UnitaryOpInterface op) {
    if (!isMergeable(op.getOperation())) {
      return false;
    }
    Operation* defOp = op.getInputQubit(0).getDefiningOp();
    return defOp == nullptr || !isMergeable(defOp);
  }

  /// Collects a chain of consecutive mergeable gates.
  ///
  /// Walks forward via single-use SSA edges. Breaks when the next operation is
  /// not considered as mergeable.
  ///
  /// @param start The chain head (must satisfy isChainStart)
  /// @return The chain of operations in circuit order (first applied to last)
  static SmallVector<UnitaryOpInterface>
  collectChain(UnitaryOpInterface start) {
    SmallVector chain{start};
    for (auto curr = std::next(WireIterator(start.getOutputQubit(0)));
         curr != std::default_sentinel; ++curr) {
      if (!isMergeable(curr.operation())) {
        break;
      }
      chain.emplace_back(cast<UnitaryOpInterface>(*curr.operation()));
    }
    return chain;
  }

  static bool hasDynamicParameter(ArrayRef<UnitaryOpInterface> chain) {
    return llvm::any_of(chain, [](UnitaryOpInterface chainOp) {
      return llvm::any_of(chainOp.getParameters(), [](Value parameter) {
        return !mqt::valueToConstantDouble(parameter).has_value();
      });
    });
  }

  static size_t
  parameterizedSynthesisGateCount(decomposition::SingleQubitBasis basis) {
    switch (basis) {
    case decomposition::SingleQubitBasis::U:
      return 1;
    case decomposition::SingleQubitBasis::ZSXX:
      return 5;
    case decomposition::SingleQubitBasis::ZYZ:
    case decomposition::SingleQubitBasis::ZXZ:
    case decomposition::SingleQubitBasis::XZX:
    case decomposition::SingleQubitBasis::XYX:
      return 3;
    case decomposition::SingleQubitBasis::R:
      return 2;
    }
    llvm_unreachable("invalid single-qubit synthesis basis"); // LCOV_EXCL_LINE
  }

  static bool shouldComposeForFusion(ArrayRef<UnitaryOpInterface> chain,
                                     decomposition::SingleQubitBasis basis) {
    if (!hasDynamicParameter(chain)) {
      return false;
    }
    const bool hasNonBasisGate = llvm::any_of(chain, [basis](auto chainOp) {
      return !decomposition::isSingleQubitBasisGate(chainOp.getOperation(),
                                                    basis);
    });
    return hasNonBasisGate ||
           chain.size() > parameterizedSynthesisGateCount(basis);
  }

  /// Reuse numerical Euler synthesis for constant chains.
  static LogicalResult
  tryMergeStaticChain(MutableArrayRef<UnitaryOpInterface> chain,
                      RewriterBase& rewriter) {
    auto matrix = Matrix2x2::identity();
    for (auto gate : chain) {
      const auto factor = gate.getUnitaryMatrix<Matrix2x2>();
      if (!factor) {
        return failure();
      }
      matrix.premultiplyBy(*factor);
    }
    const auto result = decomposition::synthesizeUnitary1QEuler(
        rewriter, chain.front()->getLoc(), chain.front().getInputQubit(0),
        matrix, chain.size(), true,
        {.singleQubit = decomposition::SingleQubitBasis::U});
    decomposition::emitGPhaseIfNeeded(rewriter, chain.front()->getLoc(),
                                      result->globalPhase);
    for (auto gate : llvm::drop_begin(chain)) {
      rewriter.replaceOp(gate, gate.getInputQubit(0));
    }
    rewriter.replaceOp(chain.front(), result->qubit);
    return success();
  }

  /// Reuse Euler angles when the chain and output share their outer axis.
  /// Either outer rotation may be absent. H/RZ pairs use H RZ = RX H to
  /// align the rotation with the output basis. Normalize gate operands before
  /// adding Euler offsets or computing the U phase correction.
  static LogicalResult
  tryMergeDirectChain(MutableArrayRef<UnitaryOpInterface> chain,
                      RewriterBase& rewriter,
                      const CompilerTarget::SynthesisBasis& synthesisBasis) {
    const auto basis = synthesisBasis.singleQubit;
    const bool outerX = basis == decomposition::SingleQubitBasis::XZX ||
                        basis == decomposition::SingleQubitBasis::XYX;
    const auto isOuter = [outerX](UnitaryOpInterface op) {
      return outerX ? isa<RXOp>(op.getOperation())
                    : isa<RZOp>(op.getOperation());
    };

    const bool hadamardPair =
        chain.size() == 2 &&
        ((isa<HOp>(chain.front()) && isa<RZOp>(chain.back())) ||
         (isa<RZOp>(chain.front()) && isa<HOp>(chain.back())));
    const size_t middle = chain.size() > 1 && isOuter(chain.front()) ? 1 : 0;
    if (!hadamardPair &&
        (chain.size() <= middle || chain.size() > middle + 2 ||
         !isa<RXOp, RYOp, RZOp>(chain[middle].getOperation()) ||
         (chain.size() > 1 && isOuter(chain[middle])) ||
         (chain.size() == middle + 2 && !isOuter(chain.back())))) {
      return failure();
    }

    /// Check the complete run before creating or replacing any operations.
    const Location loc = chain.front()->getLoc();
    const auto consts = makeConsts(rewriter, loc);
    const auto angle = [&](UnitaryOpInterface op) {
      return gateParam(op, 0, rewriter, loc);
    };
    RuntimeEulerAngles angles{
        .theta = consts.zero,
        .phi = consts.zero,
        .lambda = consts.zero,
        .phase = consts.zero,
    };
    if (hadamardPair) {
      const auto fixed = decomposition::anglesFromUnitary(
          HOp::getUnitaryMatrix(),
          outerX ? basis : decomposition::SingleQubitBasis::ZYZ);
      angles = {
          .theta = Val::constant(rewriter, loc, fixed.theta),
          .phi = Val::constant(rewriter, loc, fixed.phi),
          .lambda = Val::constant(rewriter, loc, fixed.lambda),
          .phase = Val::constant(rewriter, loc, fixed.phase),
      };
      const bool rotationFirst = isa<RZOp>(chain.front());
      auto& outer = rotationFirst != outerX ? angles.lambda : angles.phi;
      outer =
          sumAngles(outer, angle(rotationFirst ? chain.front() : chain.back()));
    } else {
      angles.theta = Val(rewriter, loc, chain[middle].getParameter(0));
      if (basis == decomposition::SingleQubitBasis::ZSXX ||
          isOuter(chain[middle]) ||
          mqt::valueToConstantDouble(angles.theta.getValue())) {
        angles.theta = normalizeGateAngle(angles.theta);
      }
      if (!outerX) {
        if (isa<RXOp>(chain[middle])) {
          angles.phi = -consts.pi / consts.two;
          angles.lambda = consts.pi / consts.two;
        } else if (isa<RZOp>(chain[middle])) {
          angles.lambda = angles.theta;
          angles.theta = consts.zero;
        }
      } else if (isOuter(chain[middle])) {
        angles.lambda = angles.theta;
        angles.theta = consts.zero;
      } else if (const bool middleZ = isa<RZOp>(chain[middle].getOperation());
                 middleZ != (basis == decomposition::SingleQubitBasis::XZX)) {
        /// RX conjugation exchanges Y and Z, with opposite quarter-turns.
        const auto halfPi = consts.pi / consts.two;
        angles.phi = middleZ ? halfPi : -halfPi;
        angles.lambda = -angles.phi;
      }
      if (middle == 1) {
        angles.lambda = sumAngles(angles.lambda, angle(chain.front()));
      }
      if (chain.size() == middle + 2) {
        angles.phi = sumAngles(angles.phi, angle(chain.back()));
      }
    }

    for (auto op : llvm::drop_begin(chain)) {
      rewriter.replaceOp(op, op.getInputQubit(0));
    }
    Value qubit =
        emitRuntimeEulerAngles(rewriter, loc, chain.front().getInputQubit(0),
                               angles, synthesisBasis, consts);
    rewriter.replaceOp(chain.front(), qubit);
    return success();
  }

  /// Merges a dynamic or mixed-angle chain through scalar SSA operations.
  //
  /// Fusion mode emits the requested basis directly. Regular merge mode emits
  /// U and applies its intrinsic global-phase correction:
  ///   correction = totalInputPhase - (phi + lambda) / 2
  /// Pass-level global-phase normalization combines and normalizes the result.
  static LogicalResult
  mergeDynamicChain(MutableArrayRef<UnitaryOpInterface> chain,
                    RewriterBase& rewriter,
                    const CompilerTarget::SynthesisBasis& synthesisBasis) {
    const auto basis = synthesisBasis.singleQubit;
    if (succeeded(tryMergeDirectChain(chain, rewriter, synthesisBasis))) {
      return success();
    }
    const Location loc = chain.front()->getLoc();
    const auto consts = makeConsts(rewriter, loc);

    std::optional<Quat> qAccum;
    Val phaseAccum = consts.zero;
    for (UnitaryOpInterface chainOp : chain) {
      const auto [qi, phase] = quaternionFromGate(chainOp, consts, rewriter);
      qAccum = qAccum ? hamiltonProduct(qi, *qAccum) : qi;
      phaseAccum = normalizeGateAngle(phaseAccum + phase);
    }

    for (auto chainOp : llvm::drop_begin(chain)) {
      rewriter.replaceOp(chainOp, chainOp.getInputQubit(0));
    }

    const bool transformed = basis == decomposition::SingleQubitBasis::XZX ||
                             basis == decomposition::SingleQubitBasis::XYX;
    auto [theta, phi, lambda, eulerPhase] =
        transformed ? anglesFromQuaternion(hadamardConjugate(*qAccum), consts)
                    : anglesFromQuaternion(*qAccum, consts);
    if (basis == decomposition::SingleQubitBasis::XZX) {
      phi = phi + consts.pi / consts.two;
      lambda = lambda - consts.pi / consts.two;
    } else if (basis == decomposition::SingleQubitBasis::XYX) {
      phi = phi + consts.pi;
      lambda = lambda + consts.pi;
      eulerPhase = eulerPhase + consts.pi;
    }
    const RuntimeEulerAngles angles{
        .theta = theta,
        .phi = phi,
        .lambda = lambda,
        .phase = phaseAccum + eulerPhase,
    };
    Value qubit =
        emitRuntimeEulerAngles(rewriter, loc, chain.front().getInputQubit(0),
                               angles, synthesisBasis, consts);
    rewriter.replaceOp(chain.front(), qubit);
    return success();
  }

  /// Matches the full chain, folds its quaternions with Hamilton products, and
  /// emits one U operation or the requested fusion basis. Constant chains use
  /// numerical Euler synthesis; runtime chains use quaternion composition.
  LogicalResult matchAndRewrite(UnitaryOpInterface op,
                                PatternRewriter& rewriter) const override {
    auto control = op->getParentOfType<CtrlOp>();
    if (((target != nullptr) && target->synthesisBasis()->singleQubit !=
                                    decomposition::SingleQubitBasis::U) &&
        control) {
      return failure();
    }
    if (!isChainStart(op)) {
      return failure();
    }

    auto chain = collectChain(op);
    if (target != nullptr && chain.size() == 1) {
      return failure();
    }
    /// Emit helper operations at the chain tail next to the merged output.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(chain.back().getOperation());

    if (fusionBasis) {
      if (!shouldComposeForFusion(chain, fusionBasis->singleQubit)) {
        return failure();
      }
      /// A multi-gate control body is not itself a native operation.
      if (target != nullptr && !control &&
          llvm::all_of(chain, [&](auto member) {
            return target->supports(member.getOperation());
          })) {
        return failure();
      }
      if (target != nullptr && !control) {
        return tryMergeDirectChain(chain, rewriter, *fusionBasis);
      }
      return mergeDynamicChain(chain, rewriter, *fusionBasis);
    }
    if (chain.size() < 2) {
      return failure();
    }

    if (succeeded(tryMergeStaticChain(chain, rewriter))) {
      return success();
    }
    return mergeDynamicChain(
        chain, rewriter, {.singleQubit = decomposition::SingleQubitBasis::U});
  }
};

/// Pass that merges consecutive rotation gates using quaternion
/// multiplication.
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

    if (failed(applyPatternsGreedily(op, std::move(patterns))) ||
        failed(mlir::mqt::normalizeGlobalPhases(op))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::qco

namespace mlir::qco::decomposition {

void populateRotationCompositionPatterns(RewritePatternSet& patterns,
                                         const CompilerTarget* target) {
  const auto unrestricted = [&](StringRef name, size_t parameters) {
    return target == nullptr || llvm::all_of(target->siteIds(), [&](auto site) {
             return target->supportsOperation(name, 1, parameters,
                                              ArrayRef(&site, 1));
           });
  };
  if (unrestricted("rx", 1)) {
    patterns.add(mergeParameterizedRotations<RXOp>);
  }
  if (unrestricted("ry", 1)) {
    patterns.add(mergeParameterizedRotations<RYOp>);
  }
  if (unrestricted("rz", 1)) {
    patterns.add(mergeParameterizedRotations<RZOp>);
    patterns.add(mergeEquatorialHalfTurns);
  }
  if (unrestricted("r", 2)) {
    patterns.add(mergeParameterizedRotations<ROp>);
  }
}

void populateParameterizedSingleQubitRunCompositionPatterns(
    RewritePatternSet& patterns, const CompilerTarget::SynthesisBasis& basis,
    const CompilerTarget* target) {
  populateRotationCompositionPatterns(patterns, target);
  RZOp::getCanonicalizationPatterns(patterns, patterns.getContext());
  if (basis.singleQubit == SingleQubitBasis::ZSXX && target != nullptr) {
    return;
  }
  RXOp::getCanonicalizationPatterns(patterns, patterns.getContext());
  RYOp::getCanonicalizationPatterns(patterns, patterns.getContext());
  POp::getCanonicalizationPatterns(patterns, patterns.getContext());
  patterns.add<MergeSingleQubitRotationGatesPattern>(patterns.getContext(),
                                                     basis, target);
}

} // namespace mlir::qco::decomposition
