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
#include "mlir/Dialect/QCO/Utils/WireIterator.h"
#include "mlir/Dialect/Utils/Transforms/GlobalPhaseNormalization.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
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

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <numbers>
#include <optional>
#include <type_traits>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_MERGESINGLEQUBITROTATIONGATES
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

namespace {

/// Dual-backend scalar for the merge algorithm: `double` uses STL; `Value`
/// emits `arith` / `math` ops. One implementation serves both static-angle and
/// dynamic-angle chains.
template <typename T> struct Val {
  static_assert(std::is_same_v<T, double> || std::is_same_v<T, Value>,
                "Val supports double and Value only");

  T v;
  PatternRewriter* rewriter;
  Location loc;

  using Pred = std::conditional_t<std::is_same_v<T, double>, bool, Value>;

  static Val constant(PatternRewriter& rewriter, Location loc, double x) {
    if constexpr (std::is_same_v<T, double>) {
      return {x, &rewriter, loc};
    } else {
      return {utils::constantFromScalar(rewriter, loc, x), &rewriter, loc};
    }
  }

  [[nodiscard]] Val operator+(Val o) const {
    if constexpr (std::is_same_v<T, double>) {
      return {v + o.v, rewriter, loc};
    } else {
      return {arith::AddFOp::create(*rewriter, loc, v, o.v).getResult(),
              rewriter, loc};
    }
  }
  [[nodiscard]] Val operator-(Val o) const {
    if constexpr (std::is_same_v<T, double>) {
      return {v - o.v, rewriter, loc};
    } else {
      return {arith::SubFOp::create(*rewriter, loc, v, o.v).getResult(),
              rewriter, loc};
    }
  }
  [[nodiscard]] Val operator*(Val o) const {
    if constexpr (std::is_same_v<T, double>) {
      return {v * o.v, rewriter, loc};
    } else {
      return {arith::MulFOp::create(*rewriter, loc, v, o.v).getResult(),
              rewriter, loc};
    }
  }
  [[nodiscard]] Val operator/(Val o) const {
    if constexpr (std::is_same_v<T, double>) {
      return {v / o.v, rewriter, loc};
    } else {
      return {arith::DivFOp::create(*rewriter, loc, v, o.v).getResult(),
              rewriter, loc};
    }
  }
  [[nodiscard]] Val operator-() const {
    if constexpr (std::is_same_v<T, double>) {
      return {-v, rewriter, loc};
    } else {
      return {arith::NegFOp::create(*rewriter, loc, v).getResult(), rewriter,
              loc};
    }
  }

  [[nodiscard]] Val sin() const {
    if constexpr (std::is_same_v<T, double>) {
      return {std::sin(v), rewriter, loc};
    } else {
      return {math::SinOp::create(*rewriter, loc, v).getResult(), rewriter,
              loc};
    }
  }
  [[nodiscard]] Val cos() const {
    if constexpr (std::is_same_v<T, double>) {
      return {std::cos(v), rewriter, loc};
    } else {
      return {math::CosOp::create(*rewriter, loc, v).getResult(), rewriter,
              loc};
    }
  }
  [[nodiscard]] Val abs() const {
    if constexpr (std::is_same_v<T, double>) {
      return {std::abs(v), rewriter, loc};
    } else {
      return {math::AbsFOp::create(*rewriter, loc, v).getResult(), rewriter,
              loc};
    }
  }
  [[nodiscard]] Val floor() const {
    if constexpr (std::is_same_v<T, double>) {
      return {std::floor(v), rewriter, loc};
    } else {
      return {math::FloorOp::create(*rewriter, loc, v).getResult(), rewriter,
              loc};
    }
  }
  [[nodiscard]] Val acos() const {
    if constexpr (std::is_same_v<T, double>) {
      return {std::acos(v), rewriter, loc};
    } else {
      return {math::AcosOp::create(*rewriter, loc, v).getResult(), rewriter,
              loc};
    }
  }
  [[nodiscard]] Val atan2(Val x) const {
    // `*this` is y, `x` is x — same order as std::atan2 / math.atan2.
    if constexpr (std::is_same_v<T, double>) {
      return {std::atan2(v, x.v), rewriter, loc};
    } else {
      return {math::Atan2Op::create(*rewriter, loc, v, x.v).getResult(),
              rewriter, loc};
    }
  }
  [[nodiscard]] Val maximum(Val o) const {
    if constexpr (std::is_same_v<T, double>) {
      return {std::max(v, o.v), rewriter, loc};
    } else {
      return {arith::MaximumFOp::create(*rewriter, loc, v, o.v).getResult(),
              rewriter, loc};
    }
  }
  [[nodiscard]] Val minimum(Val o) const {
    if constexpr (std::is_same_v<T, double>) {
      return {std::min(v, o.v), rewriter, loc};
    } else {
      return {arith::MinimumFOp::create(*rewriter, loc, v, o.v).getResult(),
              rewriter, loc};
    }
  }

  [[nodiscard]] Pred oge(Val o) const {
    if constexpr (std::is_same_v<T, double>) {
      return v >= o.v;
    } else {
      return arith::CmpFOp::create(*rewriter, loc, arith::CmpFPredicate::OGE, v,
                                   o.v)
          .getResult();
    }
  }
  [[nodiscard]] Pred olt(Val o) const {
    if constexpr (std::is_same_v<T, double>) {
      return v < o.v;
    } else {
      return arith::CmpFOp::create(*rewriter, loc, arith::CmpFPredicate::OLT, v,
                                   o.v)
          .getResult();
    }
  }

  static Pred land(Pred a, Pred b, PatternRewriter& rewriter, Location loc) {
    if constexpr (std::is_same_v<T, double>) {
      return a && b;
    } else {
      return arith::AndIOp::create(rewriter, loc, a, b).getResult();
    }
  }
  static Pred lnot(Pred a, PatternRewriter& rewriter, Location loc) {
    if constexpr (std::is_same_v<T, double>) {
      return !a;
    } else {
      auto falseV =
          arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(false));
      return arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq, a,
                                   falseV)
          .getResult();
    }
  }
  static Val select(Pred c, Val t, Val f) {
    if constexpr (std::is_same_v<T, double>) {
      return c ? t : f;
    } else {
      return {
          arith::SelectOp::create(*t.rewriter, t.loc, c, t.v, f.v).getResult(),
          t.rewriter, t.loc};
    }
  }
};

enum class RotationAxis : std::uint8_t { X, Y, Z };

/// Unit quaternion w + x i + y j + z k over the dual-backend scalar type.
template <typename T> struct Quat {
  Val<T> w;
  Val<T> x;
  Val<T> y;
  Val<T> z;
};

/// Shared numeric constants used by quaternion construction and Euler extract.
template <typename T> struct ScalarConsts {
  Val<T> negOne;
  Val<T> zero;
  Val<T> one;
  Val<T> two;
  Val<T> eps;
  Val<T> pi;
};

/**
 * @brief Creates shared f64 constants for the merge algorithm.
 *
 * `eps` (1e-12) is the gimbal-lock tolerance from the reference implementation:
 * https://github.com/evbernardes/quaternion_to_euler/blob/main/euler_from_quat.py
 */
template <typename T>
ScalarConsts<T> makeConsts(PatternRewriter& rewriter, Location loc) {
  auto c = [&](double x) { return Val<T>::constant(rewriter, loc, x); };
  return {.negOne = c(-1.0),
          .zero = c(0.0),
          .one = c(1.0),
          .two = c(2.0),
          .eps = c(1e-12),
          .pi = c(std::numbers::pi)};
}

/**
 * @brief Normalizes an angle to the range [-PI, PI].
 *
 * Uses floor-based modular arithmetic:
 *   normalize(a) = a - floor((a + π) / 2π) * 2π
 */
template <typename T> Val<T> wrapToPi(Val<T> angle, const ScalarConsts<T>& c) {
  const auto twoPi = c.two * c.pi;
  const auto floored = ((angle + c.pi) / twoPi).floor();
  return angle - (floored * twoPi);
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
 */
template <typename T>
Quat<T> hamiltonProduct(const Quat<T>& q1, const Quat<T>& q2) {
  return {
      .w = (q1.w * q2.w) - (q1.x * q2.x) - (q1.y * q2.y) - (q1.z * q2.z),
      .x = (q1.w * q2.x) + (q1.x * q2.w) + (q1.y * q2.z) - (q1.z * q2.y),
      .y = (q1.w * q2.y) - (q1.x * q2.z) + (q1.y * q2.w) + (q1.z * q2.x),
      .z = (q1.w * q2.z) + (q1.x * q2.y) - (q1.y * q2.x) + (q1.z * q2.w),
  };
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
 */
template <typename T>
Quat<T> axisQuaternion(Val<T> angle, RotationAxis axis,
                       const ScalarConsts<T>& c) {
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
 */
template <typename T>
Quat<T> quaternionFromZYZ(Val<T> theta, Val<T> phi, Val<T> lambda,
                          const ScalarConsts<T>& c) {
  const auto qTheta = axisQuaternion(theta, RotationAxis::Y, c);
  const auto qPhi = axisQuaternion(phi, RotationAxis::Z, c);
  const auto qLambda = axisQuaternion(lambda, RotationAxis::Z, c);
  return hamiltonProduct(hamiltonProduct(qPhi, qTheta), qLambda);
}

/**
 * @brief Returns the rotation axis for an RXOp, RYOp, RZOp, or POp.
 */
static std::optional<RotationAxis> getRotationAxis(Operation* op) {
  return TypeSwitch<Operation*, std::optional<RotationAxis>>(op)
      .Case<RXOp>([](auto) { return RotationAxis::X; })
      .Case<RYOp>([](auto) { return RotationAxis::Y; })
      .Case<RZOp, POp>([](auto) { return RotationAxis::Z; })
      .Default([](auto) { return std::nullopt; });
}

template <typename T>
std::optional<Val<T>> gateParam(UnitaryOpInterface op, unsigned i,
                                PatternRewriter& rewriter, Location loc) {
  Value p = op.getParameter(i);
  if constexpr (std::is_same_v<T, double>) {
    const auto folded = utils::valueToConstantDouble(p);
    if (!folded) {
      return std::nullopt;
    }
    return Val<T>::constant(rewriter, loc, *folded);
  } else {
    return Val<T>{p, &rewriter, loc};
  }
}

/**
 * @brief Converts a rotation gate to quaternion representation.
 *
 * - RX, RY, RZ, P: single-axis half-angle formulas.
 * - R(theta, phi): Q(cos(θ/2), sin(θ/2)cos(φ), sin(θ/2)sin(φ), 0).
 * - U2(phi, lambda) = U(π/2, phi, lambda).
 * - U(theta, phi, lambda): ZYZ via quaternionFromZYZ.
 *
 * @note Global phase is discarded; see quaternionFromZYZ for details.
 * @return nullopt if a required parameter cannot be represented as `T`
 *         (static path: unfoldable SSA value).
 */
template <typename T>
std::optional<Quat<T>> quaternionFromRotation(UnitaryOpInterface op,
                                              const ScalarConsts<T>& c,
                                              PatternRewriter& rewriter) {
  const Location loc = op->getLoc();
  auto param = [&](unsigned i) { return gateParam<T>(op, i, rewriter, loc); };

  // Single-axis rotations (RX, RY, RZ, P) share the same conversion pattern
  if (const auto axis = getRotationAxis(op.getOperation())) {
    const auto angle = param(0);
    if (!angle) {
      return std::nullopt;
    }
    return axisQuaternion(*angle, *axis, c);
  }

  // Multi-parameter gates each need their own conversion
  return TypeSwitch<Operation*, std::optional<Quat<T>>>(op.getOperation())
      .template Case<ROp>([&](ROp) -> std::optional<Quat<T>> {
        const auto theta = param(0);
        const auto phi = param(1);
        if (!theta || !phi) {
          return std::nullopt;
        }
        const auto halfTheta = *theta / c.two;
        const auto sinHalf = halfTheta.sin();
        return Quat<T>{.w = halfTheta.cos(),
                       .x = sinHalf * phi->cos(),
                       .y = sinHalf * phi->sin(),
                       .z = c.zero};
      })
      .template Case<U2Op>([&](U2Op) -> std::optional<Quat<T>> {
        const auto phi = param(0);
        const auto lambda = param(1);
        if (!phi || !lambda) {
          return std::nullopt;
        }
        return quaternionFromZYZ(c.pi / c.two, *phi, *lambda, c);
      })
      .template Case<UOp>([&](UOp) -> std::optional<Quat<T>> {
        const auto theta = param(0);
        const auto phi = param(1);
        const auto lambda = param(2);
        if (!theta || !phi || !lambda) {
          return std::nullopt;
        }
        return quaternionFromZYZ(*theta, *phi, *lambda, c);
      })
      .Default([](auto) -> std::optional<Quat<T>> { return std::nullopt; });
}

/**
 * @brief Returns the global phase contribution of a rotation gate.
 *
 * Rotation gates can be factored as U = e^{i * phase} * SU(2), where SU(2)
 * is the quaternion-representable part and phase is the global phase:
 *
 * - RX, RY, RZ, R         -> none (already SU(2), no global phase)
 * - P(theta)              -> theta / 2 (P = e^{i * theta / 2} * RZ(theta))
 * - U(theta, phi, lambda) -> (phi + lambda) / 2
 * - U2(phi, lambda)       -> (phi + lambda) / 2
 *
 * @return nullopt for SU(2) gates, or when a required parameter does not fold
 *         on the static (`double`) path.
 */
template <typename T>
std::optional<Val<T>> globalPhaseOf(UnitaryOpInterface op,
                                    const ScalarConsts<T>& c,
                                    PatternRewriter& rewriter) {
  const Location loc = op->getLoc();
  auto param = [&](unsigned i) { return gateParam<T>(op, i, rewriter, loc); };

  return TypeSwitch<Operation*, std::optional<Val<T>>>(op.getOperation())
      .template Case<RXOp, RYOp, RZOp, ROp>(
          [](auto) -> std::optional<Val<T>> { return std::nullopt; })
      .template Case<POp>([&](auto) -> std::optional<Val<T>> {
        const auto theta = param(0);
        if (!theta) {
          return std::nullopt;
        }
        return *theta / c.two;
      })
      .template Case<UOp, U2Op>([&](auto) -> std::optional<Val<T>> {
        // phi is at different indexes for UOp and U2Op
        const auto phiIdx = isa<UOp>(op.getOperation()) ? 1U : 0U;
        const auto phi = param(phiIdx);
        const auto lambda = param(phiIdx + 1);
        if (!phi || !lambda) {
          return std::nullopt;
        }
        return (*phi + *lambda) / c.two;
      })
      .Default([](auto) -> std::optional<Val<T>> { return std::nullopt; });
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
 * Pure-Z / XY-aligned quaternions (|x|,|y| < eps) take the beta≈0 gimbal form
 * so tiny beta drift cannot split the Z angle across phi/lambda. The `Value`
 * backend also sanitizes the atan2 y-operand when (x,y)≈0 so MLIR's constant
 * folder never sees atan2(0,0) → NaN on a dead select input.
 *
 * @note Floating-point errors may accumulate when merging many gates.
 * @return {theta, phi, lambda} = {beta, alpha, gamma} suitable for UOp
 */
template <typename T>
std::array<Val<T>, 3> anglesFromQuaternion(const Quat<T>& q,
                                           const ScalarConsts<T>& c) {
  PatternRewriter& rewriter = *q.w.rewriter;
  const Location loc = q.w.loc;

  const auto xyNearZero =
      Val<T>::land(q.x.abs().olt(c.eps), q.y.abs().olt(c.eps), rewriter, loc);

  // Host path can take the pure-Z shortcut without building the full tree.
  if constexpr (std::is_same_v<T, double>) {
    if (xyNearZero) {
      return {c.zero, wrapToPi(q.z.atan2(q.w) * c.two, c), c.zero};
    }
  }

  // beta = acos(clamp(2 * (w^2 + z^2) - 1, -1, 1))
  const auto cosBeta = ((c.two * ((q.w * q.w) + (q.z * q.z))) - c.one)
                           .maximum(c.negOne)
                           .minimum(c.one);
  const auto beta = cosBeta.acos();

  // safe1 = |beta| >= eps; safe2 = |beta - π| >= eps
  const auto safe1 = beta.abs().oge(c.eps);
  const auto safe2 = (beta - c.pi).abs().oge(c.eps);
  const auto notXy = Val<T>::lnot(xyNearZero, rewriter, loc);
  const auto safe = Val<T>::land(Val<T>::land(safe1, safe2, rewriter, loc),
                                 notXy, rewriter, loc);
  const auto usePiGimbal = Val<T>::land(safe1, notXy, rewriter, loc);

  // theta+ = atan2(z, w); theta- = atan2(-x, y)
  // Sanitize y when (x,y)≈0 for the Value backend's constant folder.
  const auto yForAtan2 = Val<T>::select(xyNearZero, c.one, q.y);
  const auto thetaPlus = q.z.atan2(q.w);
  const auto thetaMinus = (-q.x).atan2(yForAtan2);
  const auto twoThetaPlus = thetaPlus * c.two;
  const auto twoThetaMinus = thetaMinus * c.two;

  // Safe: alpha = theta+ + theta-, gamma = theta+ - theta-
  // Gimbal: beta≈0 → alpha = 2*theta+; beta≈π → alpha = 2*theta-; gamma = 0
  const auto alphaSafe = thetaPlus + thetaMinus;
  const auto gammaSafe = thetaPlus - thetaMinus;
  const auto alphaUnsafe =
      Val<T>::select(usePiGimbal, twoThetaMinus, twoThetaPlus);
  const auto alpha = Val<T>::select(safe, alphaSafe, alphaUnsafe);
  const auto gamma = Val<T>::select(safe, gammaSafe, c.zero);

  return {beta, wrapToPi(alpha, c), wrapToPi(gamma, c)};
}

static bool isMergeable(Operation* op) {
  return isa<RXOp, RYOp, RZOp, POp, ROp, U2Op, UOp>(op);
}

static bool areQuaternionMergeable(Operation* a, Operation* b) {
  return isMergeable(a) && isMergeable(b);
}

/**
 * @brief Pattern that merges consecutive rotation gates using quaternion
 * multiplication.
 */
struct MergeSingleQubitRotationGatesPattern final
    : OpInterfaceRewritePattern<UnitaryOpInterface> {
  explicit MergeSingleQubitRotationGatesPattern(MLIRContext* context)
      : OpInterfaceRewritePattern(context) {}

  /**
   * @brief Checks if this op is the start of a mergeable chain.
   *
   * A chain start is a mergeable op whose qubit input does NOT come from
   * a chain-compatible predecessor. This ensures the greedy rewriter only
   * triggers the rewrite at chain heads, building the maximal chain in one
   * shot regardless of worklist order.
   */
  static bool isChainStart(UnitaryOpInterface op) {
    if (!isMergeable(op.getOperation())) {
      return false;
    }
    Operation* defOp = op.getInputQubit(0).getDefiningOp();
    return defOp == nullptr || !areQuaternionMergeable(defOp, op);
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
    SmallVector chain{start};
    WireIterator prev(start.getOutputQubit(0));
    for (auto curr = std::next(prev); curr != std::default_sentinel; ++curr) {
      if (!areQuaternionMergeable(prev.operation(), curr.operation())) {
        break;
      }
      chain.emplace_back(cast<UnitaryOpInterface>(*curr.operation()));
      prev = curr;
    }
    return chain;
  }

  /**
   * @brief Merge a chain whose angles are all compile-time constants.
   *
   * Runs the shared algorithm on `Val<double>` (STL math) and emits constant
   * `U` / `gphase` values. Returns failure if any parameter is dynamic.
   */
  static LogicalResult
  tryMergeStaticChain(MutableArrayRef<UnitaryOpInterface> chain,
                      PatternRewriter& rewriter) {
    const Location loc = chain.front()->getLoc();
    const auto consts = makeConsts<double>(rewriter, loc);

    std::optional<Quat<double>> qAccum;
    Val<double> phaseAccum = consts.zero;
    for (UnitaryOpInterface chainOp : chain) {
      auto qi = quaternionFromRotation<double>(chainOp, consts, rewriter);
      if (!qi) {
        return failure();
      }
      if (isa<POp, UOp, U2Op>(chainOp.getOperation())) {
        const auto phase = globalPhaseOf<double>(chainOp, consts, rewriter);
        if (!phase) {
          return failure();
        }
        phaseAccum = phaseAccum + *phase;
      }
      qAccum = qAccum ? hamiltonProduct(*qi, *qAccum) : *qi;
    }

    const auto [theta, phi, lambda] = anglesFromQuaternion(*qAccum, consts);
    const auto correction = phaseAccum - ((phi + lambda) / consts.two);
    const double correctionHost = utils::normalizeAngle(correction.v);

    for (auto chainOp : llvm::drop_begin(chain)) {
      rewriter.replaceOp(chainOp, chainOp.getInputQubit(0));
    }
    if (std::abs(correctionHost) > utils::TOLERANCE) {
      GPhaseOp::create(
          rewriter, loc,
          utils::constantFromScalar(rewriter, loc, correctionHost));
    }
    rewriter.replaceOpWithNewOp<UOp>(
        chain.front(), chain.front().getInputQubit(0),
        utils::constantFromScalar(rewriter, loc, theta.v),
        utils::constantFromScalar(rewriter, loc, phi.v),
        utils::constantFromScalar(rewriter, loc, lambda.v));
    return success();
  }

  /**
   * @brief Merge a dynamic or mixed-angle chain via `Val<Value>` SSA.
   *
   * Same quaternion / Euler algorithm as the static path. Emits global phase
   * correction:
   *   outPhase = (phi + lambda) / 2
   *   correction = totalInputPhase - outPhase
   * Foldable corrections are normalized into the practical gphase range.
   */
  static void mergeDynamicChain(MutableArrayRef<UnitaryOpInterface> chain,
                                PatternRewriter& rewriter) {
    const Location loc = chain.front()->getLoc();
    const auto consts = makeConsts<Value>(rewriter, loc);

    auto qAccum =
        *quaternionFromRotation<Value>(chain.front(), consts, rewriter);
    std::optional<Val<Value>> phaseAccum =
        globalPhaseOf<Value>(chain.front(), consts, rewriter);

    for (auto chainOp : llvm::drop_begin(chain)) {
      auto qi = *quaternionFromRotation<Value>(chainOp, consts, rewriter);
      qAccum = hamiltonProduct(qi, qAccum);
      if (auto phase = globalPhaseOf<Value>(chainOp, consts, rewriter)) {
        phaseAccum = phaseAccum ? (*phaseAccum + *phase) : phase;
      }
      rewriter.replaceOp(chainOp, chainOp.getInputQubit(0));
    }

    const auto [theta, phi, lambda] = anglesFromQuaternion(qAccum, consts);
    const auto outPhase = (phi + lambda) / consts.two;
    const auto inputPhase = phaseAccum.value_or(consts.zero);
    Val<Value> phaseCorrection = inputPhase - outPhase;
    if (const auto constant = utils::valueToConstantDouble(phaseCorrection.v)) {
      phaseCorrection =
          Val<Value>::constant(rewriter, loc, utils::normalizeAngle(*constant));
    }
    GPhaseOp::create(rewriter, loc, phaseCorrection.v);
    rewriter.replaceOpWithNewOp<UOp>(chain.front(),
                                     chain.front().getInputQubit(0), theta.v,
                                     phi.v, lambda.v);
  }

  /**
   * @brief Matches and merges a chain of consecutive rotation gates.
   *
   * Detects the full chain of mergeable operations, folds their quaternions
   * via Hamilton product, and emits a single UOp. Fully static chains use
   * host STL math; otherwise the SSA `arith`/`math` path is used.
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

    if (succeeded(tryMergeStaticChain(chain, rewriter))) {
      return success();
    }
    mergeDynamicChain(chain, rewriter);
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

    if (failed(applyPatternsGreedily(op, std::move(patterns))) ||
        failed(mlir::mqt::normalizeGlobalPhases(op))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::qco
