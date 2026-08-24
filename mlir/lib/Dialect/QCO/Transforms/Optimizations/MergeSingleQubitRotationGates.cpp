/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQT/Transforms/GlobalPhaseNormalization.h"
#include "mlir/Dialect/MQT/Utils/ConstantFolding.h"
#include "mlir/Dialect/MQT/Utils/Parameters.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/WireIterator.h"

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
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
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

  T v{};
  PatternRewriter* rewriter = nullptr;
  Location loc;

  using Pred = std::conditional_t<std::is_same_v<T, double>, bool, Value>;

  static Val constant(PatternRewriter& rewriter, Location loc, double x) {
    if constexpr (std::is_same_v<T, double>) {
      return {x, &rewriter, loc};
    } else {
      return {mqt::constantFromScalar(rewriter, loc, x), &rewriter, loc};
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

enum class RotationAxis : uint8_t { X, Y, Z };

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

} // namespace

/**
 * @brief Creates shared f64 constants for the merge algorithm.
 *
 * `eps` (1e-12) is the gimbal-lock tolerance from the reference implementation:
 * https://github.com/evbernardes/quaternion_to_euler/blob/main/euler_from_quat.py
 */
template <typename T>
static ScalarConsts<T> makeConsts(PatternRewriter& rewriter, Location loc) {
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
template <typename T>
static Val<T> wrapToPi(Val<T> angle, const ScalarConsts<T>& c) {
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
static Quat<T> hamiltonProduct(const Quat<T>& q1, const Quat<T>& q2) {
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
static Quat<T> axisQuaternion(Val<T> angle, RotationAxis axis,
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
static Quat<T> quaternionFromZYZ(Val<T> theta, Val<T> phi, Val<T> lambda,
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
static std::optional<Val<T>> gateParam(UnitaryOpInterface op, unsigned i,
                                       PatternRewriter& rewriter,
                                       Location loc) {
  Value p = op.getParameter(i);
  if constexpr (std::is_same_v<T, double>) {
    const auto folded = mlir::mqt::valueToConstantDouble(p);
    if (!folded) {
      return std::nullopt;
    }
    return Val<T>::constant(rewriter, loc, *folded);
  } else {
    return Val<T>{p, &rewriter, loc};
  }
}

/**
 * @brief Converts a supported single-qubit gate to quaternion representation.
 *
 * - RX, RY, RZ, P: single-axis half-angle formulas.
 * - X, Y, Z, S, Sdg, T, Tdg, SX, SXdg: fixed-axis rotations.
 * - H: a pi rotation around the (X + Z) / sqrt(2) axis.
 * - Id: the identity quaternion.
 * - R(theta, phi): Q(cos(θ/2), sin(θ/2)cos(φ), sin(θ/2)sin(φ), 0).
 * - U2(phi, lambda) = U(π/2, phi, lambda).
 * - U(theta, phi, lambda): ZYZ via quaternionFromZYZ.
 *
 * @note Global phase is discarded; see quaternionFromZYZ for details.
 * @return nullopt if a required parameter cannot be represented as `T`
 *         (static path: unfoldable SSA value).
 */
template <typename T>
static std::optional<Quat<T>> quaternionFromGate(UnitaryOpInterface op,
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

  const auto fixedAxisRotation = [&](RotationAxis axis, double angle) {
    return axisQuaternion(Val<T>::constant(rewriter, loc, angle), axis, c);
  };

  // Fixed and multi-parameter gates each need their own conversion.
  return TypeSwitch<Operation*, std::optional<Quat<T>>>(op.getOperation())
      .template Case<XOp>([&](XOp) {
        return fixedAxisRotation(RotationAxis::X, std::numbers::pi);
      })
      .template Case<YOp>([&](YOp) {
        return fixedAxisRotation(RotationAxis::Y, std::numbers::pi);
      })
      .template Case<ZOp>([&](ZOp) {
        return fixedAxisRotation(RotationAxis::Z, std::numbers::pi);
      })
      .template Case<SOp>([&](SOp) {
        return fixedAxisRotation(RotationAxis::Z, std::numbers::pi / 2.0);
      })
      .template Case<SdgOp>([&](SdgOp) {
        return fixedAxisRotation(RotationAxis::Z, -std::numbers::pi / 2.0);
      })
      .template Case<TOp>([&](TOp) {
        return fixedAxisRotation(RotationAxis::Z, std::numbers::pi / 4.0);
      })
      .template Case<TdgOp>([&](TdgOp) {
        return fixedAxisRotation(RotationAxis::Z, -std::numbers::pi / 4.0);
      })
      .template Case<SXOp>([&](SXOp) {
        return fixedAxisRotation(RotationAxis::X, std::numbers::pi / 2.0);
      })
      .template Case<SXdgOp>([&](SXdgOp) {
        return fixedAxisRotation(RotationAxis::X, -std::numbers::pi / 2.0);
      })
      .template Case<HOp>([&](HOp) -> std::optional<Quat<T>> {
        const auto invSqrtTwo =
            Val<T>::constant(rewriter, loc, 1.0 / std::numbers::sqrt2);
        return Quat<T>{
            .w = c.zero, .x = invSqrtTwo, .y = c.zero, .z = invSqrtTwo};
      })
      .template Case<IdOp>([&](IdOp) -> std::optional<Quat<T>> {
        return Quat<T>{.w = c.one, .x = c.zero, .y = c.zero, .z = c.zero};
      })
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
 * @brief Returns the global phase contribution of a supported gate.
 *
 * Rotation gates can be factored as U = e^{i * phase} * SU(2), where SU(2)
 * is the quaternion-representable part and phase is the global phase:
 *
 * - RX, RY, RZ, R         -> 0 (already SU(2))
 * - P(theta)              -> theta / 2 (P = e^{i * theta / 2} * RZ(theta))
 * - U(theta, phi, lambda) -> (phi + lambda) / 2
 * - U2(phi, lambda)       -> (phi + lambda) / 2
 * - X, Y, Z, H            -> pi / 2
 * - S, SX                 -> pi / 4
 * - Sdg, SXdg             -> -pi / 4
 * - T / Tdg               -> +/- pi / 8
 * - Id                    -> 0
 *
 * @return Success with the phase contribution, including an explicit zero for
 *         SU(2) gates. Failure if a required parameter does not fold on the
 *         static (`double`) path, or if @p op is not a mergeable rotation.
 */
template <typename T>
static FailureOr<Val<T>> globalPhaseOf(UnitaryOpInterface op,
                                       const ScalarConsts<T>& c,
                                       PatternRewriter& rewriter) {
  const Location loc = op->getLoc();
  auto param = [&](unsigned i) { return gateParam<T>(op, i, rewriter, loc); };

  return TypeSwitch<Operation*, FailureOr<Val<T>>>(op.getOperation())
      .template Case<RXOp, RYOp, RZOp, ROp>(
          [&](auto) -> FailureOr<Val<T>> { return c.zero; })
      .template Case<XOp, YOp, ZOp, HOp>(
          [&](auto) -> FailureOr<Val<T>> { return c.pi / c.two; })
      .template Case<SOp, SXOp>([&](auto) -> FailureOr<Val<T>> {
        return Val<T>::constant(rewriter, loc, std::numbers::pi / 4.0);
      })
      .template Case<SdgOp, SXdgOp>([&](auto) -> FailureOr<Val<T>> {
        return Val<T>::constant(rewriter, loc, -std::numbers::pi / 4.0);
      })
      .template Case<TOp>([&](auto) -> FailureOr<Val<T>> {
        return Val<T>::constant(rewriter, loc, std::numbers::pi / 8.0);
      })
      .template Case<TdgOp>([&](auto) -> FailureOr<Val<T>> {
        return Val<T>::constant(rewriter, loc, -std::numbers::pi / 8.0);
      })
      .template Case<IdOp>([&](auto) -> FailureOr<Val<T>> { return c.zero; })
      .template Case<POp>([&](auto) -> FailureOr<Val<T>> {
        const auto theta = param(0);
        if (!theta) {
          return failure();
        }
        return *theta / c.two;
      })
      .template Case<UOp, U2Op>([&](auto) -> FailureOr<Val<T>> {
        // phi is at different indexes for UOp and U2Op
        const auto phiIdx = isa<UOp>(op.getOperation()) ? 1U : 0U;
        const auto phi = param(phiIdx);
        const auto lambda = param(phiIdx + 1);
        if (!phi || !lambda) {
          return failure();
        }
        return (*phi + *lambda) / c.two;
      })
      .Default([](auto) -> FailureOr<Val<T>> { return failure(); });
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
 * so tiny beta drift cannot split the Z angle across phi/lambda. The host path
 * short-circuits to `{0, 2*atan2(z,w), 0}`; the `Value` path selects `beta=0`
 * under the same predicate and sanitizes the atan2 y-operand when (x,y)≈0 so
 * MLIR's constant folder never sees atan2(0,0) → NaN on a dead select input.
 *
 * @note Floating-point errors may accumulate when merging many gates.
 * Normalizing either Z angle by 2*pi flips the corresponding SU(2)
 * quaternion sign. The returned phase correction accounts for those flips.
 *
 * @return {theta, phi, lambda, phaseCorrection} suitable for UOp
 */
template <typename T>
static std::array<Val<T>, 4> anglesFromQuaternion(const Quat<T>& q,
                                                  const ScalarConsts<T>& c) {
  PatternRewriter& rewriter = *q.w.rewriter;
  const Location loc = q.w.loc;

  const auto xyNearZero =
      Val<T>::land(q.x.abs().olt(c.eps), q.y.abs().olt(c.eps), rewriter, loc);

  // Host path can take the pure-Z shortcut without building the full tree.
  if constexpr (std::is_same_v<T, double>) {
    if (xyNearZero) {
      const auto alpha = q.z.atan2(q.w) * c.two;
      const auto phi = wrapToPi(alpha, c);
      // Wrapping alpha by 2*pi flips the SU(2) representative, which is
      // compensated by half the removed angle as a global phase.
      return {c.zero, phi, c.zero, (alpha - phi) / c.two};
    }
  }

  // beta = acos(clamp(2 * (w^2 + z^2) - 1, -1, 1))
  // Force beta=0 when (x,y)≈0 so XY-aligned / pure-Z merges do not emit a
  // drifted acos theta on the SSA path (host path already returned above).
  const auto cosBeta = ((c.two * ((q.w * q.w) + (q.z * q.z))) - c.one)
                           .maximum(c.negOne)
                           .minimum(c.one);
  const auto betaRaw = cosBeta.acos();
  const auto beta = Val<T>::select(xyNearZero, c.zero, betaRaw);

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

  const auto phi = wrapToPi(alpha, c);
  const auto lambda = wrapToPi(gamma, c);
  // Each removed 2*pi Z rotation flips the SU(2) representative. Half of the
  // total removed angle restores the original matrix as a global phase.
  return {beta, phi, lambda, ((alpha - phi) + (gamma - lambda)) / c.two};
}

/**
 * @brief Conjugates @p q by Hadamard, mapping X to Z, Y to -Y, and Z to X.
 */
template <typename T> static Quat<T> hadamardConjugate(const Quat<T>& q) {
  return {.w = q.w, .x = q.z, .y = -q.y, .z = q.x};
}

static bool isMergeable(Operation* op) {
  return isa<RXOp, RYOp, RZOp, POp, ROp, U2Op, UOp, XOp, YOp, ZOp, HOp, SOp,
             SdgOp, TOp, TdgOp, SXOp, SXdgOp, IdOp>(op);
}

static bool areQuaternionMergeable(Operation* a, Operation* b) {
  return isMergeable(a) && isMergeable(b);
}

namespace {

/**
 * @brief Pattern that merges consecutive rotation gates using quaternion
 * multiplication.
 */
struct MergeSingleQubitRotationGatesPattern final
    : OpInterfaceRewritePattern<UnitaryOpInterface> {
  explicit MergeSingleQubitRotationGatesPattern(
      MLIRContext* context,
      std::optional<decomposition::SingleQubitBasis> fusionBasis = std::nullopt,
      const bool skipControlledBodies = false)
      : OpInterfaceRewritePattern(context), fusionBasis(fusionBasis),
        skipControlledBodies(skipControlledBodies) {}

  std::optional<decomposition::SingleQubitBasis> fusionBasis;
  bool skipControlledBodies;

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
   * @brief Whether a chain contains a parameter that cannot be folded.
   */
  static bool hasDynamicParameter(ArrayRef<UnitaryOpInterface> chain) {
    return llvm::any_of(chain, [](UnitaryOpInterface chainOp) {
      return llvm::any_of(chainOp.getParameters(), [](Value parameter) {
        return !mqt::valueToConstantDouble(parameter).has_value();
      });
    });
  }

  /**
   * @brief Conservative gate count for runtime Euler synthesis.
   */
  static size_t
  parameterizedSynthesisGateCount(const decomposition::SingleQubitBasis basis) {
    switch (basis) {
    case decomposition::SingleQubitBasis::U:
      return 1;
    case decomposition::SingleQubitBasis::ZSXX:
      return 5;
    case decomposition::SingleQubitBasis::ZYZ:
    case decomposition::SingleQubitBasis::ZXZ:
    case decomposition::SingleQubitBasis::XZX:
    case decomposition::SingleQubitBasis::XYX:
    case decomposition::SingleQubitBasis::R:
      return 3;
    }
    llvm_unreachable("invalid single-qubit synthesis basis"); // LCOV_EXCL_LINE
  }

  /**
   * @brief Whether the fuser should compose this parameterized chain.
   */
  static bool
  shouldComposeForFusion(ArrayRef<UnitaryOpInterface> chain,
                         const decomposition::SingleQubitBasis basis) {
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
      auto qi = quaternionFromGate<double>(chainOp, consts, rewriter);
      if (!qi) {
        return failure();
      }
      const auto phase = globalPhaseOf<double>(chainOp, consts, rewriter);
      if (failed(phase)) {
        return failure();
      }
      phaseAccum = phaseAccum + *phase;
      qAccum = qAccum ? hamiltonProduct(*qi, *qAccum) : *qi;
    }

    const auto [theta, phi, lambda, eulerPhase] =
        anglesFromQuaternion(*qAccum, consts);
    const auto correction =
        phaseAccum - ((phi + lambda) / consts.two) + eulerPhase;

    for (auto chainOp : llvm::drop_begin(chain)) {
      rewriter.replaceOp(chainOp, chainOp.getInputQubit(0));
    }
    if (std::abs(correction.v) > mqt::PARAMETER_COMPARISON_TOLERANCE) {
      GPhaseOp::create(rewriter, loc,
                       mqt::constantFromScalar(rewriter, loc, correction.v));
    }
    rewriter.replaceOpWithNewOp<UOp>(
        chain.front(), chain.front().getInputQubit(0),
        mqt::constantFromScalar(rewriter, loc, theta.v),
        mqt::constantFromScalar(rewriter, loc, phi.v),
        mqt::constantFromScalar(rewriter, loc, lambda.v));
    return success();
  }

  /**
   * @brief Merge a dynamic or mixed-angle chain via `Val<Value>` SSA.
   *
   * Same quaternion / Euler algorithm as the static path. Fusion mode emits
   * the requested basis directly. The regular merge mode emits U. U output
   * needs its intrinsic global-phase correction:
   *   correction = totalInputPhase - (phi + lambda) / 2
   * The pass-level global-phase normalization subsequently combines and
   * normalizes the emitted correction.
   *
   * Converts every gate before rewriting so a missing conversion Case cannot
   * leave partially rewired ops.
   */
  static LogicalResult mergeDynamicChain(
      MutableArrayRef<UnitaryOpInterface> chain, PatternRewriter& rewriter,
      const std::optional<decomposition::SingleQubitBasis> fusionBasis =
          std::nullopt) {
    const Location loc = chain.front()->getLoc();
    const auto consts = makeConsts<Value>(rewriter, loc);

    SmallVector<Quat<Value>, 4> quats;
    quats.reserve(chain.size());
    Val<Value> phaseAccum = consts.zero;
    for (UnitaryOpInterface chainOp : chain) {
      auto qi = quaternionFromGate<Value>(chainOp, consts, rewriter);
      if (!qi) {
        return failure();
      }
      const auto phase = globalPhaseOf<Value>(chainOp, consts, rewriter);
      if (failed(phase)) {
        return failure();
      }
      quats.push_back(*qi);
      phaseAccum = phaseAccum + *phase;
    }

    Quat<Value> qAccum = quats.front();
    for (const Quat<Value>& qi : llvm::drop_begin(quats)) {
      qAccum = hamiltonProduct(qi, qAccum);
    }

    for (auto chainOp : llvm::drop_begin(chain)) {
      rewriter.replaceOp(chainOp, chainOp.getInputQubit(0));
    }

    const auto basis = fusionBasis.value_or(decomposition::SingleQubitBasis::U);
    const bool transformed = basis == decomposition::SingleQubitBasis::XZX ||
                             basis == decomposition::SingleQubitBasis::XYX ||
                             basis == decomposition::SingleQubitBasis::R;
    auto [theta, phi, lambda, eulerPhase] =
        transformed ? anglesFromQuaternion(hadamardConjugate(qAccum), consts)
                    : anglesFromQuaternion(qAccum, consts);
    if (basis == decomposition::SingleQubitBasis::XZX) {
      phi = phi + (consts.pi / consts.two);
      lambda = lambda - (consts.pi / consts.two);
    } else if (basis == decomposition::SingleQubitBasis::XYX ||
               basis == decomposition::SingleQubitBasis::R) {
      phi = phi + consts.pi;
      lambda = lambda + consts.pi;
      eulerPhase = eulerPhase + consts.pi;
    }
    Val<Value> phaseCorrection = phaseAccum + eulerPhase;
    Value qubit = chain.front().getInputQubit(0);
    switch (basis) {
    case decomposition::SingleQubitBasis::ZYZ:
      qubit = RZOp::create(rewriter, loc, qubit, lambda.v).getQubitOut();
      qubit = RYOp::create(rewriter, loc, qubit, theta.v).getQubitOut();
      qubit = RZOp::create(rewriter, loc, qubit, phi.v).getQubitOut();
      break;
    case decomposition::SingleQubitBasis::ZXZ:
      qubit = RZOp::create(rewriter, loc, qubit,
                           (lambda - (consts.pi / consts.two)).v)
                  .getQubitOut();
      qubit = RXOp::create(rewriter, loc, qubit, theta.v).getQubitOut();
      qubit =
          RZOp::create(rewriter, loc, qubit, (phi + (consts.pi / consts.two)).v)
              .getQubitOut();
      break;
    case decomposition::SingleQubitBasis::XZX:
      qubit = RXOp::create(rewriter, loc, qubit, lambda.v).getQubitOut();
      qubit = RZOp::create(rewriter, loc, qubit, theta.v).getQubitOut();
      qubit = RXOp::create(rewriter, loc, qubit, phi.v).getQubitOut();
      break;
    case decomposition::SingleQubitBasis::XYX:
      qubit = RXOp::create(rewriter, loc, qubit, lambda.v).getQubitOut();
      qubit = RYOp::create(rewriter, loc, qubit, theta.v).getQubitOut();
      qubit = RXOp::create(rewriter, loc, qubit, phi.v).getQubitOut();
      break;
    case decomposition::SingleQubitBasis::U:
      phaseCorrection = phaseCorrection - ((phi + lambda) / consts.two);
      qubit = UOp::create(rewriter, loc, qubit, theta.v, phi.v, lambda.v)
                  .getQubitOut();
      break;
    case decomposition::SingleQubitBasis::ZSXX:
      phaseCorrection = phaseCorrection + (consts.pi / consts.two);
      qubit = RZOp::create(rewriter, loc, qubit, lambda.v).getQubitOut();
      qubit = SXOp::create(rewriter, loc, qubit).getQubitOut();
      qubit = RZOp::create(rewriter, loc, qubit, (theta + consts.pi).v)
                  .getQubitOut();
      qubit = SXOp::create(rewriter, loc, qubit).getQubitOut();
      qubit =
          RZOp::create(rewriter, loc, qubit, (phi + consts.pi).v).getQubitOut();
      break;
    case decomposition::SingleQubitBasis::R:
      qubit = ROp::create(rewriter, loc, qubit, lambda.v, consts.zero.v)
                  .getQubitOut();
      qubit =
          ROp::create(rewriter, loc, qubit, theta.v, (consts.pi / consts.two).v)
              .getQubitOut();
      qubit =
          ROp::create(rewriter, loc, qubit, phi.v, consts.zero.v).getQubitOut();
      break;
    }
    GPhaseOp::create(rewriter, loc, phaseCorrection.v);
    rewriter.replaceOp(chain.front(), qubit);
    return success();
  }

  /**
   * @brief Matches and merges a chain of consecutive rotation gates.
   *
   * Detects the full chain of mergeable operations, folds their quaternions
   * via Hamilton product, and emits a single UOp or the requested fusion basis.
   * Fully static chains use host STL math; otherwise the SSA
   * `arith`/`math` path is used.
   */
  LogicalResult matchAndRewrite(UnitaryOpInterface op,
                                PatternRewriter& rewriter) const override {
    if (skipControlledBodies &&
        (op.getOperation()->getParentOfType<CtrlOp>() != nullptr)) {
      return failure();
    }
    if (!isChainStart(op)) {
      return failure();
    }

    auto chain = collectChain(op);
    // Emit all helper ops at the chain tail so the merged output is adjacent to
    // the last gate it replaces.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(chain.back().getOperation());

    if (fusionBasis) {
      if (!shouldComposeForFusion(chain, *fusionBasis)) {
        return failure();
      }
      return mergeDynamicChain(chain, rewriter, fusionBasis);
    }
    if (chain.size() < 2) {
      return failure();
    }

    if (succeeded(tryMergeStaticChain(chain, rewriter))) {
      return success();
    }
    return mergeDynamicChain(chain, rewriter);
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

namespace mlir::qco::decomposition {

void populateParameterizedSingleQubitRunCompositionPatterns(
    RewritePatternSet& patterns, const SingleQubitBasis basis,
    const bool skipControlledBodies) {
  if (!skipControlledBodies) {
    RXOp::getCanonicalizationPatterns(patterns, patterns.getContext());
    RYOp::getCanonicalizationPatterns(patterns, patterns.getContext());
    RZOp::getCanonicalizationPatterns(patterns, patterns.getContext());
    POp::getCanonicalizationPatterns(patterns, patterns.getContext());
  }
  patterns.add<MergeSingleQubitRotationGatesPattern>(
      patterns.getContext(), basis, skipControlledBodies);
}

} // namespace mlir::qco::decomposition
