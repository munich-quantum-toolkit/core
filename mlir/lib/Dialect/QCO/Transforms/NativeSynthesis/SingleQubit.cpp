/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/SingleQubit.h"

#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/EulerDecomposition.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/NativeSpec.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Utils.h"

#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>

#include <cmath>
#include <complex>
#include <limits>
#include <numbers>

namespace mlir::qco::native_synth {
namespace {

constexpr double PI = std::numbers::pi;
constexpr double HALF_PI = PI / 2.0;

/// Small convenience wrapper to avoid passing rewriter/loc everywhere. Each
/// method creates the corresponding QCO op threaded through `q` and returns
/// its new output qubit value.
struct SingleQubitEmitter {
  IRRewriter* rewriter;
  Location loc;

  /// Create an `arith.constant` `f64` of value `v` at `loc`.
  [[nodiscard]] Value constF(double v) const {
    return createF64Const(*rewriter, loc, v);
  }

  /// Emit `rx(theta)` with a compile-time scalar angle.
  [[nodiscard]] Value rx(Value q, double theta) const {
    return RXOp::create(*rewriter, loc, q, constF(theta)).getOutputQubit(0);
  }
  /// Emit `rx(theta)` with a runtime `f64` angle value.
  [[nodiscard]] Value rx(Value q, Value theta) const {
    return RXOp::create(*rewriter, loc, q, theta).getOutputQubit(0);
  }
  /// Emit `ry(theta)` with a compile-time scalar angle.
  [[nodiscard]] Value ry(Value q, double theta) const {
    return RYOp::create(*rewriter, loc, q, constF(theta)).getOutputQubit(0);
  }
  /// Emit `ry(theta)` with a runtime `f64` angle value.
  [[nodiscard]] Value ry(Value q, Value theta) const {
    return RYOp::create(*rewriter, loc, q, theta).getOutputQubit(0);
  }
  /// Emit `rz(theta)` with a compile-time scalar angle.
  [[nodiscard]] Value rz(Value q, double theta) const {
    return RZOp::create(*rewriter, loc, q, constF(theta)).getOutputQubit(0);
  }
  /// Emit `rz(theta)` with a runtime `f64` angle value.
  [[nodiscard]] Value rz(Value q, Value theta) const {
    return RZOp::create(*rewriter, loc, q, theta).getOutputQubit(0);
  }
  /// Emit `sx` (square-root-of-X).
  [[nodiscard]] Value sx(Value q) const {
    return SXOp::create(*rewriter, loc, q).getOutputQubit(0);
  }
  /// Emit a Pauli `x`.
  [[nodiscard]] Value x(Value q) const {
    return XOp::create(*rewriter, loc, q).getOutputQubit(0);
  }
  /// Emit `r(theta, phi)` with compile-time scalar angles.
  [[nodiscard]] Value r(Value q, double theta, double phi) const {
    return ROp::create(*rewriter, loc, q, constF(theta), constF(phi))
        .getOutputQubit(0);
  }
  /// Emit `r(theta, phi)` with runtime `f64` angle values.
  [[nodiscard]] Value r(Value q, Value theta, Value phi) const {
    return ROp::create(*rewriter, loc, q, theta, phi).getOutputQubit(0);
  }
  /// Emit `u(theta, phi, lambda)` with runtime `f64` angle values.
  [[nodiscard]] Value u(Value q, Value theta, Value phi, Value lambda) const {
    return UOp::create(*rewriter, loc, q, theta, phi, lambda).getOutputQubit(0);
  }
  /// Emit `u(theta, phi, lambda)` with compile-time scalar angles.
  [[nodiscard]] Value u(Value q, double theta, double phi,
                        double lambda) const {
    return u(q, constF(theta), constF(phi), constF(lambda));
  }
};

/// Materialize an `EulerBasis::ZSXX` decomposition (`rz` / `sx` / `x`) into
/// QCO ops. Returns null on unsupported abstract gate kinds.
Value emitEulerSequenceZsxx(SingleQubitEmitter e, Value q,
                            const decomposition::QubitGateSequence& seq) {
  for (const auto& gate : seq.gates) {
    switch (gate.type) {
    case decomposition::GateKind::RZ:
      if (gate.parameter.size() != 1) {
        return {};
      }
      q = e.rz(q, gate.parameter[0]);
      break;
    case decomposition::GateKind::SX:
      q = e.sx(q);
      break;
    case decomposition::GateKind::X:
      q = e.x(q);
      break;
    case decomposition::GateKind::I:
      break;
    default:
      return {};
    }
  }
  return q;
}

/// Materialize an `EulerBasis::XYX` decomposition into `R(theta, phi)` ops
/// for the `R` emitter: `Rx(theta)` becomes `R(theta, 0)`, `Ry(theta)`
/// becomes `R(theta, pi/2)`, Pauli `X`/`Y` become `R(pi, *)`, `I` is a
/// no-op. Returns null on any unsupported abstract gate kind.
Value emitEulerSequenceR(SingleQubitEmitter e, Value q,
                         const decomposition::QubitGateSequence& seq) {
  for (const auto& gate : seq.gates) {
    switch (gate.type) {
    case decomposition::GateKind::RX:
      if (gate.parameter.size() != 1) {
        return {};
      }
      q = e.r(q, gate.parameter[0], 0.0);
      break;
    case decomposition::GateKind::RY:
      if (gate.parameter.size() != 1) {
        return {};
      }
      q = e.r(q, gate.parameter[0], HALF_PI);
      break;
    case decomposition::GateKind::X:
      q = e.r(q, PI, 0.0);
      break;
    case decomposition::GateKind::Y:
      q = e.r(q, PI, HALF_PI);
      break;
    case decomposition::GateKind::I:
      break;
    default:
      return {};
    }
  }
  return q;
}

/// Materialize an Euler decomposition in the two rotation axes named by
/// `axis` (e.g. `{Rx, Rz}`). Every gate kind that falls outside the two
/// chosen axes (or has the wrong parameter count) is rejected by returning
/// a null `Value`; the matrix-based fallback is expected to pick a
/// different basis in that case. Pauli gates are lowered to the
/// corresponding `R*(pi)` when their axis is available.
Value emitEulerSequenceAxisPair(SingleQubitEmitter e, Value q, AxisPair axis,
                                const decomposition::QubitGateSequence& seq) {
  for (const auto& gate : seq.gates) {
    switch (gate.type) {
    case decomposition::GateKind::RX:
      if (axis == AxisPair::RyRz || gate.parameter.size() != 1) {
        return {};
      }
      q = e.rx(q, gate.parameter[0]);
      break;
    case decomposition::GateKind::RY:
      if (axis == AxisPair::RxRz || gate.parameter.size() != 1) {
        return {};
      }
      q = e.ry(q, gate.parameter[0]);
      break;
    case decomposition::GateKind::RZ:
      if (axis == AxisPair::RxRy || gate.parameter.size() != 1) {
        return {};
      }
      q = e.rz(q, gate.parameter[0]);
      break;
    case decomposition::GateKind::X:
      if (axis == AxisPair::RyRz) {
        return {};
      }
      q = e.rx(q, PI);
      break;
    case decomposition::GateKind::Y:
      if (axis == AxisPair::RxRz) {
        return {};
      }
      q = e.ry(q, PI);
      break;
    case decomposition::GateKind::Z:
      if (axis == AxisPair::RxRy) {
        return {};
      }
      q = e.rz(q, PI);
      break;
    case decomposition::GateKind::I:
      break;
    default:
      return {};
    }
  }
  return q;
}

/// Decompose `matrix` numerically into a gate sequence in `basis` with
/// zero-rotations pruned (`simplify=true`). Pure forwarder around
/// `EulerDecomposition::generateCircuit` kept as a one-liner to match the
/// matrix-based fallback call sites in `decomposeTo*`.
decomposition::QubitGateSequence runEuler(decomposition::EulerBasis basis,
                                          const Eigen::Matrix2cd& matrix) {
  return decomposition::EulerDecomposition::generateCircuit(
      basis, matrix, /*simplify=*/true, std::nullopt);
}

} // namespace

// Direct emitters only handle the gates listed in the matching
// `canDirectlyDecomposeTo*` predicate. Everything else is expected to reach the
// matrix-based Euler fallback, which produces an equivalent native sequence in
// the same basis.
Value decomposeToZSXX(IRRewriter& rewriter, Operation* op, Value inQubit,
                      bool supportsDirectRx) {
  if (isa<IdOp>(op)) {
    return inQubit;
  }
  SingleQubitEmitter e{.rewriter = &rewriter, .loc = op->getLoc()};
  if (auto p = dyn_cast<POp>(op)) {
    return e.rz(inQubit, p.getTheta());
  }
  if (!supportsDirectRx) {
    return {};
  }
  if (auto rx = dyn_cast<RXOp>(op)) {
    return rx.getOutputQubit(0);
  }
  if (auto ry = dyn_cast<RYOp>(op)) {
    return e.rz(e.rx(e.rz(inQubit, -HALF_PI), ry.getTheta()), HALF_PI);
  }
  if (auto r = dyn_cast<ROp>(op)) {
    auto negPhi =
        arith::NegFOp::create(rewriter, op->getLoc(), r.getPhi()).getResult();
    return e.rz(e.rx(e.rz(inQubit, negPhi), r.getTheta()), r.getPhi());
  }
  return {};
}

Value decomposeToU3(IRRewriter& rewriter, Operation* op, Value inQubit) {
  if (isa<IdOp>(op)) {
    return inQubit;
  }
  SingleQubitEmitter e{.rewriter = &rewriter, .loc = op->getLoc()};
  if (auto u = dyn_cast<UOp>(op)) {
    return u.getOutputQubit(0);
  }
  if (auto rx = dyn_cast<RXOp>(op)) {
    return e.u(inQubit, rx.getTheta(), e.constF(-HALF_PI), e.constF(HALF_PI));
  }
  if (auto ry = dyn_cast<RYOp>(op)) {
    return e.u(inQubit, ry.getTheta(), e.constF(0.0), e.constF(0.0));
  }
  if (auto rz = dyn_cast<RZOp>(op)) {
    return e.u(inQubit, e.constF(0.0), e.constF(0.0), rz.getTheta());
  }
  if (auto p = dyn_cast<POp>(op)) {
    return e.u(inQubit, e.constF(0.0), e.constF(0.0), p.getTheta());
  }
  if (auto u2 = dyn_cast<U2Op>(op)) {
    return e.u(inQubit, e.constF(HALF_PI), u2.getPhi(), u2.getLambda());
  }
  if (auto r = dyn_cast<ROp>(op)) {
    auto loc = op->getLoc();
    auto phiMinus =
        arith::AddFOp::create(rewriter, loc, r.getPhi(), e.constF(-HALF_PI))
            .getResult();
    auto negPhi = arith::NegFOp::create(rewriter, loc, r.getPhi()).getResult();
    auto minusPlus =
        arith::AddFOp::create(rewriter, loc, negPhi, e.constF(HALF_PI))
            .getResult();
    return e.u(inQubit, r.getTheta(), phiMinus, minusPlus);
  }
  return {};
}

std::size_t
computeSynthesizedSingleQubitLength(const Eigen::Matrix2cd& matrix,
                                    const SingleQubitEmitterSpec& emitter) {
  // `U3` always emits a single gate; every other mode maps to a fixed Euler
  // basis whose decomposition length we can measure directly.
  switch (emitter.mode) {
  case SingleQubitMode::U3:
    return 1;
  case SingleQubitMode::ZSXX:
    return runEuler(decomposition::EulerBasis::ZSXX, matrix).gates.size();
  case SingleQubitMode::R:
    return runEuler(decomposition::EulerBasis::XYX, matrix).gates.size();
  case SingleQubitMode::AxisPair: {
    const auto bases = getEulerBasesForAxisPair(emitter.axisPair);
    if (bases.empty()) {
      return std::numeric_limits<std::size_t>::max();
    }
    return runEuler(bases.front(), matrix).gates.size();
  }
  }
  llvm_unreachable("unknown single-qubit mode");
}

Value emitSynthesizedSingleQubitFromMatrix(
    IRRewriter& rewriter, Location loc, Value inQubit,
    const Eigen::Matrix2cd& matrix, const SingleQubitEmitterSpec& emitter) {
  SingleQubitEmitter e{.rewriter = &rewriter, .loc = loc};
  switch (emitter.mode) {
  case SingleQubitMode::ZSXX: {
    const auto seq = runEuler(decomposition::EulerBasis::ZSXX, matrix);
    emitGPhaseIfNonTrivial(rewriter, loc, seq.globalPhase);
    return emitEulerSequenceZsxx(e, inQubit, seq);
  }
  case SingleQubitMode::U3: {
    using namespace std::complex_literals;

    // Project `matrix` into SU(2) before running the Euler decomposition.
    // For a 2x2 unitary, det(U) sits on the unit circle, so dividing by the
    // square root of det fixes det == 1. We use `arg(det) / 2` (not
    // `/ 4` as in the 4x4 case) because `sqrt(det) = exp(i * arg(det) / 2)`.
    // The removed global phase is re-emitted via `emitGPhaseIfNonTrivial`
    // so the final sequence equals the original unitary, not just SU(2)-up
    // to global phase.
    Eigen::Matrix2cd m = matrix;
    const auto det = m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
    const double phase = std::arg(det) / 2.0;
    m *= std::exp(1i * (-phase));
    emitGPhaseIfNonTrivial(rewriter, loc, phase);
    const auto angles = decomposition::EulerDecomposition::anglesFromUnitary(
        m, decomposition::EulerBasis::ZYZ);
    return e.u(inQubit, angles[0], angles[1], angles[2]);
  }
  case SingleQubitMode::R: {
    const auto seq = runEuler(decomposition::EulerBasis::XYX, matrix);
    emitGPhaseIfNonTrivial(rewriter, loc, seq.globalPhase);
    return emitEulerSequenceR(e, inQubit, seq);
  }
  case SingleQubitMode::AxisPair: {
    const auto bases = getEulerBasesForAxisPair(emitter.axisPair);
    if (bases.empty()) {
      return {};
    }
    const auto seq = runEuler(bases.front(), matrix);
    emitGPhaseIfNonTrivial(rewriter, loc, seq.globalPhase);
    return emitEulerSequenceAxisPair(e, inQubit, emitter.axisPair, seq);
  }
  }
  llvm_unreachable("unknown single-qubit mode");
}

Value decomposeToR(IRRewriter& rewriter, Operation* op, Value inQubit) {
  if (isa<IdOp>(op)) {
    return inQubit;
  }
  SingleQubitEmitter e{.rewriter = &rewriter, .loc = op->getLoc()};
  if (auto r = dyn_cast<ROp>(op)) {
    return r.getOutputQubit(0);
  }
  if (auto rx = dyn_cast<RXOp>(op)) {
    return e.r(inQubit, rx.getTheta(), e.constF(0.0));
  }
  if (auto ry = dyn_cast<RYOp>(op)) {
    return e.r(inQubit, ry.getTheta(), e.constF(HALF_PI));
  }
  return {};
}

Value decomposeToAxisPair(IRRewriter& rewriter, Operation* op, Value inQubit,
                          AxisPair axisPair) {
  if (isa<IdOp>(op)) {
    return inQubit;
  }
  SingleQubitEmitter e{.rewriter = &rewriter, .loc = op->getLoc()};
  switch (axisPair) {
  case AxisPair::RxRz:
    if (auto rx = dyn_cast<RXOp>(op)) {
      return rx.getOutputQubit(0);
    }
    if (auto rz = dyn_cast<RZOp>(op)) {
      return rz.getOutputQubit(0);
    }
    if (auto p = dyn_cast<POp>(op)) {
      return e.rz(inQubit, p.getTheta());
    }
    return {};
  case AxisPair::RxRy:
    if (auto rx = dyn_cast<RXOp>(op)) {
      return rx.getOutputQubit(0);
    }
    if (auto ry = dyn_cast<RYOp>(op)) {
      return ry.getOutputQubit(0);
    }
    return {};
  case AxisPair::RyRz:
    if (auto ry = dyn_cast<RYOp>(op)) {
      return ry.getOutputQubit(0);
    }
    if (auto rz = dyn_cast<RZOp>(op)) {
      return rz.getOutputQubit(0);
    }
    if (auto p = dyn_cast<POp>(op)) {
      return e.rz(inQubit, p.getTheta());
    }
    return {};
  }
  llvm_unreachable("unknown axis pair");
}

} // namespace mlir::qco::native_synth
