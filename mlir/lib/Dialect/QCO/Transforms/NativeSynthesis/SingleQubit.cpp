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
#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Types.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Utils.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>

#include <numbers>

namespace mlir::qco::native_synth {

constexpr double PI = std::numbers::pi;
constexpr double HALF_PI = PI / 2.0;

namespace {

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

} // namespace

Value decomposeToZSXX(IRRewriter& rewriter, Operation* op, Value inQubit,
                      bool supportsDirectRx) {
  if (llvm::isa<IdOp>(op)) {
    return inQubit;
  }
  SingleQubitEmitter e{.rewriter = &rewriter, .loc = op->getLoc()};
  if (auto p = llvm::dyn_cast<POp>(op)) {
    auto q = e.rz(inQubit, p.getTheta());
    auto halfTheta = arith::MulFOp::create(rewriter, op->getLoc(), p.getTheta(),
                                           e.constF(0.5))
                         .getResult();
    GPhaseOp::create(rewriter, op->getLoc(), halfTheta);
    return q;
  }
  if (!supportsDirectRx) {
    return {};
  }
  if (auto rx = llvm::dyn_cast<RXOp>(op)) {
    return rx.getOutputQubit(0);
  }
  if (auto ry = llvm::dyn_cast<RYOp>(op)) {
    return e.rz(e.rx(e.rz(inQubit, -HALF_PI), ry.getTheta()), HALF_PI);
  }
  if (auto r = llvm::dyn_cast<ROp>(op)) {
    auto negPhi =
        arith::NegFOp::create(rewriter, op->getLoc(), r.getPhi()).getResult();
    return e.rz(e.rx(e.rz(inQubit, negPhi), r.getTheta()), r.getPhi());
  }
  return {};
}

Value decomposeToU3(IRRewriter& rewriter, Operation* op, Value inQubit) {
  if (llvm::isa<IdOp>(op)) {
    return inQubit;
  }
  SingleQubitEmitter e{.rewriter = &rewriter, .loc = op->getLoc()};
  if (auto u = llvm::dyn_cast<UOp>(op)) {
    return u.getOutputQubit(0);
  }
  if (auto rx = llvm::dyn_cast<RXOp>(op)) {
    return e.u(inQubit, rx.getTheta(), e.constF(-HALF_PI), e.constF(HALF_PI));
  }
  if (auto ry = llvm::dyn_cast<RYOp>(op)) {
    return e.u(inQubit, ry.getTheta(), e.constF(0.0), e.constF(0.0));
  }
  if (auto rz = llvm::dyn_cast<RZOp>(op)) {
    auto out = e.u(inQubit, e.constF(0.0), e.constF(0.0), rz.getTheta());
    auto halfTheta = arith::MulFOp::create(rewriter, op->getLoc(),
                                           rz.getTheta(), e.constF(-0.5))
                         .getResult();
    GPhaseOp::create(rewriter, op->getLoc(), halfTheta);
    return out;
  }
  if (auto p = llvm::dyn_cast<POp>(op)) {
    return e.u(inQubit, e.constF(0.0), e.constF(0.0), p.getTheta());
  }
  if (auto u2 = llvm::dyn_cast<U2Op>(op)) {
    return e.u(inQubit, e.constF(HALF_PI), u2.getPhi(), u2.getLambda());
  }
  if (auto r = llvm::dyn_cast<ROp>(op)) {
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

Value emitSingleQubitMatrix(IRRewriter& rewriter, Location loc, Value inQubit,
                            const Matrix2x2& matrix,
                            const decomposition::EulerBasis basis) {
  // Force emission (`hasNonBasisGate = true`, `runSize = 0`) so the matrix is
  // always lowered into native gates of `basis`, including any residual
  // `qco.gphase`. With these arguments `synthesizeUnitary1QEuler` never
  // returns `std::nullopt`.
  return *decomposition::synthesizeUnitary1QEuler(
      rewriter, loc, inQubit, matrix, /*runSize=*/0,
      /*hasNonBasisGate=*/true, basis);
}

Value decomposeToR(IRRewriter& rewriter, Operation* op, Value inQubit) {
  if (llvm::isa<IdOp>(op)) {
    return inQubit;
  }
  SingleQubitEmitter e{.rewriter = &rewriter, .loc = op->getLoc()};
  if (auto r = llvm::dyn_cast<ROp>(op)) {
    return r.getOutputQubit(0);
  }
  if (auto rx = llvm::dyn_cast<RXOp>(op)) {
    return e.r(inQubit, rx.getTheta(), e.constF(0.0));
  }
  if (auto ry = llvm::dyn_cast<RYOp>(op)) {
    return e.r(inQubit, ry.getTheta(), e.constF(HALF_PI));
  }
  return {};
}

Value decomposeToAxisPair(IRRewriter& rewriter, Operation* op, Value inQubit,
                          AxisPair axisPair) {
  if (llvm::isa<IdOp>(op)) {
    return inQubit;
  }
  SingleQubitEmitter e{.rewriter = &rewriter, .loc = op->getLoc()};
  switch (axisPair) {
  case AxisPair::RxRz:
    if (auto rx = llvm::dyn_cast<RXOp>(op)) {
      return rx.getOutputQubit(0);
    }
    if (auto rz = llvm::dyn_cast<RZOp>(op)) {
      return rz.getOutputQubit(0);
    }
    if (auto p = llvm::dyn_cast<POp>(op)) {
      auto q = e.rz(inQubit, p.getTheta());
      auto halfTheta = arith::MulFOp::create(rewriter, op->getLoc(),
                                             p.getTheta(), e.constF(0.5))
                           .getResult();
      GPhaseOp::create(rewriter, op->getLoc(), halfTheta);
      return q;
    }
    return {};
  case AxisPair::RxRy:
    if (auto rx = llvm::dyn_cast<RXOp>(op)) {
      return rx.getOutputQubit(0);
    }
    if (auto ry = llvm::dyn_cast<RYOp>(op)) {
      return ry.getOutputQubit(0);
    }
    return {};
  case AxisPair::RyRz:
    if (auto ry = llvm::dyn_cast<RYOp>(op)) {
      return ry.getOutputQubit(0);
    }
    if (auto rz = llvm::dyn_cast<RZOp>(op)) {
      return rz.getOutputQubit(0);
    }
    if (auto p = llvm::dyn_cast<POp>(op)) {
      auto q = e.rz(inQubit, p.getTheta());
      auto halfTheta = arith::MulFOp::create(rewriter, op->getLoc(),
                                             p.getTheta(), e.constF(0.5))
                           .getResult();
      GPhaseOp::create(rewriter, op->getLoc(), halfTheta);
      return q;
    }
    return {};
  }
  llvm_unreachable("unknown axis pair");
}

} // namespace mlir::qco::native_synth
