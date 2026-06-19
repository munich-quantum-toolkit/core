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
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Euler.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Weyl.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Support/WalkResult.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numbers>
#include <optional>
#include <utility>
#include <vector>

namespace mlir::qco {

#define GEN_PASS_DEF_FUSETWOQUBITUNITARYRUNS
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

} // namespace mlir::qco

// The following three functions are part of this pass's internal logic but are
// also exercised directly by unit tests, so they live in a named namespace that
// the tests can forward-declare. Everything else is file-local (see the
// anonymous namespace below).
namespace mlir::qco::native_synth {

using decomposition::NativeGateKind;
using decomposition::NativeProfileSpec;

/// Map a single-qubit `UnitaryOpInterface` op to the `NativeGateKind` that
/// must appear in the menu for the op to be a no-op.
static std::optional<NativeGateKind>
singleQubitNativeGateKind(UnitaryOpInterface op) {
  Operation* raw = op.getOperation();
  if (llvm::isa<UOp>(raw)) {
    return NativeGateKind::U;
  }
  if (llvm::isa<XOp>(raw)) {
    return NativeGateKind::X;
  }
  if (llvm::isa<SXOp>(raw)) {
    return NativeGateKind::Sx;
  }
  if (llvm::isa<RZOp, POp>(raw)) {
    // `p` is a Z-rotation primitive for menu purposes.
    return NativeGateKind::Rz;
  }
  if (llvm::isa<RXOp>(raw)) {
    return NativeGateKind::Rx;
  }
  if (llvm::isa<RYOp>(raw)) {
    return NativeGateKind::Ry;
  }
  if (llvm::isa<ROp>(raw)) {
    return NativeGateKind::R;
  }
  return std::nullopt;
}

// NOLINTNEXTLINE(misc-use-internal-linkage): test-visible (see comment above).
bool allowsSingleQubitOp(UnitaryOpInterface op, const NativeProfileSpec& spec) {
  if (llvm::isa<BarrierOp, GPhaseOp>(op.getOperation())) {
    return true;
  }
  const auto gate = singleQubitNativeGateKind(op);
  return gate && spec.allowedGates.contains(*gate);
}

// NOLINTNEXTLINE(misc-use-internal-linkage): test-visible (see comment above).
bool getBlockTwoQubitMatrix(Operation* op, Matrix4x4& matrix) {
  if (llvm::isa<BarrierOp, GPhaseOp>(op)) {
    return false;
  }
  if (auto ctrl = llvm::dyn_cast<CtrlOp>(op)) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      return false;
    }
    auto* body = ctrl.getBodyUnitary(0).getOperation();
    if (llvm::isa<XOp>(body)) {
      matrix = Matrix4x4::fromElements(1, 0, 0, 0, //
                                       0, 1, 0, 0, //
                                       0, 0, 0, 1, //
                                       0, 0, 1, 0);
      return true;
    }
    if (llvm::isa<ZOp>(body)) {
      matrix = Matrix4x4::identity();
      matrix(3, 3) = -1.0;
      return true;
    }
    return false;
  }
  auto unitary = llvm::dyn_cast<UnitaryOpInterface>(op);
  if (!unitary || !unitary.isTwoQubit()) {
    return false;
  }
  Matrix4x4 raw;
  if (!unitary.getUnitaryMatrix4x4(raw)) {
    return false;
  }
  matrix = raw;
  return true;
}

} // namespace mlir::qco::native_synth

namespace mlir::qco {
namespace {

using decomposition::AxisPair;
using decomposition::EntanglerBasis;
using decomposition::NativeGateKind;
using decomposition::NativeProfileSpec;
using decomposition::parseNativeSpec;
using decomposition::SingleQubitEmitterSpec;
using decomposition::SingleQubitMode;
using decomposition::synthesizeUnitary2QWeyl;
using decomposition::twoQubitEntanglerCount;
using native_synth::allowsSingleQubitOp;
using native_synth::getBlockTwoQubitMatrix;

constexpr double PI = std::numbers::pi;
constexpr double HALF_PI = PI / 2.0;

using QubitId = std::size_t;

[[nodiscard]] const Matrix4x4& swapGate() {
  static const Matrix4x4 matrix = Matrix4x4::fromElements(1, 0, 0, 0, //
                                                          0, 0, 1, 0, //
                                                          0, 1, 0, 0, //
                                                          0, 0, 0, 1);
  return matrix;
}

[[nodiscard]] Matrix4x4 expandToTwoQubits(const Matrix2x2& singleQubitMatrix,
                                          QubitId qubitId) {
  if (qubitId == 0) {
    return kron(singleQubitMatrix, Matrix2x2::identity());
  }
  if (qubitId == 1) {
    return kron(Matrix2x2::identity(), singleQubitMatrix);
  }
  llvm::reportFatalInternalError("Invalid qubit id for single-qubit expansion");
}

[[nodiscard]] Matrix4x4
fixTwoQubitMatrixQubitOrder(const Matrix4x4& twoQubitMatrix,
                            const llvm::SmallVector<QubitId, 2>& qubitIds) {
  if (qubitIds == llvm::SmallVector<QubitId, 2>{1, 0}) {
    return swapGate() * twoQubitMatrix * swapGate();
  }
  if (qubitIds == llvm::SmallVector<QubitId, 2>{0, 1}) {
    return twoQubitMatrix;
  }
  llvm::reportFatalInternalError(
      "Invalid qubit IDs for fixing two-qubit matrix");
}

[[nodiscard]] decomposition::EulerBasis
emitterEulerBasis(const SingleQubitEmitterSpec& emitter) {
  switch (emitter.mode) {
  case SingleQubitMode::ZSXX:
    return decomposition::EulerBasis::ZSXX;
  case SingleQubitMode::U3:
    return decomposition::EulerBasis::U;
  case SingleQubitMode::R:
    return decomposition::EulerBasis::R;
  case SingleQubitMode::AxisPair:
    switch (emitter.axisPair) {
    case AxisPair::RxRz:
      return decomposition::EulerBasis::XZX;
    case AxisPair::RxRy:
      return decomposition::EulerBasis::XYX;
    case AxisPair::RyRz:
      return decomposition::EulerBasis::ZYZ;
    }
    break;
  }
  llvm_unreachable("unknown single-qubit mode");
}

[[nodiscard]] bool usesCxEntangler(const NativeProfileSpec& spec) {
  return llvm::is_contained(spec.entanglerBases, EntanglerBasis::Cx);
}

[[nodiscard]] bool usesCzEntangler(const NativeProfileSpec& spec) {
  return llvm::is_contained(spec.entanglerBases, EntanglerBasis::Cz);
}

/// True when `decomposeTo*` should run instead of folding to a constant `2×2`
/// matrix: trivial `Id`/`P`, dynamic-angle ops the matrix path cannot close
/// over, and (for ZSXX with direct Rx) `Rx`/`Ry`/`R`. Static angles still use
/// matrix + Euler.
bool canDirectlyDecomposeToZSXX(Operation* op, bool supportsDirectRx) {
  if (llvm::isa<IdOp, POp>(op)) {
    return true;
  }
  return supportsDirectRx && llvm::isa<RXOp, RYOp, ROp>(op);
}

bool canDirectlyDecomposeToU3(Operation* op) {
  return llvm::isa<IdOp, RXOp, RYOp, RZOp, POp, U2Op, ROp, UOp>(op);
}

bool canDirectlyDecomposeToR(Operation* op) {
  return llvm::isa<IdOp, ROp, RXOp, RYOp>(op);
}

bool canDirectlyDecomposeToAxisPair(Operation* op, AxisPair axisPair) {
  if (llvm::isa<IdOp>(op)) {
    return true;
  }
  switch (axisPair) {
  case AxisPair::RxRz:
    // `p` on an Rx/Rz axis pair folds directly to `rz(theta)`.
    return llvm::isa<RXOp, RZOp, POp>(op);
  case AxisPair::RxRy:
    // No cheap symbolic lowering of `p` without `rz` available.
    return llvm::isa<RXOp, RYOp>(op);
  case AxisPair::RyRz:
    return llvm::isa<RYOp, RZOp, POp>(op);
  }
  llvm_unreachable("unknown axis pair");
}

Value createF64Const(IRRewriter& rewriter, Location loc, double value) {
  return arith::ConstantFloatOp::create(rewriter, loc, rewriter.getF64Type(),
                                        llvm::APFloat(value))
      .getResult();
}

std::optional<double> getConstantF64(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantFloatOp>()) {
    if (auto floatAttr = llvm::dyn_cast<FloatAttr>(constant.getValue())) {
      return floatAttr.getValueAsDouble();
    }
  }
  return std::nullopt;
}

void emitGPhaseIfNonTrivial(IRRewriter& rewriter, Location loc, double phase) {
  constexpr double epsilon = 1e-12;
  if (std::abs(phase) > epsilon) {
    GPhaseOp::create(rewriter, loc, createF64Const(rewriter, loc, phase));
  }
}

void collectUnitaryOpsInPreOrder(Operation* root,
                                 llvm::SmallVectorImpl<Operation*>& ops) {
  root->walk([&](Operation* op) {
    if (op->getParentOfType<CtrlOp>()) {
      return;
    }
    if (!llvm::isa<InvOp>(op) && op->getParentOfType<InvOp>()) {
      return;
    }
    if (llvm::isa<UnitaryOpInterface>(op)) {
      ops.push_back(op);
    }
  });
}

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

LogicalResult rewriteXXPlusMinusYYViaRzz(IRRewriter& rewriter, Operation* op) {
  rewriter.setInsertionPoint(op);
  const auto loc = op->getLoc();
  const auto constF = [&](double v) {
    return createF64Const(rewriter, loc, v);
  };
  const auto half = [&](Value v) -> Value {
    if (auto c = getConstantF64(v)) {
      return constF(*c * 0.5);
    }
    return arith::MulFOp::create(rewriter, loc, v, constF(0.5)).getResult();
  };
  const auto neg = [&](Value v) -> Value {
    if (auto c = getConstantF64(v)) {
      return constF(-*c);
    }
    return arith::NegFOp::create(rewriter, loc, v).getResult();
  };
  const auto emitH = [&](Value q) -> Value {
    auto rz0 = RZOp::create(rewriter, loc, q, constF(HALF_PI));
    auto sx = SXOp::create(rewriter, loc, rz0.getOutputQubit(0));
    return RZOp::create(rewriter, loc, sx.getOutputQubit(0), constF(HALF_PI))
        .getOutputQubit(0);
  };
  // Realize `Rxx(theta)` as `(H ⊗ H) * Rzz(theta) * (H ⊗ H)`: Hadamard
  // conjugation maps the Z axis to X on each qubit, and the tensor-product
  // identity `(H ⊗ H) * ZZ * (H ⊗ H) == XX` lifts that to the entangler.
  const auto emitRxxViaRzz = [&](Value q0, Value q1,
                                 Value theta) -> std::pair<Value, Value> {
    q0 = emitH(q0);
    q1 = emitH(q1);
    auto rzz = RZZOp::create(rewriter, loc, q0, q1, theta);
    q0 = rzz.getOutputQubit(0);
    q1 = rzz.getOutputQubit(1);
    return {emitH(q0), emitH(q1)};
  };
  // Realize `Ryy(theta)` as `(Rx(-pi/2) ⊗ Rx(-pi/2)) * Rzz(theta) *
  // (Rx(pi/2) ⊗ Rx(pi/2))`: Rx(pi/2) maps Z to Y on each qubit, so the
  // conjugation transports `ZZ` to `YY` just like the Hadamard sandwich
  // above maps it to `XX`.
  const auto emitRyyViaRzz = [&](Value q0, Value q1,
                                 Value theta) -> std::pair<Value, Value> {
    auto rx0 = RXOp::create(rewriter, loc, q0, constF(HALF_PI));
    auto rx1 = RXOp::create(rewriter, loc, q1, constF(HALF_PI));
    auto rzz = RZZOp::create(rewriter, loc, rx0.getOutputQubit(0),
                             rx1.getOutputQubit(0), theta);
    auto rxb0 =
        RXOp::create(rewriter, loc, rzz.getOutputQubit(0), constF(-HALF_PI));
    auto rxb1 =
        RXOp::create(rewriter, loc, rzz.getOutputQubit(1), constF(-HALF_PI));
    return {rxb0.getOutputQubit(0), rxb1.getOutputQubit(0)};
  };

  // `XXPlusYY(theta, beta)` and `XXMinusYY(theta, beta)` both act as
  //   Rz(-beta) on q0 -> entangling core -> Rz(+beta) on q0,
  // but differ in the entangling core:
  //   XXPlusYY:  exp(-i * theta/4 * (XX + YY))  == Ryy(theta/2) * Rxx(theta/2)
  //   XXMinusYY: exp(-i * theta/4 * (XX - YY))  == Rxx(theta/2) * Ryy(-theta/2)
  // (XX and YY commute, so the two multiplication orders produce identical
  // unitaries; the distinct order and sign below are what makes `XXMinusYY`
  // the "minus" variant and must be preserved even though an order flip
  // alone would also compile.)
  if (auto xxPlus = llvm::dyn_cast<XXPlusYYOp>(op)) {
    Value q0 = xxPlus.getInputQubit(0);
    Value q1 = xxPlus.getInputQubit(1);
    q0 = RZOp::create(rewriter, loc, q0, neg(xxPlus.getBeta()))
             .getOutputQubit(0);
    const auto halfTheta = half(xxPlus.getTheta());
    std::tie(q0, q1) = emitRyyViaRzz(q0, q1, halfTheta);
    std::tie(q0, q1) = emitRxxViaRzz(q0, q1, halfTheta);
    q0 = RZOp::create(rewriter, loc, q0, xxPlus.getBeta()).getOutputQubit(0);
    rewriter.replaceOp(op, ValueRange{q0, q1});
    return success();
  }
  if (auto xxMinus = llvm::dyn_cast<XXMinusYYOp>(op)) {
    Value q0 = xxMinus.getInputQubit(0);
    Value q1 = xxMinus.getInputQubit(1);
    q0 = RZOp::create(rewriter, loc, q0, neg(xxMinus.getBeta()))
             .getOutputQubit(0);
    const auto halfTheta = half(xxMinus.getTheta());
    std::tie(q0, q1) = emitRxxViaRzz(q0, q1, halfTheta);
    std::tie(q0, q1) = emitRyyViaRzz(q0, q1, neg(halfTheta));
    q0 = RZOp::create(rewriter, loc, q0, xxMinus.getBeta()).getOutputQubit(0);
    rewriter.replaceOp(op, ValueRange{q0, q1});
    return success();
  }
  return failure();
}

/// State for one maximal two-qubit window (plus absorbed one-qubit ops)
/// during consolidation.
struct TwoQubitBlock {
  Value wireA;
  Value wireB;
  llvm::SmallVector<Operation*, 8> ops;
  Matrix4x4 accum = Matrix4x4::identity();
  unsigned numTwoQ = 0;
  unsigned numOneQ = 0;
  bool anyNonNative = false;
  bool open = true;
};

/// Tracks overlapping two-qubit windows on a module slice.
struct TwoQubitWindowConsolidator {
  std::vector<TwoQubitBlock> blocks;
  llvm::DenseMap<Value, size_t> wireToBlock;

  void closeBlock(size_t idx);
  void closeBlockOnWire(Value v);
  void process(Operation* op, const NativeProfileSpec& spec);
  LogicalResult materialize(IRRewriter& rewriter,
                            const NativeProfileSpec& spec);
};

/// Check whether a two-qubit op `op` is already expressible by the resolved
/// native menu: a single-control `CX`/`CZ` consistent with the active
/// entangler, or `Rzz` when `spec.allowRzz` is set. Multi-control and other
/// two-qubit ops are considered non-native.
bool isNativeTwoQubitOp(Operation* op, const NativeProfileSpec& spec) {
  if (auto ctrl = llvm::dyn_cast<CtrlOp>(op)) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      return false;
    }
    auto* body = ctrl.getBodyUnitary(0).getOperation();
    if (llvm::isa<XOp>(body)) {
      return llvm::is_contained(spec.entanglerBases, EntanglerBasis::Cx);
    }
    if (llvm::isa<ZOp>(body)) {
      return llvm::is_contained(spec.entanglerBases, EntanglerBasis::Cz);
    }
    return false;
  }
  return spec.allowRzz && llvm::isa<RZZOp>(op);
}

/// Decide whether replacing a consolidated window is worthwhile. Always
/// replace a window that contains any non-native op (we have to lower them
/// anyway); otherwise only replace when the deterministic synthesizer uses
/// strictly fewer entanglers than the window already contains.
bool shouldApplyBlockReplacement(const TwoQubitBlock& block,
                                 std::uint8_t numBasisUses) {
  if (block.anyNonNative) {
    return true;
  }
  return numBasisUses < block.numTwoQ;
}

LogicalResult materializeSingleTwoQubitBlock(IRRewriter& rewriter,
                                             const TwoQubitBlock& block,
                                             const NativeProfileSpec& spec) {
  Operation* firstOp = block.ops.front();
  auto firstUnitary = llvm::cast<UnitaryOpInterface>(firstOp);
  const Value inA = firstUnitary.getInputQubit(0);
  const Value inB = firstUnitary.getInputQubit(1);
  const Value outA = block.wireA;
  const Value outB = block.wireB;

  rewriter.setInsertionPoint(firstOp);
  Value newA;
  Value newB;
  if (failed(synthesizeUnitary2QWeyl(rewriter, firstOp->getLoc(), inA, inB,
                                     block.accum, spec, newA, newB))) {
    firstOp->emitError("failed to emit synthesized two-qubit gate sequence");
    return failure();
  }
  rewriter.replaceAllUsesWith(outA, newA);
  rewriter.replaceAllUsesWith(outB, newB);
  for (auto* toErase : llvm::reverse(block.ops)) {
    rewriter.eraseOp(toErase);
  }
  return success();
}

void TwoQubitWindowConsolidator::closeBlock(size_t idx) {
  auto& block = blocks[idx];
  if (!block.open) {
    return;
  }
  block.open = false;
  wireToBlock.erase(block.wireA);
  if (block.wireB != block.wireA) {
    wireToBlock.erase(block.wireB);
  }
}

void TwoQubitWindowConsolidator::closeBlockOnWire(Value v) {
  if (auto it = wireToBlock.find(v); it != wireToBlock.end()) {
    closeBlock(it->second);
  }
}

void TwoQubitWindowConsolidator::process(Operation* op,
                                         const NativeProfileSpec& spec) {
  if (op->getParentOfType<CtrlOp>()) {
    return;
  }
  if (!llvm::isa<InvOp>(op) && op->getParentOfType<InvOp>()) {
    return;
  }
  auto unitary = llvm::dyn_cast<UnitaryOpInterface>(op);
  if (!unitary) {
    return;
  }
  if (llvm::isa<BarrierOp, GPhaseOp>(op)) {
    for (Value v : op->getOperands()) {
      closeBlockOnWire(v);
    }
    return;
  }

  if (unitary.isTwoQubit()) {
    Matrix4x4 opMatrix;
    if (!getBlockTwoQubitMatrix(op, opMatrix)) {
      closeBlockOnWire(unitary.getInputQubit(0));
      closeBlockOnWire(unitary.getInputQubit(1));
      return;
    }
    const Value v0 = unitary.getInputQubit(0);
    const Value v1 = unitary.getInputQubit(1);
    if (v0 == v1) {
      closeBlockOnWire(v0);
      return;
    }
    auto it0 = wireToBlock.find(v0);
    auto it1 = wireToBlock.find(v1);
    const bool tracked0 = it0 != wireToBlock.end();
    const bool tracked1 = it1 != wireToBlock.end();
    const std::optional<size_t> idx0 =
        tracked0 ? std::optional(it0->second) : std::nullopt;
    const std::optional<size_t> idx1 =
        tracked1 ? std::optional(it1->second) : std::nullopt;
    const bool sameBlock =
        idx0.has_value() && idx1.has_value() && *idx0 == *idx1;
    const bool singleUse = v0.hasOneUse() && v1.hasOneUse();

    if (sameBlock && singleUse) {
      const size_t idx = *idx0;
      auto& block = blocks[idx];
      llvm::SmallVector<QubitId, 2> ids;
      if (v0 == block.wireA && v1 == block.wireB) {
        ids = {0, 1};
      } else if (v0 == block.wireB && v1 == block.wireA) {
        ids = {1, 0};
      } else {
        closeBlock(idx);
        return;
      }
      block.accum = fixTwoQubitMatrixQubitOrder(opMatrix, ids) * block.accum;
      block.ops.push_back(op);
      ++block.numTwoQ;
      if (!isNativeTwoQubitOp(op, spec)) {
        block.anyNonNative = true;
      }
      const Value eraseKeyA = it0->first;
      const Value eraseKeyB = it1->first;
      wireToBlock.erase(eraseKeyA);
      if (eraseKeyA != eraseKeyB) {
        wireToBlock.erase(eraseKeyB);
      }
      Value newA;
      Value newB;
      if (v0 == block.wireA) {
        newA = unitary.getOutputQubit(0);
        newB = unitary.getOutputQubit(1);
      } else {
        newA = unitary.getOutputQubit(1);
        newB = unitary.getOutputQubit(0);
      }
      block.wireA = newA;
      block.wireB = newB;
      wireToBlock[newA] = idx;
      wireToBlock[newB] = idx;
      return;
    }

    if (idx0.has_value()) {
      closeBlock(*idx0);
    }
    if (idx1.has_value() && (!idx0.has_value() || *idx0 != *idx1)) {
      closeBlock(*idx1);
    }
    TwoQubitBlock nb;
    nb.wireA = unitary.getOutputQubit(0);
    nb.wireB = unitary.getOutputQubit(1);
    nb.ops.push_back(op);
    nb.numTwoQ = 1;
    nb.accum = opMatrix;
    nb.anyNonNative = !isNativeTwoQubitOp(op, spec);
    const size_t idx = blocks.size();
    blocks.push_back(std::move(nb));
    wireToBlock[blocks[idx].wireA] = idx;
    wireToBlock[blocks[idx].wireB] = idx;
    return;
  }

  if (unitary.isSingleQubit()) {
    const Value v = unitary.getInputQubit(0);
    auto it = wireToBlock.find(v);
    if (it == wireToBlock.end()) {
      return;
    }
    const size_t idx = it->second;
    auto& block = blocks[idx];
    Matrix2x2 raw;
    if (!unitary.getUnitaryMatrix2x2(raw) || !v.hasOneUse()) {
      closeBlock(idx);
      return;
    }
    const auto pad = (v == block.wireA) ? expandToTwoQubits(raw, 0)
                                        : expandToTwoQubits(raw, 1);
    block.accum = pad * block.accum;
    block.ops.push_back(op);
    ++block.numOneQ;
    if (!allowsSingleQubitOp(unitary, spec)) {
      block.anyNonNative = true;
    }
    wireToBlock.erase(it);
    if (v == block.wireA) {
      block.wireA = unitary.getOutputQubit(0);
      wireToBlock[block.wireA] = idx;
    } else {
      block.wireB = unitary.getOutputQubit(0);
      wireToBlock[block.wireB] = idx;
    }
    return;
  }

  for (Value v : op->getOperands()) {
    closeBlockOnWire(v);
  }
}

LogicalResult
TwoQubitWindowConsolidator::materialize(IRRewriter& rewriter,
                                        const NativeProfileSpec& spec) {
  llvm::DenseSet<Operation*> erasedOps;
  for (const auto& block : blocks) {
    if (block.ops.size() < 2) {
      continue;
    }
    if (llvm::any_of(block.ops,
                     [&](Operation* op) { return erasedOps.contains(op); })) {
      continue;
    }
    const auto numBasisUses = twoQubitEntanglerCount(block.accum, spec);
    if (!numBasisUses) {
      continue;
    }
    if (!shouldApplyBlockReplacement(block, *numBasisUses)) {
      continue;
    }
    if (failed(materializeSingleTwoQubitBlock(rewriter, block, spec))) {
      return failure();
    }
    for (Operation* op : block.ops) {
      erasedOps.insert(op);
    }
  }
  return success();
}

LogicalResult fuseTwoQubitUnitaryRuns(IRRewriter& rewriter, Operation* root,
                                      const NativeProfileSpec& spec) {
  llvm::SmallVector<Operation*, 32> ops;
  collectUnitaryOpsInPreOrder(root, ops);
  TwoQubitWindowConsolidator consolidator;
  for (Operation* op : ops) {
    consolidator.process(op, spec);
  }
  return consolidator.materialize(rewriter, spec);
}

/// Adjacent single-qubit unitaries on one wire considered for fusion.
struct OneQubitRun {
  llvm::SmallVector<UnitaryOpInterface, 4> ops;
};

/// If profitable, replace the run with one synthesized single-qubit op in
/// `basis` (mirrors `FuseSingleQubitUnitaryRuns`). Fuses when any op is
/// off-menu or when Euler resynthesis strictly shortens the run.
bool maybeFuseRun(IRRewriter& rewriter, OneQubitRun& run,
                  const decomposition::EulerBasis basis,
                  const NativeProfileSpec& spec) {
  Matrix2x2 fused = Matrix2x2::identity();
  for (UnitaryOpInterface u : run.ops) {
    Matrix2x2 m;
    if (!u.getUnitaryMatrix2x2(m)) {
      return false;
    }
    fused.premultiplyBy(m);
  }

  const bool anyNonNative = llvm::any_of(run.ops, [&](UnitaryOpInterface u) {
    return !allowsSingleQubitOp(u, spec);
  });

  Operation* firstOp = run.ops.front().getOperation();
  const Value inQubit = run.ops.front().getInputQubit(0);
  const Value outQubit = run.ops.back().getOutputQubit(0);

  rewriter.setInsertionPoint(firstOp);
  const auto replacement = decomposition::synthesizeUnitary1QEuler(
      rewriter, firstOp->getLoc(), inQubit, fused, run.ops.size(), anyNonNative,
      basis);
  if (!replacement) {
    return false;
  }
  rewriter.replaceAllUsesWith(outQubit, *replacement);
  for (auto& op : llvm::reverse(run.ops)) {
    rewriter.eraseOp(op.getOperation());
  }
  return true;
}

/// True when `op` lives in a `ctrl`/`inv` region body (not the shell op).
/// Skips nested unitaries so they are handled via the enclosing modifier.
bool isHiddenInsideCtrlOrInvBody(Operation* op) {
  if (op->getParentOfType<CtrlOp>()) {
    return true;
  }
  if (!llvm::isa<InvOp>(op) && op->getParentOfType<InvOp>()) {
    return true;
  }
  return false;
}

/// Single-qubit op eligible for fusion (constant `2×2`, not under `ctrl`).
UnitaryOpInterface fusibleSingleQubitOp(Operation* op) {
  auto unitary = llvm::dyn_cast<UnitaryOpInterface>(op);
  if (!unitary || !unitary.isSingleQubit()) {
    return {};
  }
  if (llvm::isa<BarrierOp, GPhaseOp, CtrlOp>(op)) {
    return {};
  }
  if (isHiddenInsideCtrlOrInvBody(op)) {
    return {};
  }
  Matrix2x2 matrix;
  if (!unitary.getUnitaryMatrix2x2(matrix)) {
    return {};
  }
  return unitary;
}

/// Whether `emitter` can lower the single-qubit `op` directly (used for ops
/// with non-constant angles, which have no constant `2×2` matrix).
bool emitterHasDirectLowering(Operation* op,
                              const SingleQubitEmitterSpec& emitter) {
  switch (emitter.mode) {
  case SingleQubitMode::ZSXX:
    return canDirectlyDecomposeToZSXX(op, emitter.supportsDirectRx);
  case SingleQubitMode::U3:
    return canDirectlyDecomposeToU3(op);
  case SingleQubitMode::R:
    return canDirectlyDecomposeToR(op);
  case SingleQubitMode::AxisPair:
    return canDirectlyDecomposeToAxisPair(op, emitter.axisPair);
  }
  return false;
}

/// Dispatch `op`'s direct (non-matrix) single-qubit lowering to the
/// `decomposeTo*` helper for `emitter.mode`. Returns the output qubit value
/// or a null `Value` if no direct rule applies for this op.
Value applyDirectSingleQubitLowering(IRRewriter& rewriter, Operation* op,
                                     Value in,
                                     const SingleQubitEmitterSpec& emitter) {
  switch (emitter.mode) {
  case SingleQubitMode::ZSXX:
    return decomposeToZSXX(rewriter, op, in, emitter.supportsDirectRx);
  case SingleQubitMode::U3:
    return decomposeToU3(rewriter, op, in);
  case SingleQubitMode::R:
    return decomposeToR(rewriter, op, in);
  case SingleQubitMode::AxisPair:
    return decomposeToAxisPair(rewriter, op, in, emitter.axisPair);
  }
  llvm_unreachable("unknown SingleQubitMode");
}

/// Lowers unitary QCO ops to a comma-separated native gate menu using a
/// deterministic, matrix-driven synthesizer: single-qubit fuse, two-qubit
/// window consolidation, synthesis sweeps, seam single-qubit fuse, and
/// optional cleanup sweeps.
struct FuseTwoQubitUnitaryRunsPass
    : impl::FuseTwoQubitUnitaryRunsBase<FuseTwoQubitUnitaryRunsPass> {
  /// Default-construct the pass with the TableGen-generated option defaults.
  FuseTwoQubitUnitaryRunsPass() = default;

  explicit FuseTwoQubitUnitaryRunsPass(FuseTwoQubitUnitaryRunsOptions options)
      : FuseTwoQubitUnitaryRunsBase(std::move(options)) {}

protected:
  /// Top-level pass entry point. Resolves the native-gate menu, then drives
  /// the staged rewrite pipeline: one-qubit run fusion, two-qubit window
  /// consolidation, synthesis sweeps until the single-qubit surface is native,
  /// seam cleanup, and a final fusion pass. Fails the pass on invalid input or
  /// non-convergence.
  void runOnOperation() override {
    // Empty native-gates string: no-op.
    if (llvm::StringRef(nativeGates).trim().empty()) {
      return;
    }
    auto specOpt = parseNativeSpec(nativeGates);
    if (!specOpt) {
      getOperation().emitError()
          << "unsupported native gate menu (native-gates='" << nativeGates
          << "')";
      signalPassFailure();
      return;
    }
    const auto& spec = *specOpt;
    // Deterministic single-qubit basis: the first emitter drives all matrix
    // synthesis and run fusion.
    const decomposition::EulerBasis oneQubitBasis =
        emitterEulerBasis(spec.singleQubitEmitters.front());

    IRRewriter rewriter(&getContext());

    fuseOneQubitRuns(rewriter, spec, oneQubitBasis);
    if (failed(consolidateTwoQubitBlocks(rewriter, spec))) {
      signalPassFailure();
      return;
    }
    // Two-qubit lowering can emit off-menu single-qubit ops (e.g. `rx`/`ry`);
    // repeat until clean or hit the sweep cap before seam cleanup.
    constexpr unsigned kMaxSynthesisSweeps = 4;
    for (unsigned i = 0; i < kMaxSynthesisSweeps; ++i) {
      if (failed(synthesizeRemainingOps(rewriter, spec, oneQubitBasis))) {
        signalPassFailure();
        return;
      }
      if (!hasNonNativeSingleQubitOps(spec)) {
        break;
      }
    }
    if (hasNonNativeSingleQubitOps(spec)) {
      getOperation().emitError()
          << "native gate synthesis did not converge within "
          << kMaxSynthesisSweeps
          << " sweeps (single-qubit ops remain outside the native menu)";
      signalPassFailure();
      return;
    }
    // Fuse single-qubit seams between two-qubit blocks.
    fuseOneQubitRuns(rewriter, spec, oneQubitBasis);
    // Re-check full menu (single-qubit ops, native `ctrl`, allowed bare `rzz`).
    constexpr unsigned kPostMenuCleanupSweeps = 4;
    unsigned postMenuSweepsRemaining = kPostMenuCleanupSweeps;
    while (hasNonNativeMenuOps(spec) && postMenuSweepsRemaining-- > 0) {
      if (failed(synthesizeRemainingOps(rewriter, spec, oneQubitBasis))) {
        signalPassFailure();
        return;
      }
      fuseOneQubitRuns(rewriter, spec, oneQubitBasis);
    }
    if (hasNonNativeMenuOps(spec)) {
      getOperation().emitError()
          << "native gate synthesis: operations remain outside the native menu "
             "after final cleanup";
      signalPassFailure();
      return;
    }
  }

  /// `CtrlOp` is already on-menu when the body is `X`/`Z` and the profile
  /// supplies `cx` / `cz` entanglers.
  static bool ctrlMatchesNativeMenu(CtrlOp ctrl,
                                    const NativeProfileSpec& spec) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      return false;
    }
    Operation* body = ctrl.getBodyUnitary(0).getOperation();
    const bool hasCX = llvm::isa<XOp>(body);
    const bool hasCZ = llvm::isa<ZOp>(body);
    if (!hasCX && !hasCZ) {
      return false;
    }
    return (usesCxEntangler(spec) && hasCX) || (usesCzEntangler(spec) && hasCZ);
  }

  /// Bare two-qubit on-menu: `rzz` when the profile allows it.
  static bool bareTwoQubitMatchesNativeMenu(Operation* op,
                                            const NativeProfileSpec& spec) {
    return llvm::isa<RZZOp>(op) && spec.allowRzz &&
           spec.allowedGates.contains(NativeGateKind::Rzz);
  }

  /// True if any unitary is outside `spec` (single-qubit, `ctrl`, or bare
  /// `rzz`).
  bool hasNonNativeMenuOps(const NativeProfileSpec& spec) {
    const mlir::WalkResult walkResult =
        getOperation()->walk([&](Operation* op) {
          if (llvm::isa<BarrierOp, GPhaseOp>(op)) {
            return mlir::WalkResult::advance();
          }
          if (isHiddenInsideCtrlOrInvBody(op)) {
            return mlir::WalkResult::advance();
          }
          if (auto ctrl = llvm::dyn_cast<CtrlOp>(op)) {
            if (!ctrlMatchesNativeMenu(ctrl, spec)) {
              return mlir::WalkResult::interrupt();
            }
            return mlir::WalkResult::advance();
          }
          auto unitary = llvm::dyn_cast<UnitaryOpInterface>(op);
          if (!unitary) {
            return mlir::WalkResult::advance();
          }
          if (unitary.isSingleQubit()) {
            if (!allowsSingleQubitOp(unitary, spec)) {
              return mlir::WalkResult::interrupt();
            }
            return mlir::WalkResult::advance();
          }
          if (unitary.isTwoQubit()) {
            if (!bareTwoQubitMatchesNativeMenu(op, spec)) {
              return mlir::WalkResult::interrupt();
            }
            return mlir::WalkResult::advance();
          }
          return mlir::WalkResult::interrupt();
        });
    return walkResult.wasInterrupted();
  }

  /// Any off-menu single-qubit unitary (ignores `ctrl` region bodies).
  bool hasNonNativeSingleQubitOps(const NativeProfileSpec& spec) {
    const mlir::WalkResult walkResult =
        getOperation()->walk([&](Operation* op) {
          if (llvm::isa<BarrierOp, GPhaseOp>(op)) {
            return mlir::WalkResult::advance();
          }
          if (isHiddenInsideCtrlOrInvBody(op)) {
            return mlir::WalkResult::advance();
          }
          auto unitary = llvm::dyn_cast<UnitaryOpInterface>(op);
          if (!unitary || !unitary.isSingleQubit()) {
            return mlir::WalkResult::advance();
          }
          if (!allowsSingleQubitOp(unitary, spec)) {
            return mlir::WalkResult::interrupt();
          }
          return mlir::WalkResult::advance();
        });
    return walkResult.wasInterrupted();
  }

private:
  /// Fuse adjacent single-qubit runs when the emitter wins on length or any op
  /// is off-menu.
  void fuseOneQubitRuns(IRRewriter& rewriter, const NativeProfileSpec& spec,
                        const decomposition::EulerBasis basis) {
    llvm::SmallVector<OneQubitRun> runs;
    llvm::DenseMap<Operation*, size_t> tailOpToRun;

    // Extend the current run only when this op consumes the run's *tail*
    // output with no other uses: both the `tailOpToRun` lookup and
    // `inQubit.hasOneUse()` are required. Without the single-use check a run
    // could fuse gates on a wire that also feeds another path (fan-out),
    // which would silently drop the sibling user.
    getOperation()->walk([&](Operation* op) {
      auto unitary = fusibleSingleQubitOp(op);
      if (!unitary) {
        return;
      }
      Value inQubit = unitary.getInputQubit(0);
      Operation* defOp = inQubit.getDefiningOp();
      auto it =
          (defOp != nullptr) ? tailOpToRun.find(defOp) : tailOpToRun.end();
      const bool canExtend = it != tailOpToRun.end() && inQubit.hasOneUse();
      if (canExtend) {
        const size_t runIdx = it->second;
        runs[runIdx].ops.push_back(unitary);
        tailOpToRun.erase(it);
        tailOpToRun[op] = runIdx;
      } else {
        runs.push_back(OneQubitRun{});
        runs.back().ops.push_back(unitary);
        tailOpToRun[op] = runs.size() - 1;
      }
    });

    for (auto& run : runs) {
      if (run.ops.size() < 2) {
        continue;
      }
      (void)maybeFuseRun(rewriter, run, basis, spec);
    }
  }

  /// Two-qubit windows with absorbed single-qubit ops: replace when a cheaper
  /// native sequence exists.
  LogicalResult consolidateTwoQubitBlocks(IRRewriter& rewriter,
                                          const NativeProfileSpec& spec) {
    return fuseTwoQubitUnitaryRuns(rewriter, getOperation(), spec);
  }

  /// One synthesis sweep over the whole function: rewrite every remaining
  /// off-menu unitary by dispatching to `rewriteSingleQubit` /
  /// `rewriteControlled` / `rewriteTwoQubit`. Returns `failure()` as soon as
  /// any op cannot be lowered to the native menu. Safe to call repeatedly;
  /// `runOnOperation` iterates until convergence.
  LogicalResult synthesizeRemainingOps(IRRewriter& rewriter,
                                       const NativeProfileSpec& spec,
                                       const decomposition::EulerBasis basis) {
    llvm::SmallVector<Operation*, 32> ops;
    collectUnitaryOpsInPreOrder(getOperation(), ops);
    llvm::DenseSet<Operation*> erasedOps;

    for (Operation* op : ops) {
      // Pointers were collected before this loop; avoid dereferencing ops
      // erased by earlier rewrites in this same sweep.
      if (erasedOps.contains(op)) {
        continue;
      }
      // Nested regions under `ctrl` / `inv` are handled on the shell op
      // (e.g. `ctrl { inv { ... } }`, `inv { ... }`).
      if (isHiddenInsideCtrlOrInvBody(op)) {
        continue;
      }
      if (llvm::isa<BarrierOp, GPhaseOp>(op)) {
        continue;
      }
      auto unitary = llvm::dyn_cast<UnitaryOpInterface>(op);
      if (!unitary) {
        continue;
      }

      if (unitary.isSingleQubit()) {
        if (!allowsSingleQubitOp(unitary, spec)) {
          if (failed(rewriteSingleQubit(rewriter, op, unitary, spec, basis))) {
            return failure();
          }
          erasedOps.insert(op);
        }
        continue;
      }

      if (auto ctrl = llvm::dyn_cast<CtrlOp>(op)) {
        const bool wasAlreadyNative = ctrlMatchesNativeMenu(ctrl, spec);
        if (failed(rewriteControlled(rewriter, ctrl, spec))) {
          return failure();
        }
        if (!wasAlreadyNative) {
          erasedOps.insert(op);
        }
        continue;
      }

      if (unitary.isTwoQubit()) {
        if (failed(rewriteTwoQubit(rewriter, op, unitary, spec))) {
          return failure();
        }
        erasedOps.insert(op);
        continue;
      }
    }
    return success();
  }

  /// Lower one off-menu single-qubit `op`. Constant unitaries use the
  /// matrix-driven Euler synthesizer in `basis`; ops with non-constant angles
  /// fall back to the symbolic `decomposeTo*` lowering of the first emitter
  /// that handles them.
  static LogicalResult
  rewriteSingleQubit(IRRewriter& rewriter, Operation* op,
                     UnitaryOpInterface unitary, const NativeProfileSpec& spec,
                     const decomposition::EulerBasis basis) {
    rewriter.setInsertionPoint(op);
    const Value in = unitary.getInputQubit(0);
    Matrix2x2 matrix;
    if (unitary.isSingleQubit() && unitary.getUnitaryMatrix2x2(matrix)) {
      const Value replaced =
          emitSingleQubitMatrix(rewriter, op->getLoc(), in, matrix, basis);
      rewriter.replaceOp(op, replaced);
      return success();
    }
    for (const auto& emitter : spec.singleQubitEmitters) {
      if (!emitterHasDirectLowering(op, emitter)) {
        continue;
      }
      if (const Value replaced =
              applyDirectSingleQubitLowering(rewriter, op, in, emitter)) {
        rewriter.replaceOp(op, replaced);
        return success();
      }
    }
    op->emitError("single-qubit operation not in selected native profile");
    return failure();
  }

  /// Lower a single-control, single-target `CtrlOp` to the native profile.
  /// Fast-path: already-native `CX`/`CZ` are kept as-is. Otherwise, lift the
  /// controlled op to its 4x4 matrix and run the deterministic two-qubit
  /// synthesizer.
  static LogicalResult rewriteControlled(IRRewriter& rewriter, CtrlOp ctrl,
                                         const NativeProfileSpec& spec) {
    if (ctrl.getNumControls() != 1 || ctrl.getNumTargets() != 1) {
      ctrl.emitError("native synthesis currently only supports 1-control "
                     "1-target controlled gates");
      return failure();
    }
    auto* body = ctrl.getBodyUnitary(0).getOperation();
    const bool hasCX = llvm::isa<XOp>(body);
    const bool hasCZ = llvm::isa<ZOp>(body);
    if ((usesCxEntangler(spec) && hasCX) || (usesCzEntangler(spec) && hasCZ)) {
      return success();
    }
    Matrix4x4 matrix;
    if (hasCX || hasCZ) {
      if (!getBlockTwoQubitMatrix(ctrl.getOperation(), matrix)) {
        ctrl.emitError("failed to compute 4x4 matrix for CtrlOp");
        return failure();
      }
    } else {
      auto u = llvm::cast<UnitaryOpInterface>(ctrl.getOperation());
      if (!u.isTwoQubit() || !u.getUnitaryMatrix4x4(matrix)) {
        ctrl.emitError(
            "native synthesis: cannot build a constant 4x4 matrix for this "
            "controlled gate (unsupported body or non-constant parameters)");
        return failure();
      }
    }
    rewriter.setInsertionPoint(ctrl);
    Value out0;
    Value out1;
    if (failed(synthesizeUnitary2QWeyl(
            rewriter, ctrl.getLoc(), ctrl.getInputControl(0),
            ctrl.getInputTarget(0), matrix, spec, out0, out1))) {
      ctrl.emitError("controlled gate not allowed by selected profile");
      return failure();
    }
    rewriter.replaceOp(ctrl, ValueRange{out0, out1});
    return success();
  }

  /// Lower an off-menu generic two-qubit op (`RZZ`, `XXPlusYY`, `XXMinusYY`,
  /// or any arbitrary 4x4 unitary). Handles the `Rzz`-native fast path; for
  /// `XXPlusYY` / `XXMinusYY` with `rzz` on the menu, uses the dedicated
  /// `XX±YY -> Rzz` rewrite. All other two-qubit unitaries go through the
  /// deterministic KAK synthesizer.
  static LogicalResult rewriteTwoQubit(IRRewriter& rewriter, Operation* op,
                                       UnitaryOpInterface unitary,
                                       const NativeProfileSpec& spec) {
    if (spec.allowRzz && llvm::isa<RZZOp>(op)) {
      return success();
    }
    if (spec.allowRzz &&
        (llvm::isa<XXPlusYYOp>(op) || llvm::isa<XXMinusYYOp>(op))) {
      rewriter.setInsertionPoint(op);
      if (succeeded(rewriteXXPlusMinusYYViaRzz(rewriter, op))) {
        return success();
      }
      // Fall through to entangler-based synthesis when the dedicated rewrite
      // could not be applied (e.g. no entangler-free realization).
    }
    Matrix4x4 matrix;
    if (!getBlockTwoQubitMatrix(op, matrix)) {
      op->emitError("unsupported two-qubit operation for selected profile");
      return failure();
    }
    rewriter.setInsertionPoint(op);
    Value out0;
    Value out1;
    if (failed(synthesizeUnitary2QWeyl(
            rewriter, op->getLoc(), unitary.getInputQubit(0),
            unitary.getInputQubit(1), matrix, spec, out0, out1))) {
      op->emitError("unsupported two-qubit operation for selected profile");
      return failure();
    }
    rewriter.replaceOp(op, ValueRange{out0, out1});
    return success();
  }
};

} // namespace
} // namespace mlir::qco
