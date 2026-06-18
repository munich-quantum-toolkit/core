/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/TwoQubit.h"

#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/BasisDecomposer.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/Gate.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/GateKind.h"
#include "mlir/Dialect/QCO/Transforms/Decomposition/WeylDecomposition.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/NativeSpec.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Policy.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/SingleQubit.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Types.h"
#include "mlir/Dialect/QCO/Transforms/NativeSynthesis/Utils.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

#include <cstddef>
#include <cstdint>
#include <numbers>
#include <optional>
#include <utility>

namespace mlir::qco::native_synth {

constexpr double PI = std::numbers::pi;
constexpr double HALF_PI = PI / 2.0;

/// Deterministic entangler choice: prefer CX over CZ. Returns `std::nullopt`
/// when the menu has no entangler basis.
static std::optional<EntanglerBasis>
selectEntangler(const NativeProfileSpec& spec) {
  if (usesCxEntangler(spec)) {
    return EntanglerBasis::Cx;
  }
  if (usesCzEntangler(spec)) {
    return EntanglerBasis::Cz;
  }
  return std::nullopt;
}

/// Build the decomposition-layer basis gate for `entangler`. The qubit ids
/// align with `getBlockTwoQubitMatrix` / CX layout (control on qubit 0).
static decomposition::Gate entanglerGate(EntanglerBasis entangler) {
  return decomposition::Gate{
      .type = entangler == EntanglerBasis::Cz ? decomposition::GateKind::Z
                                              : decomposition::GateKind::X,
      .qubitId = {0, 1},
  };
}

/// Run the Weyl + basis decomposer for `target` against `entangler`, returning
/// the raw single-qubit factors and entangler count (or `std::nullopt`).
static std::optional<decomposition::TwoQubitNativeDecomposition>
decomposeWithEntangler(const Matrix4x4& target, EntanglerBasis entangler) {
  const auto basisGate = entanglerGate(entangler);
  auto decomposer =
      decomposition::TwoQubitBasisDecomposer::create(basisGate, 1.0);
  auto weyl =
      decomposition::TwoQubitWeylDecomposition::create(target, std::nullopt);
  return decomposer.twoQubitDecompose(weyl, std::nullopt);
}

std::optional<std::uint8_t>
twoQubitEntanglerCount(const Matrix4x4& target, const NativeProfileSpec& spec) {
  const auto entangler = selectEntangler(spec);
  if (!entangler) {
    return std::nullopt;
  }
  const auto native = decomposeWithEntangler(target, *entangler);
  if (!native) {
    return std::nullopt;
  }
  return native->numBasisUses;
}

LogicalResult emitTwoQubitNative(IRRewriter& rewriter, Location loc,
                                 Value qubit0, Value qubit1,
                                 const Matrix4x4& target,
                                 const NativeProfileSpec& spec,
                                 Value& outQubit0, Value& outQubit1) {
  const auto entangler = selectEntangler(spec);
  if (!entangler) {
    return failure();
  }
  const auto native = decomposeWithEntangler(target, *entangler);
  if (!native) {
    return failure();
  }
  const auto basis = emitterEulerBasis(spec.singleQubitEmitters.front());

  // Residual global phase not represented by the factors / entanglers.
  emitGPhaseIfNonTrivial(rewriter, loc, native->globalPhase);

  Value wire0 = qubit0;
  Value wire1 = qubit1;
  const auto& factors = native->singleQubitFactors;
  const std::uint8_t numBasisUses = native->numBasisUses;
  const auto emitFactor = [&](Value& wire, std::size_t index) {
    wire = emitSingleQubitMatrix(rewriter, loc, wire, factors[index], basis);
  };
  const auto emitEntangler = [&]() {
    // The entangler acts with its control on wire 0 and target on wire 1.
    auto ctrlOp = CtrlOp::create(
        rewriter, loc, ValueRange{wire0}, ValueRange{wire1},
        [&](ValueRange targetArgs) -> llvm::SmallVector<Value> {
          if (*entangler == EntanglerBasis::Cz) {
            return {
                ZOp::create(rewriter, loc, targetArgs[0]).getOutputQubit(0)};
          }
          return {XOp::create(rewriter, loc, targetArgs[0]).getOutputQubit(0)};
        });
    wire0 = ctrlOp.getOutputControl(0);
    wire1 = ctrlOp.getOutputTarget(0);
  };

  // factor[2i] on wire 1, factor[2i + 1] on wire 0, then one entangler.
  for (std::uint8_t i = 0; i < numBasisUses; ++i) {
    emitFactor(wire1, static_cast<std::size_t>(2 * i));
    emitFactor(wire0, static_cast<std::size_t>((2 * i) + 1));
    emitEntangler();
  }
  emitFactor(wire1, static_cast<std::size_t>(2 * numBasisUses));
  emitFactor(wire0, static_cast<std::size_t>((2 * numBasisUses) + 1));

  outQubit0 = wire0;
  outQubit1 = wire1;
  return success();
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

} // namespace mlir::qco::native_synth
