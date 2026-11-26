/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mqt/Dialect/MQT/Utils/Angles.h"
#include "mqt/Dialect/MQT/Utils/ConstantFolding.h"
#include "mqt/Dialect/MQT/Utils/GatePowering.h"
#include "mqt/Dialect/MQT/Utils/Modifiers.h"
#include "mqt/Dialect/MQT/Utils/Parameters.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include <cmath>
#include <numbers>
#include <optional>

namespace mlir::mqt {

/// Fold a power modifier around one named gate into named gates.
///
/// - Fixed one-qubit gates: any constant or runtime exponent, using their exact
///   period and rotation axis
/// - Rotation and phase gates: scale a constant angle when the exponent is
///   integral or the gate's eigenphases lie in (−π, π)
/// - U gates: positive integral exponents
/// - iSWAP: `xx_plus_yy(−r·π, 0)`
/// - Identity and barrier: unchanged
///
/// Exponents equivalent to zero or one are left to `foldTrivialPower`.
/// `Ops` names the power modifier, unitary interface, and gates of the QC or
/// QCO dialect. QC gates have no results, so the same code serves both
/// dialects: a QC replacement simply erases the modifier.
template <typename Ops>
struct FoldPowIntoGate final : OpRewritePattern<typename Ops::PowOp> {
  using PowOp = typename Ops::PowOp;
  using UnitaryOpInterface = typename Ops::UnitaryOpInterface;
  using GPhaseOp = typename Ops::GPhaseOp;
  using IdOp = typename Ops::IdOp;
  using BarrierOp = typename Ops::BarrierOp;
  using XOp = typename Ops::XOp;
  using YOp = typename Ops::YOp;
  using ZOp = typename Ops::ZOp;
  using HOp = typename Ops::HOp;
  using SOp = typename Ops::SOp;
  using SdgOp = typename Ops::SdgOp;
  using TOp = typename Ops::TOp;
  using TdgOp = typename Ops::TdgOp;
  using SXOp = typename Ops::SXOp;
  using SXdgOp = typename Ops::SXdgOp;
  using RXOp = typename Ops::RXOp;
  using RYOp = typename Ops::RYOp;
  using RZOp = typename Ops::RZOp;
  using POp = typename Ops::POp;
  using ROp = typename Ops::ROp;
  using UOp = typename Ops::UOp;
  using RXXOp = typename Ops::RXXOp;
  using RYYOp = typename Ops::RYYOp;
  using RZXOp = typename Ops::RZXOp;
  using RZZOp = typename Ops::RZZOp;
  using XXPlusYYOp = typename Ops::XXPlusYYOp;
  using XXMinusYYOp = typename Ops::XXMinusYYOp;
  using iSWAPOp = typename Ops::iSWAPOp;

  using OpRewritePattern<PowOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    auto gate = getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!gate) {
      return failure();
    }
    auto* inner = gate.getOperation();
    std::optional<double> exponent = op.getExponentValue();
    const auto period = getFixedGatePowerPeriod(gate.getBaseSymbol());
    if (exponent && period != 0U) {
      exponent = std::remainder(*exponent, static_cast<double>(period));
      if (*exponent == 0. || *exponent == 1. ||
          (period == 2U && *exponent == -1.)) {
        return failure();
      }
    }

    if (isa<XOp, YOp, ZOp, HOp, SOp, SdgOp, TOp, TdgOp, SXOp, SXdgOp>(inner)) {
      return foldFixedGate(op, inner, exponent, period, rewriter);
    }
    // pow(r) { id } → id and pow(r) { barrier } → barrier
    if (isa<IdOp, BarrierOp>(inner)) {
      inlineModifierBody(op, *op.getBody(), qubits(op), rewriter);
      return success();
    }
    if (!exponent) {
      return failure();
    }
    const double r = *exponent;

    // pow(r) { iswap } → xx_plus_yy(−r·π, 0): β = 0 aligns the axis with XX.
    if (isa<iSWAPOp>(inner)) {
      const auto targets = inputs(op, inner);
      return replace(op, inner,
                     XXPlusYYOp::create(rewriter, op.getLoc(), targets[0],
                                        targets[1], -r * std::numbers::pi, 0.),
                     rewriter);
    }

    // pow(n) { u(θ, φ, λ) } → gphase(δ); u(θ', φ', λ')
    if (isa<UOp>(inner)) {
      const auto theta = valueToDouble(gate.getParameter(0));
      const auto phi = valueToDouble(gate.getParameter(1));
      const auto lambda = valueToDouble(gate.getParameter(2));
      if (!theta || !phi || !lambda) {
        return failure();
      }
      const auto power = powerUParameters(*theta, *phi, *lambda, r);
      if (!power) {
        return failure();
      }
      if (std::abs(normalizeAngle(power->phase)) >
          PARAMETER_COMPARISON_TOLERANCE) {
        GPhaseOp::create(rewriter, op.getLoc(), power->phase);
      }
      return inlineWithParameters(
          op, gate, {power->theta, power->phi, power->lambda}, rewriter);
    }

    // pow(r) { g(θ, …) } → g(r·θ, …)
    if (!isa<GPhaseOp, RXOp, RYOp, RZOp, POp, ROp, RXXOp, RYYOp, RZXOp, RZZOp,
             XXPlusYYOp, XXMinusYYOp>(inner)) {
      return failure();
    }
    const auto parameter = valueToDouble(gate.getParameter(0));
    // Phase gates have eigenphase θ. Rotations have eigenphases ±θ/2.
    const double limit =
        isa<GPhaseOp, POp>(inner) ? std::numbers::pi : 2. * std::numbers::pi;
    if (!parameter ||
        (std::abs(*parameter) >= limit && !isIntegerExponent(r))) {
      return failure();
    }
    const auto scaled = scaleConstantAngle(*parameter, r);
    if (!scaled ||
        (isa<GPhaseOp>(inner) && !isValidGlobalPhaseAngle(*scaled))) {
      return failure();
    }
    return inlineWithParameters(op, gate, *scaled, rewriter);
  }

private:
  /// Return the qubit operands of @p op; the exponent precedes them.
  static OperandRange qubits(PowOp op) {
    return op->getOperands().drop_front();
  }

  /// Resolve the qubit operands of a parameterless body gate to @p op's.
  static SmallVector<Value, 2> inputs(PowOp op, Operation* gate) {
    return llvm::map_to_vector<2>(gate->getOperands(), [&](Value qubit) {
      return getValueFromBlockArgument(qubit, qubits(op));
    });
  }

  /// Replace @p op by @p replacement of its body gate @p gate.
  static LogicalResult replace(PowOp op, Operation* gate,
                               Operation* replacement,
                               PatternRewriter& rewriter) {
    IRMapping mapping;
    mapping.map(op.getBody()->getArguments(), qubits(op));
    mapping.map(gate->getResults(), replacement->getResults());
    rewriter.replaceOp(
        op, llvm::map_to_vector(
                op.getBody()->getTerminator()->getOperands(),
                [&](Value yielded) { return mapping.lookup(yielded); }));
    return success();
  }

  /// Replace the leading parameters of @p gate and inline it.
  static LogicalResult inlineWithParameters(PowOp op, UnitaryOpInterface gate,
                                            ArrayRef<double> values,
                                            PatternRewriter& rewriter) {
    const auto parameters = llvm::map_to_vector(values, [&](double value) {
      return constantFromScalar(rewriter, op.getLoc(), value);
    });
    const auto first = gate.getParameters().getBeginOperandIndex();
    rewriter.modifyOpInPlace(
        gate, [&] { gate->setOperands(first, parameters.size(), parameters); });
    inlineModifierBody(op, *op.getBody(), qubits(op), rewriter);
    return success();
  }

  /// Fold a power of a fixed one-qubit gate with an exact @p period.
  static LogicalResult foldFixedGate(PowOp op, Operation* gate,
                                     std::optional<double> exponent,
                                     unsigned period,
                                     PatternRewriter& rewriter) {
    const auto loc = op.getLoc();
    Value qubit = inputs(op, gate).front();
    const auto done = [&](Operation* replacement) {
      return replace(op, gate, replacement, rewriter);
    };
    const double sign = isa<SdgOp, TdgOp, SXdgOp>(gate) ? -1. : 1.;
    const double scale =
        sign * 2. * std::numbers::pi / static_cast<double>(period);

    Value angle;
    if (exponent) {
      const double r = *exponent;
      if (isa<XOp>(gate) &&
          std::abs(std::abs(r) - 0.5) < PARAMETER_COMPARISON_TOLERANCE) {
        return done(r > 0. ? SXOp::create(rewriter, loc, qubit).getOperation()
                           : SXdgOp::create(rewriter, loc, qubit));
      }
      if (isa<SXOp, SXdgOp>(gate) &&
          std::abs(std::abs(r) - 2.) < PARAMETER_COMPARISON_TOLERANCE) {
        return done(XOp::create(rewriter, loc, qubit));
      }
      if (isa<ZOp, SOp, SdgOp, TOp, TdgOp>(gate)) {
        return foldPhaseGate(op, gate, qubit, r * scale, rewriter);
      }
      angle = constantFromScalar(rewriter, loc, r * scale);
    } else {
      auto reduced = arith::RemFOp::create(
          rewriter, loc, op.getExponent(),
          constantFromScalar(rewriter, loc, static_cast<double>(period)));
      angle = rewriter.createOrFold<arith::MulFOp>(
          loc, reduced, constantFromScalar(rewriter, loc, scale));
    }

    if (isa<ZOp, SOp, SdgOp, TOp, TdgOp>(gate)) {
      return done(POp::create(rewriter, loc, qubit, angle));
    }
    // X, Y, H, SX, and SXdg have eigenvalues 1 and e^{iπ·period/2}.
    GPhaseOp::create(rewriter, loc,
                     rewriter.createOrFold<arith::MulFOp>(
                         loc, angle, constantFromScalar(rewriter, loc, 0.5)));
    if (isa<YOp>(gate)) {
      return done(RYOp::create(rewriter, loc, qubit, angle));
    }
    if (!isa<HOp>(gate)) {
      return done(RXOp::create(rewriter, loc, qubit, angle));
    }
    // H rotates about (X + Z)/√2, which RY(π/4) maps onto Z.
    const auto thread = [&](Operation* created) {
      qubit = created->getNumResults() == 0 ? qubit : created->getResult(0);
    };
    thread(RYOp::create(rewriter, loc, qubit, -std::numbers::pi / 4.));
    thread(RZOp::create(rewriter, loc, qubit, angle));
    return done(RYOp::create(rewriter, loc, qubit, std::numbers::pi / 4.));
  }

  /// Replace a power of a diagonal phase gate by a named gate or a P gate.
  static LogicalResult foldPhaseGate(PowOp op, Operation* gate, Value qubit,
                                     double angle, PatternRewriter& rewriter) {
    const auto loc = op.getLoc();
    const auto done = [&](Operation* replacement) {
      return replace(op, gate, replacement, rewriter);
    };
    const auto phaseGate = classifyPhaseGate(angle);
    if (!phaseGate) {
      return done(POp::create(rewriter, loc, qubit, angle));
    }
    switch (*phaseGate) {
    case PhaseGate::Identity:
      rewriter.replaceOp(op, qubits(op).take_front(op->getNumResults()));
      return success();
    case PhaseGate::Z:
      return done(ZOp::create(rewriter, loc, qubit));
    case PhaseGate::S:
      return done(SOp::create(rewriter, loc, qubit));
    case PhaseGate::Sdg:
      return done(SdgOp::create(rewriter, loc, qubit));
    case PhaseGate::T:
      return done(TOp::create(rewriter, loc, qubit));
    case PhaseGate::Tdg:
      return done(TdgOp::create(rewriter, loc, qubit));
    }
    llvm_unreachable("unknown phase gate");
  }
};

} // namespace mlir::mqt
