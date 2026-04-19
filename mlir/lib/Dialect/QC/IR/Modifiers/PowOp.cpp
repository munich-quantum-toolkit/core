/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cmath>
#include <cstdint>
#include <numbers>

using namespace mlir;
using namespace mlir::qc;

/**
 * @brief If the computed P-gate angle corresponds to a named gate, emit it
 * directly.
 *
 * Uses these equivalences:
 *   Z = P(π),  S = P(π/2),  Sdg = P(-π/2),  T = P(π/4),  Tdg = P(-π/4)
 *
 * Since P is diagonal, raising to a power just multiplies the angle:
 *   Z^r   = P(π)^r    = P(r·π)
 *   S^r   = P(π/2)^r  = P(r·π/2)
 *   Sdg^r = P(-π/2)^r = P(-r·π/2)
 *   T^r   = P(π/4)^r  = P(r·π/4)
 *   Tdg^r = P(-π/4)^r = P(-r·π/4)
 *
 * The caller computes angle = r * base_angle and passes the raw (unnormalized)
 * value here; normalization to (-π, π] is performed internally.
 *
 * Matched angles and their replacements:
 *   angle ≈ 0           → identity (op replaced with qubit pass-through)
 *   angle ≈ ±π  → Z,   angle ≈  π/2 → S,   angle ≈ -π/2 → Sdg
 *   angle ≈  π/4 → T,  angle ≈ -π/4 → Tdg
 *
 * @param angle    Raw phase angle (r · base_angle), in radians.
 * @param op       The PowOp being rewritten.
 * @param rewriter The pattern rewriter.
 * @return success() if replaced, failure() if a general P gate should be used.
 */
static LogicalResult tryReplaceWithNamedPhaseGate(double angle, PowOp op,
                                                  PatternRewriter& rewriter) {
  constexpr double eps = 1e-10;
  const double norm = utils::normalizeAngle(angle);
  const double pi = std::numbers::pi;

  if (std::abs(norm) < eps) {
    rewriter.eraseOp(op);
    return success();
  }
  if (std::abs(std::abs(norm) - pi) < eps) {
    rewriter.replaceOpWithNewOp<ZOp>(op, op.getTarget(0));
    return success();
  }
  if (std::abs(norm - (pi / 2.0)) < eps) {
    rewriter.replaceOpWithNewOp<SOp>(op, op.getTarget(0));
    return success();
  }
  if (std::abs(norm + (pi / 2.0)) < eps) {
    rewriter.replaceOpWithNewOp<SdgOp>(op, op.getTarget(0));
    return success();
  }
  if (std::abs(norm - (pi / 4.0)) < eps) {
    rewriter.replaceOpWithNewOp<TOp>(op, op.getTarget(0));
    return success();
  }
  if (std::abs(norm + (pi / 4.0)) < eps) {
    rewriter.replaceOpWithNewOp<TdgOp>(op, op.getTarget(0));
    return success();
  }
  return failure();
}

/// Materialize exponent * param as arith ops
static Value scaleByExponent(Value param, PowOp op, PatternRewriter& rewriter) {
  auto loc = op.getLoc();
  auto exponent =
      arith::ConstantOp::create(rewriter, loc, op.getExponentAttr());
  return arith::MulFOp::create(rewriter, loc, exponent, param);
}

template <typename GateOp>
static LogicalResult replaceOneTargetOneParam(auto theta, PowOp op,
                                              PatternRewriter& rewriter) {
  rewriter.replaceOpWithNewOp<GateOp>(op, op.getTarget(0), theta);
  return success();
}

template <typename GateOp>
static LogicalResult replaceTwoTargetsOneParam(auto theta, PowOp op,
                                               PatternRewriter& rewriter) {
  rewriter.replaceOpWithNewOp<GateOp>(op, op.getTarget(0), op.getTarget(1),
                                      theta);
  return success();
}

template <typename GateOp>
static LogicalResult replaceOneTargetTwoParams(auto theta, auto phi, PowOp op,
                                               PatternRewriter& rewriter) {
  rewriter.replaceOpWithNewOp<GateOp>(op, op.getTarget(0), theta, phi);
  return success();
}

template <typename GateOp>
static LogicalResult replaceTwoTargetsTwoParams(auto theta, auto beta, PowOp op,
                                                PatternRewriter& rewriter) {
  rewriter.replaceOpWithNewOp<GateOp>(op, op.getTarget(0), op.getTarget(1),
                                      theta, beta);
  return success();
}

namespace {

/// pow(1.0) @ g  =>  g
struct InlinePow1 final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getExponentValue() != 1.0) {
      return failure();
    }
    auto* innerOp = op.getBodyUnitary().getOperation();
    rewriter.inlineBlockBefore(op.getBody(), op, {});
    rewriter.eraseOp(op->getPrevNode()); // erase the now-inlined YieldOp
    rewriter.replaceOp(op, innerOp->getResults());
    return success();
  }
};

/// pow(0.0) @ g  =>  erase (identity / no-op)
struct ErasePow0 final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getExponentValue() != 0.0) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

/// pow(p) where p < 0  =>  pow(-p) { inv { g } }
struct NegPowToInvPow final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    const double exp = op.getExponentValue();
    // U^{-r} = (U^{-1})^r only when r is an integer: for fractional r,
    // eigenvalue -1 yields (-1)^{-r} ≠ (-1)^r (conjugated phase factors).
    if (exp >= 0.0 || !utils::isIntegerExponent(-exp)) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<PowOp>(op, -exp, [&] {
      InvOp::create(rewriter, op.getLoc(), [&] {
        rewriter.clone(*op.getBodyUnitary().getOperation());
      });
    });
    return success();
  }
};

/// pow(a, pow(b, g))  =>  pow(a*b, g)
struct MergeNestedPow final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    auto innerPow = llvm::dyn_cast<PowOp>(op.getBodyUnitary().getOperation());
    if (!innerPow) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<PowOp>(
        op, op.getExponentValue() * innerPow.getExponentValue(),
        [&] { rewriter.clone(*innerPow.getBodyUnitary().getOperation()); });
    return success();
  }
};

/// pow(p, ctrl(q, g))  =>  ctrl(q, pow(p, g))
struct MoveCtrlOutside final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    auto innerCtrl = llvm::dyn_cast<CtrlOp>(op.getBodyUnitary().getOperation());
    if (!innerCtrl) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<CtrlOp>(op, innerCtrl.getControls(), [&] {
      PowOp::create(rewriter, op.getLoc(), op.getExponentValue(), [&] {
        rewriter.clone(*innerCtrl.getBodyUnitary().getOperation());
      });
    });
    return success();
  }
};

/**
 * @brief Fold pow(r) around gates into simpler operations.
 *
 * Rotation gates: multiply angle by exponent, e.g., pow(r) { rx(θ) } → rx(r*θ)
 * Phase/diagonal gates: named gate if angle matches, else P gate, e.g., pow(r)
 * { s } → s/sdg/t/tdg/z or p(r*π/2) Hermitian gates (integer exponent): even →
 * erase, odd → gate Identity/barrier: pass through unchanged
 */
struct FoldPowIntoGate final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    auto* innerOp = op.getBodyUnitary().getOperation();
    const double r = op.getExponentValue();
    auto loc = op.getLoc();

    // Folds for X/Y/SX/SXdg emit an additional GPhase op, which is not
    // allowed when nested inside a modifier (single-child constraint).
    if (llvm::isa<XOp, YOp, SXOp, SXdgOp>(innerOp) &&
        llvm::isa<CtrlOp, InvOp, PowOp>(op->getParentOp())) {
      return failure();
    }

    // Pre-check: only proceed for gate types we can fold.
    // HOp, ECROp, SWAPOp additionally require an integer exponent.
    if (llvm::isa<HOp, ECROp, SWAPOp>(innerOp) &&
        !utils::isIntegerExponent(r)) {
      return failure();
    }
    if (!llvm::isa<GPhaseOp, XOp, YOp, ZOp, SOp, SdgOp, TOp, TdgOp, SXOp,
                   SXdgOp, HOp, ECROp, SWAPOp, RXOp, RYOp, RZOp, POp, ROp,
                   RXXOp, RYYOp, RZXOp, RZZOp, XXPlusYYOp, XXMinusYYOp, iSWAPOp,
                   IdOp, BarrierOp>(innerOp)) {
      return failure();
    }

    // Move supporting ops (constants, arithmetic) out of the body so their
    // Values are accessible from outside and survive PowOp erasure.
    for (auto& bodyOp : llvm::make_early_inc_range(*op.getBody())) {
      if (&bodyOp != innerOp && !llvm::isa<YieldOp>(&bodyOp)) {
        rewriter.moveOpBefore(&bodyOp, op);
      }
    }

    // Set insertion point before the PowOp so that new ops (constants,
    // arithmetic) are created outside the body region.
    // rewriter.setInsertionPoint(op);

    return llvm::TypeSwitch<Operation*, LogicalResult>(innerOp)
        // --- Rotation gates: multiply angle by exponent ---
        // pow(r) { gphase(θ) } → gphase(r*θ)
        .Case<GPhaseOp>([&](auto gate) {
          auto newParam = scaleByExponent(gate.getTheta(), op, rewriter);
          rewriter.replaceOpWithNewOp<GPhaseOp>(op, newParam);
          return success();
        })
        // pow(r) { rx/ry/rz/p(θ) } → rx/ry/rz/p(r*θ)
        .Case<RXOp, RYOp, RZOp, POp>([&](auto gate) {
          auto newParam = scaleByExponent(gate.getTheta(), op, rewriter);
          return replaceOneTargetOneParam<decltype(gate)>(newParam, op,
                                                          rewriter);
        })
        // pow(r) { rxx/ryy/rzx/rzz(θ) } → rxx/ryy/rzx/rzz(r*θ)
        .Case<RXXOp, RYYOp, RZXOp, RZZOp>([&](auto gate) {
          auto newParam = scaleByExponent(gate.getTheta(), op, rewriter);
          return replaceTwoTargetsOneParam<decltype(gate)>(newParam, op,
                                                           rewriter);
        })
        // pow(r) { r(θ, φ) } → r(r*θ, φ)
        .Case<ROp>([&](auto gate) {
          auto mul = scaleByExponent(gate.getTheta(), op, rewriter);
          return replaceOneTargetTwoParams<ROp>(mul, gate.getPhi(), op,
                                                rewriter);
        })
        // pow(r) { xx±yy(θ, β) } → xx±yy(r*θ, β)
        .Case<XXPlusYYOp, XXMinusYYOp>([&](auto gate) {
          auto mul = scaleByExponent(gate.getTheta(), op, rewriter);
          return replaceTwoTargetsTwoParams<decltype(gate)>(mul, gate.getBeta(),
                                                            op, rewriter);
        })
        // --- Pauli gates: decompose to rotation + global phase ---
        // pow(r) { z } → named gate if angle matches, else p(r*π)
        .Case<ZOp>([&](auto) {
          const double angle = r * std::numbers::pi;
          if (succeeded(tryReplaceWithNamedPhaseGate(angle, op, rewriter))) {
            return success();
          }
          rewriter.replaceOpWithNewOp<POp>(
              op, op.getTarget(0),
              utils::constantFromScalar(rewriter, op.getLoc(),
                                        r * std::numbers::pi));
          return success();
        })
        // pow(r) { x } → gphase(-r*π/2); rx(r*π)
        // pow(1/2) x → sx      (X^(1/2) = SX exactly)
        // pow(-1/2) x → sxdg   (X^(-1/2) = SXdg exactly)
        .Case<XOp>([&](auto) {
          if (r == 0.5) {
            rewriter.replaceOpWithNewOp<SXOp>(op, op.getTarget(0));
            return success();
          }
          if (r == -0.5) {
            rewriter.replaceOpWithNewOp<SXdgOp>(op, op.getTarget(0));
            return success();
          }
          GPhaseOp::create(
              rewriter, loc,
              utils::constantFromScalar(rewriter, op.getLoc(),
                                        r * (-std::numbers::pi / 2.0)));
          rewriter.replaceOpWithNewOp<RXOp>(
              op, op.getTarget(0),
              utils::constantFromScalar(rewriter, op.getLoc(),
                                        r * std::numbers::pi));
          return success();
        })
        // pow(r) { y } → gphase(-r*π/2); ry(r*π)
        .Case<YOp>([&](auto) {
          GPhaseOp::create(
              rewriter, loc,
              utils::constantFromScalar(rewriter, op.getLoc(),
                                        r * (-std::numbers::pi / 2.0)));
          rewriter.replaceOpWithNewOp<RYOp>(
              op, op.getTarget(0),
              utils::constantFromScalar(rewriter, op.getLoc(),
                                        r * std::numbers::pi));
          return success();
        })
        // --- Phase/diagonal gates: named gate if angle matches, else P gate
        // --- pow(r) { s } → named gate if angle matches, else p(r*π/2)
        .Case<SOp>([&](auto) {
          const double angle = r * std::numbers::pi / 2.0;
          if (succeeded(tryReplaceWithNamedPhaseGate(angle, op, rewriter))) {
            return success();
          }
          rewriter.replaceOpWithNewOp<POp>(
              op, op.getTarget(0),
              utils::constantFromScalar(rewriter, op.getLoc(),
                                        r * (std::numbers::pi / 2.0)));
          return success();
        })
        // pow(r) { sdg } → named gate if angle matches, else p(-r*π/2)
        .Case<SdgOp>([&](auto) {
          const double angle = r * -std::numbers::pi / 2.0;
          if (succeeded(tryReplaceWithNamedPhaseGate(angle, op, rewriter))) {
            return success();
          }
          rewriter.replaceOpWithNewOp<POp>(
              op, op.getTarget(0),
              utils::constantFromScalar(rewriter, op.getLoc(),
                                        r * (-std::numbers::pi / 2.0)));
          return success();
        })
        // pow(r) { t } → named gate if angle matches, else p(r*π/4)
        .Case<TOp>([&](auto) {
          const double angle = r * std::numbers::pi / 4.0;
          if (succeeded(tryReplaceWithNamedPhaseGate(angle, op, rewriter))) {
            return success();
          }
          rewriter.replaceOpWithNewOp<POp>(
              op, op.getTarget(0),
              utils::constantFromScalar(rewriter, op.getLoc(),
                                        r * (std::numbers::pi / 4.0)));
          return success();
        })
        // pow(r) { tdg } → named gate if angle matches, else p(-r*π/4)
        .Case<TdgOp>([&](auto) {
          const double angle = r * -std::numbers::pi / 4.0;
          if (succeeded(tryReplaceWithNamedPhaseGate(angle, op, rewriter))) {
            return success();
          }
          rewriter.replaceOpWithNewOp<POp>(
              op, op.getTarget(0),
              utils::constantFromScalar(rewriter, op.getLoc(),
                                        r * (-std::numbers::pi / 4.0)));
          return success();
        })
        // --- SX/SXdg gates: decompose to rotation + global phase ---
        // pow(r) { sx } → gphase(-r*π/4); rx(r*π/2)
        // pow(±2) sx → x
        .Case<SXOp>([&](auto) {
          if (std::abs(r) == 2.0) {
            rewriter.replaceOpWithNewOp<XOp>(op, op.getTarget(0));
            return success();
          }
          GPhaseOp::create(
              rewriter, loc,
              utils::constantFromScalar(rewriter, op.getLoc(),
                                        r * (-std::numbers::pi / 4.0)));
          rewriter.replaceOpWithNewOp<RXOp>(
              op, op.getTarget(0),
              utils::constantFromScalar(rewriter, op.getLoc(),
                                        r * (std::numbers::pi / 2.0)));
          return success();
        })
        // pow(r) { sxdg } → gphase(r*π/4); rx(-r*π/2)
        // pow(±2) sxdg → x
        .Case<SXdgOp>([&](auto) {
          if (std::abs(r) == 2.0) {
            rewriter.replaceOpWithNewOp<XOp>(op, op.getTarget(0));
            return success();
          }
          GPhaseOp::create(
              rewriter, loc,
              utils::constantFromScalar(rewriter, op.getLoc(),
                                        r * (std::numbers::pi / 4.0)));
          rewriter.replaceOpWithNewOp<RXOp>(
              op, op.getTarget(0),
              utils::constantFromScalar(rewriter, op.getLoc(),
                                        r * (-std::numbers::pi / 2.0)));
          return success();
        })
        // --- Hermitian gates (integer exponent): even → erase, odd → gate ---
        // pow(n) { h/ecr } → erase (n even) | h/ecr (n odd)
        .Case<HOp, ECROp>([&](auto gate) {
          if (!utils::isIntegerExponent(r)) {
            return failure();
          }
          const auto n = static_cast<int64_t>(r);
          if (n % 2 == 0) {
            rewriter.eraseOp(op);
          } else {
            rewriter.moveOpBefore(gate, op);
            rewriter.eraseOp(op);
          }
          return success();
        })
        // pow(n) { swap } → erase (n even) | swap (n odd)
        .Case<SWAPOp>([&](auto gate) {
          if (!utils::isIntegerExponent(r)) {
            return failure();
          }
          const auto n = static_cast<int64_t>(r);
          if (n % 2 == 0) {
            rewriter.eraseOp(op);
          } else {
            rewriter.moveOpBefore(gate, op);
            rewriter.eraseOp(op);
          }
          return success();
        })
        // --- iSWAP: decompose to parametric gate ---
        // pow(r) { iswap } → xx_plus_yy(-r*π, 0)
        .Case<iSWAPOp>([&](auto) {
          rewriter.replaceOpWithNewOp<XXPlusYYOp>(
              op, op.getTarget(0), op.getTarget(1),
              utils::constantFromScalar(rewriter, op.getLoc(),
                                        r * (-std::numbers::pi)),
              utils::constantFromScalar(rewriter, op.getLoc(), 0.0));
          return success();
        })
        // --- Identity and barrier: pass through unchanged ---
        // pow(r) { id } → id
        .Case<IdOp>([&](auto) {
          rewriter.replaceOpWithNewOp<IdOp>(op, op.getTarget(0));
          return success();
        })
        // pow(r) { barrier } → barrier
        .Case<BarrierOp>([&](auto gate) {
          rewriter.replaceOpWithNewOp<BarrierOp>(op, gate.getTargets());
          return success();
        })
        .Default([&](auto) { return failure(); });
  }
};

} // namespace

UnitaryOpInterface PowOp::getBodyUnitary() {
  return llvm::cast<UnitaryOpInterface>(*(++getBody()->rbegin()));
}

void PowOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                  double exponent,
                  const llvm::function_ref<void()>& bodyBuilder) {
  const OpBuilder::InsertionGuard guard(odsBuilder);
  odsState.addAttribute("exponent", odsBuilder.getF64FloatAttr(exponent));
  auto* region = odsState.addRegion();
  auto& block = region->emplaceBlock();

  odsBuilder.setInsertionPointToStart(&block);
  bodyBuilder();
  YieldOp::create(odsBuilder, odsState.location);
}

LogicalResult PowOp::verify() {
  auto& block = *getBody();
  if (block.getOperations().size() < 2) {
    return emitOpError("body region must have at least two operations");
  }
  if (!llvm::isa<YieldOp>(block.back())) {
    return emitOpError(
        "last operation in body region must be a yield operation");
  }
  auto iter = ++block.rbegin();
  if (!llvm::isa<UnitaryOpInterface>(*iter)) {
    return emitOpError(
        "second to last operation in body region must be a unitary operation");
  }
  for (auto it = ++iter; it != block.rend(); ++it) {
    if (llvm::isa<UnitaryOpInterface>(*it)) {
      return emitOpError("body region may only contain a single unitary op");
    }
  }
  return success();
}

void PowOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<InlinePow1, ErasePow0, FoldPowIntoGate, NegPowToInvPow,
              MergeNestedPow, MoveCtrlOutside>(context);
}
