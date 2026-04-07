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

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cmath>
#include <numbers>

using namespace mlir;
using namespace mlir::qc;

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
    rewriter.moveOpBefore(innerOp, op);
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
    if (op.getExponentValue() >= 0.0) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<PowOp>(op, -op.getExponentValue(), [&] {
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

/// Check if a floating-point value is an integer
bool isIntegerExponent(double r) {
  return r == std::floor(r) && std::isfinite(r);
}

/// Materialize r * constant as an arith.constant
Value mulConst(double r, double c, PowOp op, PatternRewriter& rewriter) {
  return arith::ConstantOp::create(rewriter, op.getLoc(),
                                   rewriter.getF64FloatAttr(r * c));
}

/// Materialize exponent * param as arith ops
Value scaleByExponent(Value param, PowOp op, PatternRewriter& rewriter) {
  auto loc = op.getLoc();
  auto exponent =
      arith::ConstantOp::create(rewriter, loc, op.getExponentAttr());
  return arith::MulFOp::create(rewriter, loc, exponent, param);
}

template <typename GateOp>
LogicalResult replaceOneTargetOneParam(auto theta, PowOp op,
                                       PatternRewriter& rewriter) {
  rewriter.replaceOpWithNewOp<GateOp>(op, op.getTarget(0), theta);
  return success();
}

template <typename GateOp>
LogicalResult replaceTwoTargetsOneParam(auto theta, PowOp op,
                                        PatternRewriter& rewriter) {
  rewriter.replaceOpWithNewOp<GateOp>(op, op.getTarget(0), op.getTarget(1),
                                      theta);
  return success();
}

template <typename GateOp>
LogicalResult replaceOneTargetTwoParams(auto theta, auto phi, PowOp op,
                                        PatternRewriter& rewriter) {
  rewriter.replaceOpWithNewOp<GateOp>(op, op.getTarget(0), theta, phi);
  return success();
}

template <typename GateOp>
LogicalResult replaceTwoTargetsTwoParams(auto theta, auto beta, PowOp op,
                                         PatternRewriter& rewriter) {
  rewriter.replaceOpWithNewOp<GateOp>(op, op.getTarget(0), op.getTarget(1),
                                      theta, beta);
  return success();
}

/**
 * @brief Fold pow(r) around gates into simpler operations.
 *
 * Rotation gates: multiply angle by exponent, e.g., pow(r) { rx(θ) } → rx(r*θ)
 * Phase/diagonal gates: convert to P gate, e.g., pow(r) { s } → p(r*π/2)
 * Hermitian gates (integer exponent): even → erase, odd → gate
 * Identity/barrier: pass through unchanged
 */
struct FoldPowIntoGate final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    auto* innerOp = op.getBodyUnitary().getOperation();
    const double r = op.getExponentValue();
    auto loc = op.getLoc();

    // Move supporting ops (constants, arithmetic) out of the body so their
    // Values are accessible from outside and survive PowOp erasure.
    for (auto& bodyOp : llvm::make_early_inc_range(*op.getBody())) {
      if (&bodyOp != innerOp && !llvm::isa<YieldOp>(&bodyOp)) {
        rewriter.moveOpBefore(&bodyOp, op);
      }
    }

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
        // pow(r) { z } → p(r*π)
        .Case<ZOp>([&](auto) {
          rewriter.replaceOpWithNewOp<POp>(
              op, op.getTarget(0), mulConst(r, std::numbers::pi, op, rewriter));
          return success();
        })
        // pow(r) { x } → gphase(-r*π/2); rx(r*π)
        .Case<XOp>([&](auto) {
          if (llvm::isa<CtrlOp, InvOp, PowOp>(op->getParentOp())) {
            return failure();
          }
          GPhaseOp::create(rewriter, loc,
                           mulConst(r, -std::numbers::pi / 2.0, op, rewriter));
          rewriter.replaceOpWithNewOp<RXOp>(
              op, op.getTarget(0), mulConst(r, std::numbers::pi, op, rewriter));
          return success();
        })
        // pow(r) { y } → gphase(-r*π/2); ry(r*π)
        .Case<YOp>([&](auto) {
          if (llvm::isa<CtrlOp, InvOp, PowOp>(op->getParentOp())) {
            return failure();
          }
          GPhaseOp::create(rewriter, loc,
                           mulConst(r, -std::numbers::pi / 2.0, op, rewriter));
          rewriter.replaceOpWithNewOp<RYOp>(
              op, op.getTarget(0), mulConst(r, std::numbers::pi, op, rewriter));
          return success();
        })
        // --- Phase/diagonal gates: convert to P gate ---
        // pow(r) { s } → p(r*π/2)
        .Case<SOp>([&](auto) {
          rewriter.replaceOpWithNewOp<POp>(
              op, op.getTarget(0),
              mulConst(r, std::numbers::pi / 2.0, op, rewriter));
          return success();
        })
        // pow(r) { sdg } → p(-r*π/2)
        .Case<SdgOp>([&](auto) {
          rewriter.replaceOpWithNewOp<POp>(
              op, op.getTarget(0),
              mulConst(r, -std::numbers::pi / 2.0, op, rewriter));
          return success();
        })
        // pow(r) { t } → p(r*π/4)
        .Case<TOp>([&](auto) {
          rewriter.replaceOpWithNewOp<POp>(
              op, op.getTarget(0),
              mulConst(r, std::numbers::pi / 4.0, op, rewriter));
          return success();
        })
        // pow(r) { tdg } → p(-r*π/4)
        .Case<TdgOp>([&](auto) {
          rewriter.replaceOpWithNewOp<POp>(
              op, op.getTarget(0),
              mulConst(r, -std::numbers::pi / 4.0, op, rewriter));
          return success();
        })
        // --- SX/SXdg gates: decompose to rotation + global phase ---
        // pow(r) { sx } → gphase(-r*π/4); rx(r*π/2)
        .Case<SXOp>([&](auto) {
          if (llvm::isa<CtrlOp, InvOp, PowOp>(op->getParentOp())) {
            return failure();
          }
          GPhaseOp::create(rewriter, loc,
                           mulConst(r, -std::numbers::pi / 4.0, op, rewriter));
          rewriter.replaceOpWithNewOp<RXOp>(
              op, op.getTarget(0),
              mulConst(r, std::numbers::pi / 2.0, op, rewriter));
          return success();
        })
        // pow(r) { sxdg } → gphase(r*π/4); rx(-r*π/2)
        .Case<SXdgOp>([&](auto) {
          if (llvm::isa<CtrlOp, InvOp, PowOp>(op->getParentOp())) {
            return failure();
          }
          GPhaseOp::create(rewriter, loc,
                           mulConst(r, std::numbers::pi / 4.0, op, rewriter));
          rewriter.replaceOpWithNewOp<RXOp>(
              op, op.getTarget(0),
              mulConst(r, -std::numbers::pi / 2.0, op, rewriter));
          return success();
        })
        // --- Hermitian gates (integer exponent): even → erase, odd → gate ---
        // pow(n) { h/ecr } → erase (n even) | h/ecr (n odd)
        .Case<HOp, ECROp>([&](auto gate) {
          if (!isIntegerExponent(r)) {
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
          if (!isIntegerExponent(r)) {
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
              mulConst(r, -std::numbers::pi, op, rewriter),
              mulConst(r, 0.0, op, rewriter));
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
  results.add<InlinePow1, ErasePow0, NegPowToInvPow, MergeNestedPow,
              MoveCtrlOutside>(context);
  // Prefer Known Gate optimizations over everything else
  results.add<FoldPowIntoGate>(context, /*benefit=*/2);
}
