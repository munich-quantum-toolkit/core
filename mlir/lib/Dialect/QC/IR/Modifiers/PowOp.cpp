/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCInterfaces.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cmath>
#include <cstddef>
#include <numbers>
#include <variant>

using namespace mlir;
using namespace mlir::qc;
using namespace mlir::utils;
using llvm::make_early_inc_range;
using llvm::reportFatalUsageError;

/**
 * @brief If the computed P-gate angle corresponds to a named gate, emit it
 * directly.
 *
 * @details Uses these equivalences:
 *
 * `Z = P(π)`, `S = P(π/2)`, `Sdg = P(-π/2)`, `T = P(π/4)`, `Tdg = P(-π/4)`
 *
 * Since `P` is diagonal, raising to a power just multiplies the angle:
 *
 * ```
 * Z^r   = P(π)^r    = P(r·π)
 * S^r   = P(π/2)^r  = P(r·π/2)
 * Sdg^r = P(-π/2)^r = P(-r·π/2)
 * T^r   = P(π/4)^r  = P(r·π/4)
 * Tdg^r = P(-π/4)^r = P(-r·π/4)
 * ```
 *
 * The caller computes `angle = r * base_angle` and passes the raw
 * (unnormalized) value here; normalization to (-π, π] is performed internally.
 *
 * Matched angles and their replacements:
 *
 * | Angle          | Replacement |
 * |----------------|-------------|
 * | `angle ≈ 0`    | identity (op replaced with qubit pass-through) |
 * | `angle ≈ +/-π` | `Z`         |
 * | `angle ≈ π/2`  | `S`         |
 * | `angle ≈ -π/2` | `Sdg`       |
 * | `angle ≈ π/4`  | `T`         |
 * | `angle ≈ -π/4` | `Tdg`       |
 *
 * @param angle    Raw phase angle (`r * base_angle`), in radians.
 * @param op       The `PowOp` being rewritten.
 * @param rewriter The pattern rewriter.
 * @return `success()` if replaced, `failure()` if a general `P` gate should be
 * used.
 */
static LogicalResult tryReplaceWithNamedPhaseGate(double angle, PowOp op,
                                                  PatternRewriter& rewriter,
                                                  bool insideModifier) {
  const double norm = normalizeAngle(angle);
  const double pi = std::numbers::pi;

  if (std::abs(norm) < TOLERANCE) {
    if (insideModifier) {
      rewriter.replaceOpWithNewOp<IdOp>(op, op.getTarget(0));
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
  if (std::abs(std::abs(norm) - pi) < TOLERANCE) {
    rewriter.replaceOpWithNewOp<ZOp>(op, op.getTarget(0));
    return success();
  }
  if (std::abs(norm - (pi / 2.0)) < TOLERANCE) {
    rewriter.replaceOpWithNewOp<SOp>(op, op.getTarget(0));
    return success();
  }
  if (std::abs(norm + (pi / 2.0)) < TOLERANCE) {
    rewriter.replaceOpWithNewOp<SdgOp>(op, op.getTarget(0));
    return success();
  }
  if (std::abs(norm - (pi / 4.0)) < TOLERANCE) {
    rewriter.replaceOpWithNewOp<TOp>(op, op.getTarget(0));
    return success();
  }
  if (std::abs(norm + (pi / 4.0)) < TOLERANCE) {
    rewriter.replaceOpWithNewOp<TdgOp>(op, op.getTarget(0));
    return success();
  }
  return failure();
}

/// Materialize exponent * param as arith ops
static Value scaleByExponent(Value param, PowOp op, PatternRewriter& rewriter) {
  return arith::MulFOp::create(rewriter, op.getLoc(), op.getExponent(), param);
}

namespace {

/// pow(1.0) { g }  =>  g
struct InlinePow1 final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    if (std::abs(op.getExponentValue() - 1.0) > TOLERANCE) {
      return failure();
    }
    auto inner = utils::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!inner) {
      return failure();
    }
    utils::inlineModifierBody(op, *op.getBody(), {}, rewriter);
    return success();
  }
};

/// pow(0.0) { g }  =>  identity (no-op)
struct ErasePow0 final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    if (std::abs(op.getExponentValue()) > TOLERANCE) {
      return failure();
    }
    if (isa<CtrlOp, InvOp, PowOp>(op->getParentOp())) {
      if (op.getNumTargets() == 1) {
        rewriter.replaceOpWithNewOp<IdOp>(op, op.getTarget(0));
      } else {
        return failure();
      }
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

/// pow(p) where p < 0  =>  pow(-p) { inv(q) { g } }
struct NegPowToInvPow final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    auto inner = utils::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!inner) {
      return failure();
    }
    const double exp = op.getExponentValue();
    // U^{-r} = (U^{-1})^r only when r is an integer: for fractional r,
    // eigenvalue -1 yields (-1)^{-r} ≠ (-1)^r (conjugated phase factors).
    if (exp >= 0.0 || !utils::isIntegerExponent(-exp)) {
      return failure();
    }
    auto qubits = llvm::to_vector(inner.getQubits());
    rewriter.replaceOpWithNewOp<PowOp>(op, -exp, [&] {
      InvOp::create(rewriter, op.getLoc(), qubits, [&](ValueRange) {
        auto* invBody = rewriter.getInsertionBlock();
        rewriter.inlineBlockBefore(op.getBody(), invBody, invBody->begin());
        rewriter.eraseOp(&invBody->back()); // erase the inlined YieldOp
      });
    });
    return success();
  }
};

/// pow(a) { pow(b) { g } }  =>  pow(a*b) { g }
struct MergeNestedPow final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    auto inner = utils::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!inner) {
      return failure();
    }
    auto innerPow = dyn_cast<PowOp>(inner.getOperation());
    if (!innerPow) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<PowOp>(
        op, op.getExponentValue() * innerPow.getExponentValue(), [&] {
          auto* newBody = rewriter.getInsertionBlock();
          rewriter.inlineBlockBefore(innerPow.getBody(), newBody,
                                     newBody->begin());
          rewriter.eraseOp(&newBody->back()); // erase the inlined YieldOp
        });
    return success();
  }
};

/// pow(p) { ctrl(c, t) { g } }  =>  ctrl(c, t) { pow(p) { g } }
struct MoveCtrlOutside final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    auto inner = utils::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!inner) {
      return failure();
    }
    auto innerCtrl = dyn_cast<CtrlOp>(inner.getOperation());
    if (!innerCtrl) {
      return failure();
    }
    auto controls = llvm::to_vector(innerCtrl.getControls());
    auto targets = llvm::to_vector(innerCtrl.getTargets());
    rewriter.replaceOpWithNewOp<CtrlOp>(
        op, controls, targets, [&](ValueRange ctrlArgs) {
          PowOp::create(rewriter, op.getLoc(), op.getExponentValue(), [&] {
            auto* powBody = rewriter.getInsertionBlock();
            // Inline the old CtrlOp's body, remapping its block arguments to
            // the new CtrlOp's block arguments.
            rewriter.inlineBlockBefore(innerCtrl.getBody(), powBody,
                                       powBody->begin(), ctrlArgs);
            rewriter.eraseOp(&powBody->back()); // erase the inlined YieldOp
          });
        });
    return success();
  }
};

/**
 * @brief Fold pow(r) around gates into simpler operations.
 *
 * @details
 * - Rotation gates: multiply angle by exponent,
 *   e.g., `pow(r) { rx(θ) } => rx(r*θ)`
 * - Phase/diagonal gates: named gate if angle matches, else `P` gate,
 *   e.g., `pow(r) { s } => s/sdg/t/tdg/z` or `p(r*π/2)`
 * - Hermitian gates (integer exponent): even => erase, odd => gate
 * - Identity/barrier: pass through unchanged
 */
struct FoldPowIntoGate final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    auto inner = utils::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!inner) {
      return failure();
    }
    auto* innerOp = inner.getOperation();
    const double r = op.getExponentValue();
    auto loc = op.getLoc();
    const bool insideModifier = isa<CtrlOp, InvOp, PowOp>(op->getParentOp());

    // Folds for X/Y/SX/SXdg emit an additional GPhase op, which is not
    // allowed when nested inside a modifier (single-child constraint).
    if (isa<XOp, YOp, SXOp, SXdgOp>(innerOp) && insideModifier) {
      return failure();
    }

    // Pre-check: only proceed for gate types we can fold.
    // HOp, ECROp, SWAPOp additionally require an integer exponent.
    if (isa<HOp, ECROp, SWAPOp>(innerOp) && !utils::isIntegerExponent(r)) {
      return failure();
    }
    if (!isa<GPhaseOp, XOp, YOp, ZOp, SOp, SdgOp, TOp, TdgOp, SXOp, SXdgOp, HOp,
             ECROp, SWAPOp, RXOp, RYOp, RZOp, POp, ROp, RXXOp, RYYOp, RZXOp,
             RZZOp, XXPlusYYOp, XXMinusYYOp, iSWAPOp, IdOp, BarrierOp>(
            innerOp)) {
      return failure();
    }

    // Move supporting ops (constants, arithmetic) out of the body so their
    // Values are accessible from outside and survive PowOp erasure.
    for (auto& bodyOp : make_early_inc_range(*op.getBody())) {
      if (&bodyOp != innerOp && !isa<YieldOp>(&bodyOp)) {
        rewriter.moveOpBefore(&bodyOp, op);
      }
    }

    return TypeSwitch<Operation*, LogicalResult>(innerOp)
        // --- Rotation gates: multiply angle by exponent ---
        // pow(r) { gphase(θ) } => gphase(r*θ)
        .Case<GPhaseOp>([&](auto gate) {
          auto newParam = scaleByExponent(gate.getTheta(), op, rewriter);
          rewriter.replaceOpWithNewOp<GPhaseOp>(op, newParam);
          return success();
        })
        // pow(r) { rx/ry/rz/p(θ) } => rx/ry/rz/p(r*θ)
        .Case<RXOp, RYOp, RZOp, POp>([&](auto gate) {
          auto newParam = scaleByExponent(gate.getTheta(), op, rewriter);
          rewriter.replaceOpWithNewOp<decltype(gate)>(op, op.getTarget(0),
                                                      newParam);
          return success();
        })
        // pow(r) { rxx/ryy/rzx/rzz(θ) } => rxx/ryy/rzx/rzz(r*θ)
        .Case<RXXOp, RYYOp, RZXOp, RZZOp>([&](auto gate) {
          auto newParam = scaleByExponent(gate.getTheta(), op, rewriter);
          rewriter.replaceOpWithNewOp<decltype(gate)>(
              op, op.getTarget(0), op.getTarget(1), newParam);
          return success();
        })
        // pow(r) { r(θ, φ) } => r(r*θ, φ)
        .Case<ROp>([&](auto gate) {
          auto mul = scaleByExponent(gate.getTheta(), op, rewriter);
          rewriter.replaceOpWithNewOp<ROp>(op, op.getTarget(0), mul,
                                           gate.getPhi());
          return success();
        })
        // pow(r) { xx±yy(θ, β) } => xx±yy(r*θ, β)
        .Case<XXPlusYYOp, XXMinusYYOp>([&](auto gate) {
          auto mul = scaleByExponent(gate.getTheta(), op, rewriter);
          rewriter.replaceOpWithNewOp<decltype(gate)>(
              op, op.getTarget(0), op.getTarget(1), mul, gate.getBeta());
          return success();
        })
        // --- Pauli gates: decompose to rotation + global phase ---
        // pow(r) { z } => named gate if angle matches, else p(r*π)
        .Case<ZOp>([&](auto) {
          const double angle = r * std::numbers::pi;
          if (succeeded(tryReplaceWithNamedPhaseGate(angle, op, rewriter,
                                                     insideModifier))) {
            return success();
          }
          rewriter.replaceOpWithNewOp<POp>(
              op, op.getTarget(0),
              utils::constantFromScalar(rewriter, op.getLoc(),
                                        r * std::numbers::pi));
          return success();
        })
        // pow(r) { x } => gphase(-r*π/2); rx(r*π)
        // pow(1/2) x => sx      (X^(1/2) = SX exactly)
        // pow(-1/2) x => sxdg   (X^(-1/2) = SXdg exactly)
        .Case<XOp>([&](auto) {
          if (std::abs(r - 0.5) < TOLERANCE) {
            rewriter.replaceOpWithNewOp<SXOp>(op, op.getTarget(0));
            return success();
          }
          if (std::abs(r + 0.5) < TOLERANCE) {
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
        // pow(r) { y } => gphase(-r*π/2); ry(r*π)
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
        // --- pow(r) { s } => named gate if angle matches, else p(r*π/2)
        .Case<SOp>([&](auto) {
          const double angle = r * std::numbers::pi / 2.0;
          if (succeeded(tryReplaceWithNamedPhaseGate(angle, op, rewriter,
                                                     insideModifier))) {
            return success();
          }
          rewriter.replaceOpWithNewOp<POp>(
              op, op.getTarget(0),
              utils::constantFromScalar(rewriter, op.getLoc(),
                                        r * (std::numbers::pi / 2.0)));
          return success();
        })
        // pow(r) { sdg } => named gate if angle matches, else p(-r*π/2)
        .Case<SdgOp>([&](auto) {
          const double angle = r * -std::numbers::pi / 2.0;
          if (succeeded(tryReplaceWithNamedPhaseGate(angle, op, rewriter,
                                                     insideModifier))) {
            return success();
          }
          rewriter.replaceOpWithNewOp<POp>(
              op, op.getTarget(0),
              utils::constantFromScalar(rewriter, op.getLoc(),
                                        r * (-std::numbers::pi / 2.0)));
          return success();
        })
        // pow(r) { t } => named gate if angle matches, else p(r*π/4)
        .Case<TOp>([&](auto) {
          const double angle = r * std::numbers::pi / 4.0;
          if (succeeded(tryReplaceWithNamedPhaseGate(angle, op, rewriter,
                                                     insideModifier))) {
            return success();
          }
          rewriter.replaceOpWithNewOp<POp>(
              op, op.getTarget(0),
              utils::constantFromScalar(rewriter, op.getLoc(),
                                        r * (std::numbers::pi / 4.0)));
          return success();
        })
        // pow(r) { tdg } => named gate if angle matches, else p(-r*π/4)
        .Case<TdgOp>([&](auto) {
          const double angle = r * -std::numbers::pi / 4.0;
          if (succeeded(tryReplaceWithNamedPhaseGate(angle, op, rewriter,
                                                     insideModifier))) {
            return success();
          }
          rewriter.replaceOpWithNewOp<POp>(
              op, op.getTarget(0),
              utils::constantFromScalar(rewriter, op.getLoc(),
                                        r * (-std::numbers::pi / 4.0)));
          return success();
        })
        // --- SX/SXdg gates: decompose to rotation + global phase ---
        // pow(r) { sx } => gphase(-r*π/4); rx(r*π/2)
        // pow(±2) sx => x
        .Case<SXOp>([&](auto) {
          if (std::abs(std::abs(r) - 2.0) < TOLERANCE) {
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
        // pow(r) { sxdg } => gphase(r*π/4); rx(-r*π/2)
        // pow(±2) sxdg => x
        .Case<SXdgOp>([&](auto) {
          if (std::abs(std::abs(r) - 2.0) < TOLERANCE) {
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
        // --- Hermitian gates (integer exponent): even => erase/id, odd => gate
        // --- pow(n) { h } => id (n even) | h (n odd)
        .Case<HOp>([&](auto gate) {
          if (!utils::isIntegerExponent(r)) {
            return failure();
          }
          if (utils::isEvenExponent(r)) {
            if (insideModifier) {
              rewriter.replaceOpWithNewOp<IdOp>(op, op.getTarget(0));
            } else {
              rewriter.eraseOp(op);
            }
          } else {
            rewriter.moveOpBefore(gate, op);
            rewriter.eraseOp(op);
          }
          return success();
        })
        // pow(n) { ecr/swap } => erase (n even) | ecr/swap (n odd)
        .Case<ECROp, SWAPOp>([&](auto gate) {
          if (!utils::isIntegerExponent(r)) {
            return failure();
          }
          if (utils::isEvenExponent(r)) {
            if (insideModifier) {
              return failure();
            }
            rewriter.eraseOp(op);
          } else {
            rewriter.moveOpBefore(gate, op);
            rewriter.eraseOp(op);
          }
          return success();
        })
        // --- iSWAP: decompose to parametric gate ---
        // pow(r) { iswap } => xx_plus_yy(-r*π, 0)
        .Case<iSWAPOp>([&](auto) {
          rewriter.replaceOpWithNewOp<XXPlusYYOp>(
              op, op.getTarget(0), op.getTarget(1),
              utils::constantFromScalar(rewriter, op.getLoc(),
                                        r * (-std::numbers::pi)),
              utils::constantFromScalar(rewriter, op.getLoc(), 0.0));
          return success();
        })
        // --- Identity and barrier: pass through unchanged ---
        // pow(r) { id } => id
        .Case<IdOp>([&](auto) {
          rewriter.replaceOpWithNewOp<IdOp>(op, op.getTarget(0));
          return success();
        })
        // pow(r) { barrier } => barrier
        .Case<BarrierOp>([&](auto gate) {
          rewriter.replaceOpWithNewOp<BarrierOp>(op, gate.getTargets());
          return success();
        })
        .Default([&](auto) { return failure(); });
  }
};

} // namespace

double PowOp::getExponentValue() {
  FloatAttr attr;
  if (!matchPattern(getExponent(), m_Constant(&attr))) {
    reportFatalUsageError("PowOp exponent must be a constant");
  }
  return attr.getValueAsDouble();
}

size_t PowOp::getNumBodyUnitaries() {
  return utils::getNumBodyUnitaries<UnitaryOpInterface>(*getBody());
}

UnitaryOpInterface PowOp::getBodyUnitary(const size_t i) {
  return utils::getBodyUnitary<UnitaryOpInterface>(*getBody(), i);
}

void PowOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                  const std::variant<double, Value>& exponent,
                  const function_ref<void()>& bodyBuilder) {
  auto expValue = variantToValue(odsBuilder, odsState.location, exponent);
  odsState.addOperands(expValue);
  auto* region = odsState.addRegion();
  auto& block = region->emplaceBlock();

  const OpBuilder::InsertionGuard guard(odsBuilder);
  odsBuilder.setInsertionPointToStart(&block);
  bodyBuilder();
  YieldOp::create(odsBuilder, odsState.location);
}

LogicalResult PowOp::verify() {
  FloatAttr attr;
  if (!matchPattern(getExponent(), m_Constant(&attr))) {
    return emitOpError("exponent must be a constant");
  }

  auto& block = *getBody();
  if (block.getOperations().size() < 2) {
    return emitOpError("body region must have at least two operations");
  }
  if (!isa<YieldOp>(block.back())) {
    return emitOpError(
        "last operation in body region must be a yield operation");
  }
  auto iter = ++block.rbegin();
  if (!isa<UnitaryOpInterface>(*iter)) {
    return emitOpError(
        "second to last operation in body region must be a unitary operation");
  }
  for (auto it = ++iter; it != block.rend(); ++it) {
    if (isa<UnitaryOpInterface>(*it)) {
      return emitOpError("body region may only contain a single unitary op");
    }
  }
  return success();
}

void PowOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<InlinePow1, ErasePow0, FoldPowIntoGate, MergeNestedPow,
              MoveCtrlOutside, NegPowToInvPow>(context);
}
