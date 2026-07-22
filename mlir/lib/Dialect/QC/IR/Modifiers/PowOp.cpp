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
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <optional>
#include <variant>

using namespace mlir;
using namespace mlir::qc;
using namespace mlir::utils;

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
static LogicalResult tryReplacePOpWithNamedGate(double angle, PowOp op,
                                                PatternRewriter& rewriter) {
  const double norm = normalizeAngle(angle);
  const double pi = std::numbers::pi;

  if (std::abs(norm) < TOLERANCE) {
    rewriter.eraseOp(op);
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

/// pow(1.0) { U }  =>  U
struct InlinePow1 final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    const auto exponent = op.getExponentValue();
    if (!exponent || std::abs(*exponent - 1.0) > TOLERANCE) {
      return failure();
    }
    utils::inlineModifierBody(op, *op.getBody(), op.getQubits(), rewriter);
    return success();
  }
};

/// pow(0.0) { U }  =>  identity (no-op)
struct ErasePow0 final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    const auto exponent = op.getExponentValue();
    if (!exponent || std::abs(*exponent) > TOLERANCE) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

/// pow(p) with p < 0  =>  pow(-p) { inv(q) { U } }
struct NegPowToInvPow final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    const auto exponent = op.getExponentValue();
    // U^{-r} = (U^{-1})^r only when r is an integer: for fractional r,
    // eigenvalue -1 yields (-1)^{-r} ≠ (-1)^r (conjugated phase factors).
    if (!exponent || *exponent >= 0.0 ||
        !utils::isIntegerExponent(-*exponent)) {
      return failure();
    }
    const double exp = *exponent;
    auto qubits = llvm::to_vector(op.getQubits());
    rewriter.replaceOpWithNewOp<PowOp>(
        op, -exp, qubits, [&](ValueRange powArgs) {
          InvOp::create(rewriter, op.getLoc(), powArgs,
                        [&](ValueRange invArgs) {
                          // Inline the old pow body, remapping its block args
                          // to the new inv body's block args.
                          utils::inlineBodyReturningYields(*op.getBody(),
                                                           invArgs, rewriter);
                        });
        });
    return success();
  }
};

/// pow(a) { pow(b) { U } }  =>  pow(a*b) { U }
struct MergeNestedPow final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    const auto outerExponent = op.getExponentValue();
    // Principal matrix powers do not generally satisfy (U^b)^a = U^(a*b)
    // across branch cuts. The rewrite is valid for integral outer powers,
    // where any branch phase is raised to an integer and cancels.
    if (!outerExponent || !utils::isIntegerExponent(*outerExponent)) {
      return failure();
    }
    auto inner = utils::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!inner) {
      return failure();
    }
    auto innerPow = dyn_cast<PowOp>(inner.getOperation());
    if (!innerPow) {
      return failure();
    }
    // The inner pow's operands alias the outer pow's block args, possibly in a
    // different order / subset. Translate them back to the outer pow's operands
    // so the merged pow's footprint matches the inner pow positionally.
    auto outerQubits = op.getQubits();
    const auto qubits = llvm::map_to_vector(innerPow.getQubits(), [&](Value v) {
      return utils::getValueFromBlockArgument(v, outerQubits);
    });
    // Move supporting ops (constants, arithmetic) out of the body so their
    // Values are accessible from outside and survive PowOp erasure.
    utils::hoistSupportingOpsBefore(*op.getBody(), innerPow.getOperation(), op,
                                    rewriter);
    auto merged = scaleByExponent(innerPow.getExponent(), op, rewriter);
    rewriter.replaceOpWithNewOp<PowOp>(
        op, merged, qubits, [&](ValueRange powArgs) {
          // Inner pow body args now match the new pow's args positionally.
          utils::inlineBodyReturningYields(*innerPow.getBody(), powArgs,
                                           rewriter);
        });
    return success();
  }
};

/// pow(p) { ctrl(q) { U } }  =>  ctrl(q) { pow(p) { U } }
struct MoveCtrlOutside final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    auto inner = utils::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!inner) {
      return failure();
    }
    auto innerCtrlOp = dyn_cast<CtrlOp>(inner.getOperation());
    if (!innerCtrlOp) {
      return failure();
    }

    // The inner control's controls and targets are block arguments aliasing the
    // power modifier's qubits. Pull the controls out to a new control
    // modifier and wrap the inner body in a power modifier whose block
    // arguments match the inner targets, so the inner body is reused verbatim.
    auto outerQubits = op.getQubits();
    const auto controls =
        llvm::map_to_vector(innerCtrlOp.getControls(), [&](Value c) {
          return utils::getValueFromBlockArgument(c, outerQubits);
        });
    const auto targets =
        llvm::map_to_vector(innerCtrlOp.getTargets(), [&](Value t) {
          return utils::getValueFromBlockArgument(t, outerQubits);
        });

    rewriter.replaceOpWithNewOp<CtrlOp>(
        op, controls, targets, [&](ValueRange targetArgs) {
          auto innerPow = PowOp::create(rewriter, op.getLoc(), op.getExponent(),
                                        targetArgs);
          rewriter.inlineRegionBefore(innerCtrlOp.getRegion(),
                                      innerPow.getRegion(),
                                      innerPow.getRegion().end());
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
    const auto exponent = op.getExponentValue();
    if (!exponent) {
      return failure();
    }
    const double r = *exponent;
    auto loc = op.getLoc();

    // Pre-check: only proceed for gate types we can fold.
    // HOp, ECROp, RCCXOp, and SWAPOp additionally require an integer exponent.
    if (isa<HOp, ECROp, RCCXOp, SWAPOp>(innerOp) &&
        !utils::isIntegerExponent(r)) {
      return failure();
    }
    if (!isa<GPhaseOp, XOp, YOp, ZOp, SOp, SdgOp, TOp, TdgOp, SXOp, SXdgOp, HOp,
             ECROp, RCCXOp, SWAPOp, RXOp, RYOp, RZOp, POp, ROp, RXXOp, RYYOp,
             RZXOp, RZZOp, XXPlusYYOp, XXMinusYYOp, iSWAPOp, IdOp, BarrierOp>(
            innerOp)) {
      return failure();
    }

    // Move supporting ops (constants, arithmetic) out of the body so their
    // Values are accessible from outside and survive PowOp erasure.
    utils::hoistSupportingOpsBefore(*op.getBody(), innerOp, op, rewriter);

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
          if (succeeded(tryReplacePOpWithNamedGate(angle, op, rewriter))) {
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
          if (succeeded(tryReplacePOpWithNamedGate(angle, op, rewriter))) {
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
          if (succeeded(tryReplacePOpWithNamedGate(angle, op, rewriter))) {
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
          if (succeeded(tryReplacePOpWithNamedGate(angle, op, rewriter))) {
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
          if (succeeded(tryReplacePOpWithNamedGate(angle, op, rewriter))) {
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
        .Case<HOp>([&](auto) {
          if (utils::isEvenExponent(r)) {
            // pow(even) { h } => identity. Erase it.
            rewriter.eraseOp(op);
          } else {
            // pow(odd) { h } => h. The body gate's operands alias the pow's
            // block args; inline the body, remapping them to the outer qubit
            // operands, instead of hoisting the gate out (which would leave it
            // referencing the erased block args).
            utils::inlineModifierBody(op, *op.getBody(), op.getQubits(),
                                      rewriter);
          }
          return success();
        })
        // pow(n) { ecr/rccx/swap } => erase (n even) | gate (n odd)
        .Case<ECROp, RCCXOp, SWAPOp>([&](auto) {
          if (utils::isEvenExponent(r)) {
            // pow(even) { ecr/rccx/swap } => identity. Erase it.
            rewriter.eraseOp(op);
          } else {
            // pow(odd) { ecr/rccx/swap } => gate. Inline the body, remapping
            // its block args to the outer qubit operands (see HOp case above).
            utils::inlineModifierBody(op, *op.getBody(), op.getQubits(),
                                      rewriter);
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
        .Case<BarrierOp>([&](auto) {
          rewriter.replaceOpWithNewOp<BarrierOp>(op, op.getTargets());
          return success();
        })
        .Default([&](auto) { return failure(); });
  }
};

/**
 * @brief Erase power modifiers that do not have any body unitaries.
 */
struct EraseEmptyPow final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getNumBodyUnitaries() != 0) {
      return failure();
    }

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

std::optional<double> PowOp::getExponentValue() {
  return utils::valueToDouble(getExponent());
}

size_t PowOp::getNumBodyUnitaries() {
  return utils::getNumBodyUnitaries<UnitaryOpInterface>(*getBody());
}

UnitaryOpInterface PowOp::getBodyUnitary(const size_t i) {
  return utils::getBodyUnitary<UnitaryOpInterface>(*getBody(), i);
}

void PowOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                  const std::variant<double, Value>& exponent,
                  ValueRange qubits,
                  const function_ref<void(ValueRange)>& bodyBuilder) {
  auto expValue = variantToValue(odsBuilder, odsState.location, exponent);
  build(odsBuilder, odsState, expValue, qubits);
  auto& block = odsState.regions.front()->emplaceBlock();

  auto qubitType = QubitType::get(odsBuilder.getContext());
  for (size_t i = 0; i < qubits.size(); ++i) {
    block.addArgument(qubitType, odsState.location);
  }

  const OpBuilder::InsertionGuard guard(odsBuilder);
  odsBuilder.setInsertionPointToStart(&block);
  bodyBuilder(block.getArguments());
  YieldOp::create(odsBuilder, odsState.location);
}

LogicalResult PowOp::verify() {
  if (llvm::any_of(*getBody(), [](Operation& op) {
        return isa<AllocOp, DeallocOp, MeasureOp, ResetOp, memref::LoadOp,
                   memref::StoreOp>(op);
      })) {
    return emitOpError("body must not contain non-unitary quantum operations "
                       "or modify a quantum register");
  }
  return success();
}

void PowOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<InlinePow1, ErasePow0, FoldPowIntoGate, MergeNestedPow,
              MoveCtrlOutside, NegPowToInvPow, EraseEmptyPow>(context);
}
