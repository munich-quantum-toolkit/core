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
#include "mlir/Dialect/QCO/Utils/Matrix.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/QTensor/IR/QTensorOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cmath>
#include <cstddef>
#include <numbers>
#include <optional>
#include <variant>

using namespace mlir;
using namespace mlir::qco;
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
    // pow(r) folds to the identity: thread the input qubits to the results.
    rewriter.replaceOp(op, op.getQubitsIn());
    return success();
  }
  if (std::abs(std::abs(norm) - pi) < TOLERANCE) {
    rewriter.replaceOpWithNewOp<ZOp>(op, op.getInputTarget(0));
    return success();
  }
  if (std::abs(norm - (pi / 2.0)) < TOLERANCE) {
    rewriter.replaceOpWithNewOp<SOp>(op, op.getInputTarget(0));
    return success();
  }
  if (std::abs(norm + (pi / 2.0)) < TOLERANCE) {
    rewriter.replaceOpWithNewOp<SdgOp>(op, op.getInputTarget(0));
    return success();
  }
  if (std::abs(norm - (pi / 4.0)) < TOLERANCE) {
    rewriter.replaceOpWithNewOp<TOp>(op, op.getInputTarget(0));
    return success();
  }
  if (std::abs(norm + (pi / 4.0)) < TOLERANCE) {
    rewriter.replaceOpWithNewOp<TdgOp>(op, op.getInputTarget(0));
    return success();
  }
  return failure();
}

/// Materialize exponent * param as arith ops
static Value scaleByExponent(auto param, PowOp op, PatternRewriter& rewriter) {
  return arith::MulFOp::create(rewriter, op.getLoc(), op.getExponent(), param);
}

namespace {

/// pow(1.0) { g }  =>  inline g
struct InlinePow1 final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    if (std::abs(op.getExponentValue() - 1.0) > TOLERANCE) {
      return failure();
    }

    utils::inlineModifierBody(op, *op.getBody(), op.getInputQubits(), rewriter);
    return success();
  }
};

/// pow(0.0) { g }  =>  identity (pass-through)
struct ErasePow0 final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    if (std::abs(op.getExponentValue()) > TOLERANCE) {
      return failure();
    }

    // pow(0) is the identity: thread the input qubits straight to the results.
    rewriter.replaceOp(op, op.getQubitsIn());
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

    rewriter.replaceOpWithNewOp<PowOp>(
        op, op.getQubitsIn(), -exp,
        [&](ValueRange powArgs) -> SmallVector<Value> {
          return InvOp::create(rewriter, op.getLoc(), powArgs,
                               [&](ValueRange invArgs) -> SmallVector<Value> {
                                 auto* invBody = rewriter.getInsertionBlock();
                                 rewriter.inlineBlockBefore(
                                     op.getBody(), invBody, invBody->begin(),
                                     invArgs);
                                 auto yieldedValues =
                                     llvm::to_vector(invBody->back().getOperands());
                                 rewriter.eraseOp(&invBody->back());
                                 return yieldedValues;
                               })
              .getResults();
        });

    return success();
  }
};

/// pow(a) { pow(b) { g } }  =>  pow(a*b) { g }
struct MergeNestedPow final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    auto bodyUnitary =
        utils::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!bodyUnitary) {
      return failure();
    }
    auto innerPow = dyn_cast<PowOp>(bodyUnitary.getOperation());
    if (!innerPow) {
      return failure();
    }

    const double merged = op.getExponentValue() * innerPow.getExponentValue();
    auto mergedConst = arith::ConstantFloatOp::create(
        rewriter, op.getLoc(), rewriter.getF64Type(), APFloat(merged));

    rewriter.moveOpBefore(innerPow, op);
    rewriter.modifyOpInPlace(innerPow, [&]() {
      innerPow.getExponentMutable().assign(mergedConst.getResult());
      innerPow.getQubitsInMutable().assign(op.getInputQubits());
    });
    rewriter.replaceOp(op, innerPow->getResults());
    return success();
  }
};

/// pow(p) { ctrl(q, g) }  =>  ctrl(q, pow(p, g))
struct MoveCtrlOutside final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    auto bodyUnitary =
        utils::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!bodyUnitary) {
      return failure();
    }
    auto innerCtrlOp = dyn_cast<CtrlOp>(bodyUnitary.getOperation());
    if (!innerCtrlOp) {
      return failure();
    }

    const auto numControls = innerCtrlOp.getNumControls();
    const auto numTargets = innerCtrlOp.getNumTargets();
    auto inputQubits = op.getInputQubits();
    auto controls = inputQubits.take_front(numControls);
    auto targets = inputQubits.take_back(numTargets);
    const double exponent = op.getExponentValue();

    rewriter.replaceOpWithNewOp<CtrlOp>(
        op, controls, targets,
        [&](ValueRange ctrlTargetArgs) -> SmallVector<Value> {
          return PowOp::create(rewriter, op.getLoc(), ctrlTargetArgs, exponent,
                               [&](ValueRange powArgs) -> SmallVector<Value> {
                                 auto* powBody = rewriter.getInsertionBlock();
                                 rewriter.inlineBlockBefore(
                                     innerCtrlOp.getBody(), powBody,
                                     powBody->begin(), powArgs);
                                 auto yieldedValues =
                                     llvm::to_vector(powBody->back().getOperands());
                                 rewriter.eraseOp(&powBody->back());
                                 return yieldedValues;
                               })
              .getResults();
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
    auto bodyUnitary =
        utils::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!bodyUnitary) {
      return failure();
    }
    auto* innerOp = bodyUnitary.getOperation();
    const double r = op.getExponentValue();
    auto loc = op.getLoc();
    const bool insideCtrl = op->getParentOfType<CtrlOp>() != nullptr;

    // These folds emit a separate GPhase alongside the rotation. Under a Ctrl
    // (at any nesting depth) a GPhase becomes an observable controlled phase
    // that is only resolved (ctrl{ gphase } => P) when it is the ctrl body's
    // sole op, so don't fold X/Y/SX/SXdg anywhere inside a control.
    if (isa<XOp, YOp, SXOp, SXdgOp>(innerOp) && insideCtrl) {
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

    // Inline the body before op so all parameter-defining ops (constants,
    // arithmetic) are in scope and survive op replacement.
    rewriter.inlineBlockBefore(op.getBody(), op, op.getInputQubits());
    rewriter.eraseOp(op->getPrevNode()); // erase the now-inlined YieldOp
    rewriter.setInsertionPoint(op);

    const LogicalResult result =
        TypeSwitch<Operation*, LogicalResult>(innerOp)
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
              rewriter.replaceOpWithNewOp<decltype(gate)>(
                  op, op.getInputTarget(0), newParam);
              return success();
            })
            // pow(r) { rxx/ryy/rzx/rzz(θ) } => rxx/ryy/rzx/rzz(r*θ)
            .Case<RXXOp, RYYOp, RZXOp, RZZOp>([&](auto gate) {
              auto newParam = scaleByExponent(gate.getTheta(), op, rewriter);
              rewriter.replaceOpWithNewOp<decltype(gate)>(
                  op, op.getInputTarget(0), op.getInputTarget(1), newParam);
              return success();
            })
            // pow(r) { r(θ, φ) } => r(r*θ, φ)
            .Case<ROp>([&](auto gate) {
              auto mul = scaleByExponent(gate.getTheta(), op, rewriter);
              rewriter.replaceOpWithNewOp<ROp>(op, op.getInputTarget(0), mul,
                                               gate.getPhi());
              return success();
            })
            // pow(r) { xx±yy(θ, β) } => xx±yy(r*θ, β)
            .Case<XXPlusYYOp, XXMinusYYOp>([&](auto gate) {
              auto mul = scaleByExponent(gate.getTheta(), op, rewriter);
              rewriter.replaceOpWithNewOp<decltype(gate)>(
                  op, op.getInputTarget(0), op.getInputTarget(1), mul,
                  gate.getBeta());
              return success();
            })
            // --- Pauli gates: decompose to rotation + global phase ---
            // pow(r) { x } => gphase(-r*π/2); rx(r*π)
            // pow(1/2) x => sx      (X^(1/2) = SX exactly)
            // pow(-1/2) x => sxdg   (X^(-1/2) = SXdg exactly)
            .Case<XOp>([&](auto) {
              if (std::abs(r - 0.5) < TOLERANCE) {
                rewriter.replaceOpWithNewOp<SXOp>(op, op.getInputTarget(0));
                return success();
              }
              if (std::abs(r + 0.5) < TOLERANCE) {
                rewriter.replaceOpWithNewOp<SXdgOp>(op, op.getInputTarget(0));
                return success();
              }
              GPhaseOp::create(
                  rewriter, loc,
                  utils::constantFromScalar(rewriter, op.getLoc(),
                                            r * (-std::numbers::pi / 2.0)));
              rewriter.replaceOpWithNewOp<RXOp>(
                  op, op.getInputTarget(0),
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
                  op, op.getInputTarget(0),
                  utils::constantFromScalar(rewriter, op.getLoc(),
                                            r * std::numbers::pi));
              return success();
            })
            // pow(r) { z } => named gate if angle matches, else p(r*π)
            .Case<ZOp>([&](auto) {
              const double angle = r * std::numbers::pi;
              if (succeeded(
                      tryReplacePOpWithNamedGate(angle, op, rewriter))) {
                return success();
              }
              rewriter.replaceOpWithNewOp<POp>(
                  op, op.getInputTarget(0),
                  utils::constantFromScalar(rewriter, op.getLoc(),
                                            r * std::numbers::pi));
              return success();
            })
            // --- Phase/diagonal gates: named gate if angle matches, else P
            // gate
            // --- pow(r) { s } => named gate if angle matches, else p(r*π/2)
            .Case<SOp>([&](auto) {
              const double angle = r * std::numbers::pi / 2.0;
              if (succeeded(
                      tryReplacePOpWithNamedGate(angle, op, rewriter))) {
                return success();
              }
              rewriter.replaceOpWithNewOp<POp>(
                  op, op.getInputTarget(0),
                  utils::constantFromScalar(rewriter, op.getLoc(),
                                            r * (std::numbers::pi / 2.0)));
              return success();
            })
            // pow(r) { sdg } => named gate if angle matches, else p(-r*π/2)
            .Case<SdgOp>([&](auto) {
              const double angle = r * -std::numbers::pi / 2.0;
              if (succeeded(
                      tryReplacePOpWithNamedGate(angle, op, rewriter))) {
                return success();
              }
              rewriter.replaceOpWithNewOp<POp>(
                  op, op.getInputTarget(0),
                  utils::constantFromScalar(rewriter, op.getLoc(),
                                            r * (-std::numbers::pi / 2.0)));
              return success();
            })
            // pow(r) { t } => named gate if angle matches, else p(r*π/4)
            .Case<TOp>([&](auto) {
              const double angle = r * std::numbers::pi / 4.0;
              if (succeeded(
                      tryReplacePOpWithNamedGate(angle, op, rewriter))) {
                return success();
              }
              rewriter.replaceOpWithNewOp<POp>(
                  op, op.getInputTarget(0),
                  utils::constantFromScalar(rewriter, op.getLoc(),
                                            r * (std::numbers::pi / 4.0)));
              return success();
            })
            // pow(r) { tdg } => named gate if angle matches, else p(-r*π/4)
            .Case<TdgOp>([&](auto) {
              const double angle = r * -std::numbers::pi / 4.0;
              if (succeeded(
                      tryReplacePOpWithNamedGate(angle, op, rewriter))) {
                return success();
              }
              rewriter.replaceOpWithNewOp<POp>(
                  op, op.getInputTarget(0),
                  utils::constantFromScalar(rewriter, op.getLoc(),
                                            r * (-std::numbers::pi / 4.0)));
              return success();
            })
            // --- SX/SXdg gates: decompose to rotation + global phase ---
            // pow(r) { sx } => gphase(-r*π/4); rx(r*π/2)
            // pow(±2) sx => x
            .Case<SXOp>([&](auto) {
              if (std::abs(std::abs(r) - 2.0) < TOLERANCE) {
                rewriter.replaceOpWithNewOp<XOp>(op, op.getInputTarget(0));
                return success();
              }
              GPhaseOp::create(
                  rewriter, loc,
                  utils::constantFromScalar(rewriter, op.getLoc(),
                                            r * (-std::numbers::pi / 4.0)));
              rewriter.replaceOpWithNewOp<RXOp>(
                  op, op.getInputTarget(0),
                  utils::constantFromScalar(rewriter, op.getLoc(),
                                            r * (std::numbers::pi / 2.0)));
              return success();
            })
            // pow(r) { sxdg } => gphase(r*π/4); rx(-r*π/2)
            // pow(±2) sxdg => x
            .Case<SXdgOp>([&](auto) {
              if (std::abs(std::abs(r) - 2.0) < TOLERANCE) {
                rewriter.replaceOpWithNewOp<XOp>(op, op.getInputTarget(0));
                return success();
              }
              GPhaseOp::create(
                  rewriter, loc,
                  utils::constantFromScalar(rewriter, op.getLoc(),
                                            r * (std::numbers::pi / 4.0)));
              rewriter.replaceOpWithNewOp<RXOp>(
                  op, op.getInputTarget(0),
                  utils::constantFromScalar(rewriter, op.getLoc(),
                                            r * (-std::numbers::pi / 2.0)));
              return success();
            })
            // --- Hermitian gates (integer exponent): even => id, odd => gate
            // --- pow(n) { h } => id (n even) | h (n odd)
            .Case<HOp>([&](auto gate) {
              if (utils::isEvenExponent(r)) {
                // pow(even) { h } => identity: thread inputs to results.
                rewriter.replaceOp(op, op.getQubitsIn());
              } else {
                rewriter.replaceOp(op, gate->getResults());
              }
              return success();
            })
            // pow(n) { ecr/swap } => id (n even) | ecr/swap (n odd)
            .Case<ECROp, SWAPOp>([&](auto gate) {
              if (utils::isEvenExponent(r)) {
                rewriter.replaceOp(op, op.getQubitsIn());
              } else {
                rewriter.replaceOp(op, gate->getResults());
              }
              return success();
            })
            // --- iSWAP: decompose to parametric gate ---
            // pow(r) { iswap } => xx_plus_yy(-r*π, 0)
            // β=0: axis is aligned with XX, matching the iSWAP interaction
            // plane
            .Case<iSWAPOp>([&](auto) {
              rewriter.replaceOpWithNewOp<XXPlusYYOp>(
                  op, op.getInputTarget(0), op.getInputTarget(1),
                  utils::constantFromScalar(rewriter, op.getLoc(),
                                            r * (-std::numbers::pi)),
                  utils::constantFromScalar(rewriter, op.getLoc(), 0.0));
              return success();
            })
            // --- Identity and barrier: pass through unchanged ---
            // pow(r) { id } => id
            .Case<IdOp>([&](auto) {
              rewriter.replaceOpWithNewOp<IdOp>(op, op.getInputTarget(0));
              return success();
            })
            // pow(r) { barrier } => barrier
            .Case<BarrierOp>([&](auto) {
              rewriter.replaceOpWithNewOp<BarrierOp>(op, op.getQubitsIn());
              return success();
            })
            .Default([](auto*) -> LogicalResult {
              llvm_unreachable("unhandled gate type after pre-check");
              return failure(); // unreachable — satisfies compiler
            });
    if (innerOp->use_empty()) {
      rewriter.eraseOp(innerOp);
    }
    return result;
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

    rewriter.replaceOp(op, op.getInputQubits());
    return success();
  }
};

} // namespace

double PowOp::getExponentValue() {
  FloatAttr attr;
  if (!matchPattern(getExponent(), m_Constant(&attr))) {
    llvm::reportFatalUsageError("PowOp exponent must be a constant");
  }
  return attr.getValueAsDouble();
}

size_t PowOp::getNumBodyUnitaries() {
  return utils::getNumBodyUnitaries<UnitaryOpInterface>(*getBody());
}

UnitaryOpInterface PowOp::getBodyUnitary(const size_t i) {
  return utils::getBodyUnitary<UnitaryOpInterface>(*getBody(), i);
}

Value PowOp::getInputQubit(const size_t i) {
  if (i >= getNumTargets()) {
    llvm::reportFatalUsageError("Qubit index out of bounds");
  }
  return getQubitsIn()[i];
}

Value PowOp::getOutputQubit(const size_t i) {
  if (i >= getNumTargets()) {
    llvm::reportFatalUsageError("Qubit index out of bounds");
  }
  return getQubitsOut()[i];
}

Value PowOp::getInputForOutput(Value output) {
  if (const auto result = dyn_cast<OpResult>(output);
      result && result.getOwner() == getOperation()) {
    return getInputQubit(result.getResultNumber());
  }
  llvm::reportFatalUsageError("Given qubit is not an output of the operation");
}

Value PowOp::getOutputForInput(Value input) {
  for (auto [in, out] : llvm::zip_equal(getInputQubits(), getOutputQubits())) {
    if (in == input) {
      return out;
    }
  }
  llvm::reportFatalUsageError("Given qubit is not an input of the operation");
}

void PowOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                  ValueRange qubits,
                  const std::variant<double, Value>& exponent) {
  auto expValue = variantToValue(odsBuilder, odsState.location, exponent);
  build(odsBuilder, odsState, qubits.getTypes(), expValue, qubits);
}

void PowOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                  ValueRange qubits,
                  const std::variant<double, Value>& exponent,
                  function_ref<SmallVector<Value>(ValueRange)> bodyBuilder) {
  build(odsBuilder, odsState, qubits, exponent);
  auto& block = odsState.regions.front()->emplaceBlock();

  const auto qubitType = QubitType::get(odsBuilder.getContext());
  for (size_t i = 0; i < qubits.size(); ++i) {
    block.addArgument(qubitType, odsState.location);
  }

  const OpBuilder::InsertionGuard guard(odsBuilder);
  odsBuilder.setInsertionPointToStart(&block);
  YieldOp::create(odsBuilder, odsState.location,
                  bodyBuilder(block.getArguments()));
}

LogicalResult PowOp::verify() {
  FloatAttr attr;
  if (!matchPattern(getExponent(), m_Constant(&attr))) {
    return emitOpError("exponent must be a constant");
  }

  auto& block = *getBody();
  // The body may contain an arbitrary number of unitary (and classical) ops;
  // only non-unitary quantum operations are disallowed.
  if (llvm::any_of(block, [](Operation& op) {
        return isa<AllocOp, SinkOp, MeasureOp, ResetOp, qtensor::ExtractOp,
                   qtensor::InsertOp>(op);
      })) {
    return emitOpError("body must not contain non-unitary quantum operations "
                       "or modify a quantum register");
  }
  const auto numTargets = getNumTargets();
  if (block.getArguments().size() != numTargets) {
    return emitOpError(
        "number of block arguments must match the number of targets");
  }
  const auto qubitType = QubitType::get(getContext());
  for (size_t i = 0; i < numTargets; ++i) {
    if (block.getArgument(i).getType() != qubitType) {
      return emitOpError("block argument type at index ")
             << i << " does not match target type";
    }
  }
  auto* blockTerminator = block.getTerminator();
  if (const auto numYieldOperands = blockTerminator->getNumOperands();
      numYieldOperands != numTargets) {
    return emitOpError("yield operation must yield ")
           << numTargets << " values, but found " << numYieldOperands;
  }

  SmallPtrSet<Value, 4> uniqueQubitsIn;
  for (const auto& target : getQubitsIn()) {
    if (!uniqueQubitsIn.insert(target).second) {
      return emitOpError("duplicate qubit found");
    }
  }

  return success();
}

void PowOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<InlinePow1, ErasePow0, FoldPowIntoGate, MergeNestedPow,
              MoveCtrlOutside, NegPowToInvPow, EraseEmptyPow>(context);
}

bool PowOp::hasCompileTimeKnownUnitaryMatrix() {
  return all_of(getBody()->getOps<UnitaryOpInterface>(),
                [](UnitaryOpInterface op) {
                  return op.hasCompileTimeKnownUnitaryMatrix();
                });
}

std::optional<DynamicMatrix> PowOp::getUnitaryMatrix() {
  auto bodyUnitary = utils::getSoleBodyUnitary<UnitaryOpInterface>(*getBody());
  if (!bodyUnitary) {
    return std::nullopt;
  }
  auto&& targetMatrix = bodyUnitary.getUnitaryMatrix<DynamicMatrix>();
  if (!targetMatrix) {
    return std::nullopt;
  }

  const double p = getExponentValue();

  // U^1 = U (no computation needed)
  if (std::abs(p - 1.0) < TOLERANCE) {
    return targetMatrix;
  }

  // U^0 = I
  if (std::abs(p) < TOLERANCE) {
    return DynamicMatrix::identity(targetMatrix->cols());
  }

  // General case requires eigendecomposition which is not supported yet.
  return std::nullopt;
}
