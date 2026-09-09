/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ModifierUtils.h"
#include "mlir/Dialect/MQT/Utils/Angles.h"
#include "mlir/Dialect/MQT/Utils/ConstantFolding.h"
#include "mlir/Dialect/MQT/Utils/GatePowering.h"
#include "mlir/Dialect/MQT/Utils/Modifiers.h"
#include "mlir/Dialect/MQT/Utils/Parameters.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCInterfaces.h"
#include "mlir/Dialect/QC/IR/QCOps.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
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
using namespace mlir::qc;
using namespace mlir::mqt;

/// Replace a power of a fixed phase gate with a named gate or a P gate.
static void replaceWithPhaseGate(double angle, PowOp op, Value target,
                                 PatternRewriter& rewriter) {
  const auto phaseGate = classifyPhaseGate(angle);
  if (!phaseGate) {
    rewriter.replaceOpWithNewOp<POp>(op, target, angle);
    return;
  }
  switch (*phaseGate) {
  case PhaseGate::Identity:
    rewriter.eraseOp(op);
    return;
  case PhaseGate::Z:
    rewriter.replaceOpWithNewOp<ZOp>(op, target);
    return;
  case PhaseGate::S:
    rewriter.replaceOpWithNewOp<SOp>(op, target);
    return;
  case PhaseGate::Sdg:
    rewriter.replaceOpWithNewOp<SdgOp>(op, target);
    return;
  case PhaseGate::T:
    rewriter.replaceOpWithNewOp<TOp>(op, target);
    return;
  case PhaseGate::Tdg:
    rewriter.replaceOpWithNewOp<TdgOp>(op, target);
    return;
  }
}

namespace {

/// pow(1.0) { U }  =>  U
struct InlinePow1 final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    const auto exponent = op.getExponentValue();
    if (!exponent ||
        std::abs(*exponent - 1.0) > PARAMETER_COMPARISON_TOLERANCE) {
      return failure();
    }
    mqt::inlineModifierBody(op, *op.getBody(), op.getQubits(), rewriter);
    return success();
  }
};

/// pow(0.0) { U }  =>  identity (no-op)
struct ErasePow0 final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    const auto exponent = op.getExponentValue();
    if (!exponent || std::abs(*exponent) > PARAMETER_COMPARISON_TOLERANCE) {
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
    if (!exponent || *exponent >= 0.0 || !mqt::isIntegerExponent(-*exponent)) {
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
                          mqt::inlineBodyReturningYields(*op.getBody(), invArgs,
                                                         rewriter);
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
    if (!outerExponent || !mqt::isIntegerExponent(*outerExponent)) {
      return failure();
    }
    auto inner = mqt::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!inner) {
      return failure();
    }
    auto innerPow = dyn_cast<PowOp>(inner.getOperation());
    if (!innerPow) {
      return failure();
    }
    const auto innerExponent = innerPow.getExponentValue();
    if (!innerExponent) {
      return failure();
    }
    const auto mergedExponent =
        scaleConstantAngle(*innerExponent, *outerExponent);
    if (!mergedExponent) {
      return failure();
    }
    // The inner pow's operands alias the outer pow's block args, possibly in a
    // different order / subset. Translate them back to the outer pow's operands
    // so the merged pow's footprint matches the inner pow positionally.
    auto outerQubits = op.getQubits();
    const auto qubits = llvm::map_to_vector(innerPow.getQubits(), [&](Value v) {
      return mqt::getValueFromBlockArgument(v, outerQubits);
    });
    // Move supporting ops (constants, arithmetic) out of the body so their
    // Values are accessible from outside and survive PowOp erasure.
    mqt::hoistSupportingOpsBefore(*op.getBody(), innerPow.getOperation(), op,
                                  rewriter);
    rewriter.replaceOpWithNewOp<PowOp>(
        op, *mergedExponent, qubits, [&](ValueRange powArgs) {
          // Inner pow body args now match the new pow's args positionally.
          mqt::inlineBodyReturningYields(*innerPow.getBody(), powArgs,
                                         rewriter);
        });
    return success();
  }
};

/// pow(p) { ctrl(q) { U } }  =>  ctrl(q) { pow(p) { U } }
struct MoveCtrlOutsidePow final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    auto inner = mqt::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
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
          return mqt::getValueFromBlockArgument(c, outerQubits);
        });
    const auto targets =
        llvm::map_to_vector(innerCtrlOp.getTargets(), [&](Value t) {
          return mqt::getValueFromBlockArgument(t, outerQubits);
        });

    mqt::hoistSupportingOpsBefore(*op.getBody(), innerCtrlOp, op, rewriter);

    rewriter.replaceOpWithNewOp<CtrlOp>(
        op, controls, targets, [&](ValueRange targetArgs) {
          PowOp::create(rewriter, op.getLoc(), op.getExponent(), targetArgs,
                        [&](ValueRange powArgs) {
                          mqt::inlineBodyReturningYields(*innerCtrlOp.getBody(),
                                                         powArgs, rewriter);
                        });
        });

    return success();
  }
};

/// Fold pow(r) around gates into simpler operations.
///
/// - Rotation gates: multiply a constant angle by an integer exponent when
///   the product meets the constant-angle rounding bound
/// - Phase/diagonal gates: named gate if angle matches, else `P` gate,
///   e.g., `pow(r) { s } => s/sdg/t/tdg/z` or `p(r*π/2)`
/// - Hermitian gates (integer exponent): even => erase, odd => gate
/// - Constant U gates (positive integer exponent): synthesize as a U gate and
///   phase
/// - Identity/barrier: pass through unchanged
struct FoldPowIntoGate final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    auto inner = mqt::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!inner) {
      return failure();
    }
    auto* innerOp = inner.getOperation();
    const auto exponent = op.getExponentValue();
    if (!exponent) {
      return failure();
    }
    double r = *exponent;
    const auto period = getFixedGatePowerPeriod(inner.getBaseSymbol());
    if (period != 0U) {
      r = std::remainder(r, static_cast<double>(period));
    }
    auto loc = op.getLoc();

    std::optional<UPowerParameters> uPower;
    if (auto uOp = dyn_cast<UOp>(innerOp)) {
      const auto theta = valueToDouble(uOp.getTheta());
      const auto phi = valueToDouble(uOp.getPhi());
      const auto lambda = valueToDouble(uOp.getLambda());
      if (!theta || !phi || !lambda) {
        return failure();
      }
      uPower = powerUParameters(*theta, *phi, *lambda, r);
      if (!uPower) {
        return failure();
      }
    }

    // Scaling a gate parameter represents a principal matrix power only for an
    // integral exponent unless the parameter is known to remain within the
    // principal branch. Keep arbitrary parameters inside fractional powers.
    std::optional<double> scaledParameter;
    if (isa<GPhaseOp, RXOp, RYOp, RZOp, POp, ROp, RXXOp, RYYOp, RZXOp, RZZOp,
            XXPlusYYOp, XXMinusYYOp>(innerOp)) {
      if (!mqt::isIntegerExponent(r)) {
        return failure();
      }
      const auto parameter = valueToDouble(inner.getParameter(0));
      if (!parameter) {
        return failure();
      }
      scaledParameter = scaleConstantAngle(*parameter, r);
      if (!scaledParameter || (isa<GPhaseOp>(innerOp) &&
                               !isValidGlobalPhaseAngle(*scaledParameter))) {
        return failure();
      }
    }
    // HOp, ECROp, RCCXOp, and SWAPOp also only have the simple parity fold for
    // integral exponents.
    if (isa<HOp, ECROp, RCCXOp, SWAPOp>(innerOp) &&
        !mqt::isIntegerExponent(r)) {
      return failure();
    }
    if (!isa<GPhaseOp, XOp, YOp, ZOp, SOp, SdgOp, TOp, TdgOp, SXOp, SXdgOp, HOp,
             ECROp, RCCXOp, SWAPOp, RXOp, RYOp, RZOp, POp, ROp, RXXOp, RYYOp,
             RZXOp, RZZOp, XXPlusYYOp, XXMinusYYOp, iSWAPOp, UOp, IdOp,
             BarrierOp>(innerOp)) {
      return failure();
    }

    // Move supporting ops (constants, arithmetic) out of the body so their
    // Values are accessible from outside and survive PowOp erasure.
    mqt::hoistSupportingOpsBefore(*op.getBody(), innerOp, op, rewriter);

    if (period != 0U) {
      if (r == 0.0) {
        rewriter.eraseOp(op);
        return success();
      }
      if (r == 1.0 || (period == 2U && r == -1.0)) {
        mqt::inlineModifierBody(op, *op.getBody(), op.getQubits(), rewriter);
        return success();
      }
    }

    Value scaledValue;
    if (scaledParameter) {
      scaledValue = constantFromScalar(rewriter, loc, *scaledParameter);
    }

    return TypeSwitch<Operation*, LogicalResult>(innerOp)
        // --- Rotation gates: multiply angle by exponent ---
        // pow(r) { gphase(θ) } => gphase(r*θ)
        .Case([&](GPhaseOp) {
          rewriter.replaceOpWithNewOp<GPhaseOp>(op, scaledValue);
          return success();
        })
        // pow(r) { rx/ry/rz/p(θ) } => rx/ry/rz/p(r*θ)
        .Case<RXOp, RYOp, RZOp, POp>([&](auto gate) {
          rewriter.replaceOpWithNewOp<decltype(gate)>(
              op,
              mqt::getValueFromBlockArgument(gate.getTarget(0), op.getQubits()),
              scaledValue);
          return success();
        })
        // pow(r) { rxx/ryy/rzx/rzz(θ) } => rxx/ryy/rzx/rzz(r*θ)
        .Case<RXXOp, RYYOp, RZXOp, RZZOp>([&](auto gate) {
          rewriter.replaceOpWithNewOp<decltype(gate)>(
              op,
              mqt::getValueFromBlockArgument(gate.getTarget(0), op.getQubits()),
              mqt::getValueFromBlockArgument(gate.getTarget(1), op.getQubits()),
              scaledValue);
          return success();
        })
        // pow(r) { r(θ, φ) } => r(r*θ, φ)
        .Case([&](ROp gate) {
          rewriter.replaceOpWithNewOp<ROp>(
              op,
              mqt::getValueFromBlockArgument(gate.getTarget(0), op.getQubits()),
              scaledValue, gate.getPhi());
          return success();
        })
        // pow(r) { xx±yy(θ, β) } => xx±yy(r*θ, β)
        .Case<XXPlusYYOp, XXMinusYYOp>([&](auto gate) {
          rewriter.replaceOpWithNewOp<decltype(gate)>(
              op,
              mqt::getValueFromBlockArgument(gate.getTarget(0), op.getQubits()),
              mqt::getValueFromBlockArgument(gate.getTarget(1), op.getQubits()),
              scaledValue, gate.getBeta());
          return success();
        })
        // pow(n) { u(theta, phi, lambda) } =>
        // gphase(delta); u(theta', phi', lambda')
        .Case([&](UOp gate) {
          if (std::abs(normalizeAngle(uPower->phase)) >
              PARAMETER_COMPARISON_TOLERANCE) {
            GPhaseOp::create(rewriter, loc, uPower->phase);
          }
          rewriter.replaceOpWithNewOp<UOp>(
              op,
              mqt::getValueFromBlockArgument(gate.getTarget(0), op.getQubits()),
              uPower->theta, uPower->phi, uPower->lambda);
          return success();
        })
        // Powers of Z, S, Sdg, T, and Tdg are phase gates.
        .Case<ZOp, SOp, SdgOp, TOp, TdgOp>([&](auto gate) {
          const double signedExponent =
              isa<SdgOp, TdgOp>(gate.getOperation()) ? -r : r;
          const double angle = signedExponent * (2.0 * std::numbers::pi /
                                                 static_cast<double>(period));
          replaceWithPhaseGate(
              angle, op,
              mqt::getValueFromBlockArgument(gate.getTarget(0), op.getQubits()),
              rewriter);
          return success();
        })
        // pow(r) { x } => gphase(r*π/2); rx(r*π)
        // pow(1/2) x => sx      (X^(1/2) = SX exactly)
        // pow(-1/2) x => sxdg   (X^(-1/2) = SXdg exactly)
        .Case([&](XOp gate) {
          if (std::abs(r - 0.5) < PARAMETER_COMPARISON_TOLERANCE) {
            rewriter.replaceOpWithNewOp<SXOp>(
                op, mqt::getValueFromBlockArgument(gate.getTarget(0),
                                                   op.getQubits()));
            return success();
          }
          if (std::abs(r + 0.5) < PARAMETER_COMPARISON_TOLERANCE) {
            rewriter.replaceOpWithNewOp<SXdgOp>(
                op, mqt::getValueFromBlockArgument(gate.getTarget(0),
                                                   op.getQubits()));
            return success();
          }
          GPhaseOp::create(
              rewriter, loc,
              mqt::constantFromScalar(rewriter, op.getLoc(),
                                      r * (std::numbers::pi / 2.0)));
          rewriter.replaceOpWithNewOp<RXOp>(
              op,
              mqt::getValueFromBlockArgument(gate.getTarget(0), op.getQubits()),
              mqt::constantFromScalar(rewriter, op.getLoc(),
                                      r * std::numbers::pi));
          return success();
        })
        // pow(r) { y } => gphase(r*π/2); ry(r*π)
        .Case([&](YOp gate) {
          GPhaseOp::create(
              rewriter, loc,
              mqt::constantFromScalar(rewriter, op.getLoc(),
                                      r * (std::numbers::pi / 2.0)));
          rewriter.replaceOpWithNewOp<RYOp>(
              op,
              mqt::getValueFromBlockArgument(gate.getTarget(0), op.getQubits()),
              mqt::constantFromScalar(rewriter, op.getLoc(),
                                      r * std::numbers::pi));
          return success();
        })
        // --- SX/SXdg gates: decompose to rotation + global phase ---
        // pow(r) { sx } => gphase(r*π/4); rx(r*π/2)
        // pow(±2) sx => x
        .Case([&](SXOp gate) {
          if (std::abs(std::abs(r) - 2.0) < PARAMETER_COMPARISON_TOLERANCE) {
            rewriter.replaceOpWithNewOp<XOp>(
                op, mqt::getValueFromBlockArgument(gate.getTarget(0),
                                                   op.getQubits()));
            return success();
          }
          GPhaseOp::create(
              rewriter, loc,
              mqt::constantFromScalar(rewriter, op.getLoc(),
                                      r * (std::numbers::pi / 4.0)));
          rewriter.replaceOpWithNewOp<RXOp>(
              op,
              mqt::getValueFromBlockArgument(gate.getTarget(0), op.getQubits()),
              mqt::constantFromScalar(rewriter, op.getLoc(),
                                      r * (std::numbers::pi / 2.0)));
          return success();
        })
        // pow(r) { sxdg } => gphase(-r*π/4); rx(-r*π/2)
        // pow(±2) sxdg => x
        .Case([&](SXdgOp gate) {
          if (std::abs(std::abs(r) - 2.0) < PARAMETER_COMPARISON_TOLERANCE) {
            rewriter.replaceOpWithNewOp<XOp>(
                op, mqt::getValueFromBlockArgument(gate.getTarget(0),
                                                   op.getQubits()));
            return success();
          }
          GPhaseOp::create(
              rewriter, loc,
              mqt::constantFromScalar(rewriter, op.getLoc(),
                                      r * (-std::numbers::pi / 4.0)));
          rewriter.replaceOpWithNewOp<RXOp>(
              op,
              mqt::getValueFromBlockArgument(gate.getTarget(0), op.getQubits()),
              mqt::constantFromScalar(rewriter, op.getLoc(),
                                      r * (-std::numbers::pi / 2.0)));
          return success();
        })
        // --- iSWAP: decompose to parametric gate ---
        // pow(r) { iswap } => xx_plus_yy(-r*π, 0)
        .Case([&](iSWAPOp gate) {
          rewriter.replaceOpWithNewOp<XXPlusYYOp>(
              op,
              mqt::getValueFromBlockArgument(gate.getTarget(0), op.getQubits()),
              mqt::getValueFromBlockArgument(gate.getTarget(1), op.getQubits()),
              mqt::constantFromScalar(rewriter, op.getLoc(),
                                      r * (-std::numbers::pi)),
              mqt::constantFromScalar(rewriter, op.getLoc(), 0.0));
          return success();
        })
        // --- Identity and barrier: pass through unchanged ---
        // pow(r) { id } => id
        .Case([&](IdOp gate) {
          rewriter.replaceOpWithNewOp<IdOp>(
              op, mqt::getValueFromBlockArgument(gate.getTarget(0),
                                                 op.getQubits()));
          return success();
        })
        // pow(r) { barrier } => barrier
        .Case([&](BarrierOp gate) {
          rewriter.replaceOpWithNewOp<BarrierOp>(
              op, llvm::map_to_vector(gate.getTargets(), [&](Value qubit) {
                return mqt::getValueFromBlockArgument(qubit, op.getQubits());
              }));
          return success();
        })
        .Default([](auto*) -> LogicalResult {
          llvm_unreachable("unhandled gate type after pre-check");
        });
  }
};

/// Erase power modifiers that do not have any body unitaries.
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

/// Drop the qubits that the body does not use.
struct DropUnusedPowQubits final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    auto* body = op.getBody();
    auto qubits = op.getQubits();
    return qc::detail::dropUnusedQubits(
        op, *body, qubits,
        [&](ValueRange narrowedQubits, ArrayRef<size_t> used) {
          PowOp::create(rewriter, op.getLoc(), op.getExponent(), narrowedQubits,
                        [&](ValueRange args) {
                          qc::detail::inlineNarrowedBody(*body, qubits, used,
                                                         args, rewriter);
                        });
        },
        rewriter);
  }
};

} // namespace

std::optional<double> PowOp::getExponentValue() {
  return mlir::mqt::valueToDouble(getExponent());
}

size_t PowOp::getNumBodyUnitaries() {
  return mqt::getNumBodyUnitaries<UnitaryOpInterface>(*getBody());
}

UnitaryOpInterface PowOp::getBodyUnitary(const size_t i) {
  return mqt::getBodyUnitary<UnitaryOpInterface>(*getBody(), i);
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

void PowOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                  const std::variant<double, Value>& exponent, Value qubit,
                  const function_ref<void(Value)>& bodyBuilder) {
  auto expValue = variantToValue(odsBuilder, odsState.location, exponent);
  odsState.addOperands(expValue);
  odsState.addOperands(qubit);
  odsState.addRegion();
  auto& block = odsState.regions.front()->emplaceBlock();
  block.addArgument(QubitType::get(odsBuilder.getContext()), odsState.location);

  const OpBuilder::InsertionGuard guard(odsBuilder);
  odsBuilder.setInsertionPointToStart(&block);
  bodyBuilder(block.getArgument(0));
  YieldOp::create(odsBuilder, odsState.location);
}

LogicalResult PowOp::verify() {
  return detail::verifyModifierBody(getOperation(), *getBody());
}

void PowOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<InlinePow1, ErasePow0, FoldPowIntoGate, MergeNestedPow,
              MoveCtrlOutsidePow, NegPowToInvPow, EraseEmptyPow,
              DropUnusedPowQubits>(context);
}
