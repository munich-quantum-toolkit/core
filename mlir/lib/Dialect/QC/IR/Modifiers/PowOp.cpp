/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/Utils/Angles.h"
#include "mqt/Dialect/MQT/Utils/ConstantFolding.h"
#include "mqt/Dialect/MQT/Utils/GatePowering.h"
#include "mqt/Dialect/MQT/Utils/Modifiers.h"
#include "mqt/Dialect/MQT/Utils/Parameters.h"
#include "mqt/Dialect/MQT/Utils/PowerFolding.h"
#include "mqt/Dialect/QC/IR/QCDialect.h"
#include "mqt/Dialect/QC/IR/QCInterfaces.h"
#include "mqt/Dialect/QC/IR/QCOps.h"

#include "ModifierUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"

#include <cstddef>
#include <optional>
#include <tuple>
#include <variant>

using namespace mlir;
using namespace mlir::qc;
using namespace mlir::mqt;

namespace {

/// The QC operations that power folding recognizes or creates.
struct PowerFoldingOps {
  using PowOp = qc::PowOp;
  using UnitaryOpInterface = qc::UnitaryOpInterface;
  using GPhaseOp = qc::GPhaseOp;
  using IdOp = qc::IdOp;
  using BarrierOp = qc::BarrierOp;
  using XOp = qc::XOp;
  using YOp = qc::YOp;
  using ZOp = qc::ZOp;
  using HOp = qc::HOp;
  using SOp = qc::SOp;
  using SdgOp = qc::SdgOp;
  using TOp = qc::TOp;
  using TdgOp = qc::TdgOp;
  using SXOp = qc::SXOp;
  using SXdgOp = qc::SXdgOp;
  using RXOp = qc::RXOp;
  using RYOp = qc::RYOp;
  using RZOp = qc::RZOp;
  using POp = qc::POp;
  using ROp = qc::ROp;
  using UOp = qc::UOp;
  using RXXOp = qc::RXXOp;
  using RYYOp = qc::RYYOp;
  using RZXOp = qc::RZXOp;
  using RZZOp = qc::RZZOp;
  using XXPlusYYOp = qc::XXPlusYYOp;
  using XXMinusYYOp = qc::XXMinusYYOp;
  using iSWAPOp = qc::iSWAPOp;
};

/// pow(p) with p < 0 → pow(-p) { inv(q) { U } }
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
                          std::ignore = mqt::inlineBodyReturningYields(
                              *op.getBody(), invArgs, rewriter);
                        });
        });
    return success();
  }
};

/// pow(a) { pow(b) { U } } → pow(a*b) { U }
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
          std::ignore = mqt::inlineBodyReturningYields(*innerPow.getBody(),
                                                       powArgs, rewriter);
        });
    return success();
  }
};

/// pow(p) { ctrl(q) { U } } → ctrl(q) { pow(p) { U } }
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
                          std::ignore = mqt::inlineBodyReturningYields(
                              *innerCtrlOp.getBody(), powArgs, rewriter);
                        });
        });

    return success();
  }
};

/// Erase power modifiers that do not have any body unitaries.
struct EraseEmptyPow final : OpRewritePattern<PowOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PowOp op,
                                PatternRewriter& rewriter) const override {
    if (!op.getBody()->getOps<UnitaryOpInterface>().empty()) {
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

void PowOp::getLoweringPatterns(RewritePatternSet& results,
                                MLIRContext* context) {
  results.add(mqt::foldTrivialPower<UnitaryOpInterface, PowOp>);
  results.add<FoldPowIntoGate<PowerFoldingOps>, MergeNestedPow,
              MoveCtrlOutsidePow, NegPowToInvPow>(context);
  getCanonicalizationPatterns(results, context);
}

void PowOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<EraseEmptyPow, DropUnusedPowQubits>(context);
}
