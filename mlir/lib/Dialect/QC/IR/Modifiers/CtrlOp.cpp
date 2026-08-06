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
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCInterfaces.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cassert>
#include <cstddef>
#include <cstdint>

using namespace mlir;
using namespace mlir::qc;

/**
 * @brief Materialize a global phase controlled by @p controls.
 */
static void createControlledPhase(PatternRewriter& rewriter,
                                  const Location controlledLoc,
                                  const Location phaseLoc,
                                  const ValueRange controls,
                                  const Value theta) {
  assert(!controls.empty());
  if (controls.size() == 1) {
    POp::create(rewriter, controlledLoc, controls.front(), theta);
    return;
  }

  CtrlOp::create(
      rewriter, controlledLoc, controls.drop_back(), controls.back(),
      [&](Value target) { POp::create(rewriter, phaseLoc, target, theta); });
}

namespace {

/**
 * @brief Merge nested control modifiers into a single one.
 */
struct MergeNestedCtrl final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    // Require at least one control
    // Trivial case is handled by ReduceCtrl
    if (op.getNumControls() == 0) {
      return failure();
    }

    // Only proceed if body contains only one operation besides terminator
    if (op.getBody()->getOperations().size() != 2) {
      return failure();
    }

    auto inner = utils::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!inner) {
      return failure();
    }
    auto innerCtrlOp = dyn_cast<CtrlOp>(inner.getOperation());
    if (!innerCtrlOp) {
      return failure();
    }

    // The inner control's controls and targets are block arguments of the outer
    // body that alias outer targets. Re-resolve them to the outer qubits: inner
    // controls join the outer controls, inner targets become the merged
    // targets. Keeping the inner-target order lets the inner body be reused
    // verbatim, since its block arguments already line up with the merged
    // targets.
    auto outerTargets = op.getTargets();
    SmallVector<Value> controls(op.getControls());
    for (auto control : innerCtrlOp.getControls()) {
      controls.push_back(
          utils::getValueFromBlockArgument(control, outerTargets));
    }
    const auto targets =
        llvm::map_to_vector(innerCtrlOp.getTargets(), [&](Value t) {
          return utils::getValueFromBlockArgument(t, outerTargets);
        });

    CtrlOp::create(rewriter, op.getLoc(), controls, targets,
                   [&](ValueRange mergedTargets) {
                     utils::inlineBodyReturningYields(*innerCtrlOp.getBody(),
                                                      mergedTargets, rewriter);
                   });
    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Pull global phases out of multi-operation control modifiers.
 */
struct PullGPhaseOutOfCtrl final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getNumControls() == 0 || op.getNumBodyUnitaries() < 2) {
      return failure();
    }

    SmallVector<GPhaseOp> globalPhases;
    for (auto gphase : op.getBody()->getOps<GPhaseOp>()) {
      // Moving the phase out must not leave its angle defined in the body.
      if (gphase.getTheta().getParentBlock() == op.getBody()) {
        return failure();
      }
      globalPhases.push_back(gphase);
    }
    if (globalPhases.empty()) {
      return failure();
    }

    const OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    for (auto gphase : globalPhases) {
      createControlledPhase(rewriter, gphase.getLoc(), gphase.getLoc(),
                            op.getControls(), gphase.getTheta());
      rewriter.eraseOp(gphase);
    }
    return success();
  }
};

/**
 * @brief Reduce controls for well-known gates.
 * @details Removes empty control ops and handles controlled IdOp, GPhaseOp and
 * BarrierOp.
 */
struct ReduceCtrl final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    auto inner = utils::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!inner) {
      return failure();
    }
    auto* innerOp = inner.getOperation();

    // Inline ops from empty control modifiers, IdOp and BarrierOp
    if (op.getNumControls() == 0 || isa<IdOp, BarrierOp>(innerOp)) {
      utils::inlineModifierBody(op, *op.getBody(), op.getTargets(), rewriter);
      return success();
    }

    // The remaining code explicitly handles GPhaseOp and nothing else
    auto gPhaseOp = dyn_cast<GPhaseOp>(innerOp);
    if (!gPhaseOp) {
      return failure();
    }

    // Only proceed if the GPhaseOp is the only operation besides the terminator
    if (op.getBody()->getOperations().size() != 2) {
      return failure();
    }

    const OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);
    createControlledPhase(rewriter, op.getLoc(), gPhaseOp.getLoc(),
                          op.getControls(), gPhaseOp.getTheta());
    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * @brief Erase control modifiers without unitary operations in the body.
 */
struct EraseEmptyCtrl final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getNumBodyUnitaries() != 0) {
      return failure();
    }

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

static void
buildModifierBody(OpBuilder& odsBuilder, OperationState& odsState,
                  const size_t numBlockArgs,
                  const function_ref<void(OpBuilder&, Block&)>& emitBody) {
  auto& block = odsState.regions.front()->emplaceBlock();
  const auto qubitType = QubitType::get(odsBuilder.getContext());
  for (size_t i = 0; i < numBlockArgs; ++i) {
    block.addArgument(qubitType, odsState.location);
  }

  const OpBuilder::InsertionGuard guard(odsBuilder);
  odsBuilder.setInsertionPointToStart(&block);
  emitBody(odsBuilder, block);
}

size_t CtrlOp::getNumBodyUnitaries() {
  return utils::getNumBodyUnitaries<UnitaryOpInterface>(*getBody());
}

UnitaryOpInterface CtrlOp::getBodyUnitary(const size_t i) {
  return utils::getBodyUnitary<UnitaryOpInterface>(*getBody(), i);
}

void CtrlOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                   ValueRange controls, ValueRange targets,
                   const function_ref<void(ValueRange)>& body) {
  build(odsBuilder, odsState, controls, targets);
  buildModifierBody(odsBuilder, odsState, targets.size(),
                    [&](OpBuilder& builder, Block& block) {
                      body(block.getArguments());
                      YieldOp::create(builder, odsState.location);
                    });
}

void CtrlOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                   ValueRange controls, Value target,
                   const function_ref<void(Value)>& bodyBuilder) {
  odsState.addOperands(controls);
  odsState.addOperands(target);
  llvm::copy(
      llvm::ArrayRef<int32_t>({static_cast<int32_t>(controls.size()), 1}),
      odsState.getOrAddProperties<CtrlOp::Properties>()
          .operandSegmentSizes.begin());
  odsState.addRegion();
  buildModifierBody(odsBuilder, odsState, 1,
                    [&](OpBuilder& builder, Block& block) {
                      bodyBuilder(block.getArgument(0));
                      YieldOp::create(builder, odsState.location);
                    });
}

void CtrlOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                   Value control, Value target,
                   const function_ref<void(Value)>& bodyBuilder) {
  build(odsBuilder, odsState, ValueRange{control}, target, bodyBuilder);
}

LogicalResult CtrlOp::verify() {
  if (failed(detail::verifyModifierBody(getOperation(), *getBody()))) {
    return failure();
  }

  SmallPtrSet<Value, 4> uniqueQubits;
  for (const auto& qubit : getQubits()) {
    if (!uniqueQubits.insert(qubit).second) {
      return emitOpError("duplicate qubit found");
    }
  }

  return success();
}

void CtrlOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results.add<MergeNestedCtrl, PullGPhaseOutOfCtrl, ReduceCtrl, EraseEmptyCtrl>(
      context);
}
