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
#include "mlir/Dialect/MQT/Utils/Modifiers.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCInterfaces.h"
#include "mlir/Dialect/QC/IR/QCOps.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cstddef>
#include <cstdint>

using namespace mlir;
using namespace mlir::qc;

namespace {

/// Merge nested control modifiers into a single one.
struct MergeNestedCtrl final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    // Require at least one control
    // Trivial case is handled by ReduceCtrl
    if (op.getNumControls() == 0) {
      return failure();
    }

    auto inner = mqt::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
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
      controls.push_back(mqt::getValueFromBlockArgument(control, outerTargets));
    }
    const auto targets =
        llvm::map_to_vector(innerCtrlOp.getTargets(), [&](Value t) {
          return mqt::getValueFromBlockArgument(t, outerTargets);
        });

    mqt::hoistSupportingOpsBefore(*op.getBody(), innerCtrlOp, op, rewriter);

    CtrlOp::create(rewriter, op.getLoc(), controls, targets,
                   [&](ValueRange mergedTargets) {
                     mqt::inlineBodyReturningYields(*innerCtrlOp.getBody(),
                                                    mergedTargets, rewriter);
                   });
    rewriter.eraseOp(op);
    return success();
  }
};

/// Reduce controls for well-known gates.
///
/// Removes empty control ops and handles controlled IdOp, GPhaseOp and
/// BarrierOp.
struct ReduceCtrl final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getNumControls() == 0) {
      mqt::inlineModifierBody(op, *op.getBody(), op.getTargets(), rewriter);
      return success();
    }

    auto inner = mqt::getSoleBodyUnitary<UnitaryOpInterface>(*op.getBody());
    if (!inner) {
      return failure();
    }
    auto* innerOp = inner.getOperation();

    // Control does not change an identity gate or barrier.
    if (isa<IdOp, BarrierOp>(innerOp)) {
      mqt::inlineModifierBody(op, *op.getBody(), op.getTargets(), rewriter);
      return success();
    }

    // The remaining code explicitly handles GPhaseOp and nothing else
    auto gPhaseOp = dyn_cast<GPhaseOp>(innerOp);
    if (!gPhaseOp) {
      return failure();
    }

    mqt::hoistSupportingOpsBefore(*op.getBody(), gPhaseOp, op, rewriter);

    // Special case for single control: replace with a single POp
    if (op.getNumControls() == 1) {
      rewriter.replaceOpWithNewOp<POp>(op, op.getControl(0),
                                       gPhaseOp.getTheta());
      return success();
    }

    // The phase acts on the last control. The original targets are unused.
    rewriter.replaceOpWithNewOp<CtrlOp>(
        op, op.getControls().drop_back(), op.getControls().back(),
        [&](Value target) {
          POp::create(rewriter, gPhaseOp.getLoc(), target, gPhaseOp.getTheta());
        });

    return success();
  }
};

/// Erase control modifiers without unitary operations in the body.
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

/// Drop the target qubits that the body does not use.
struct DropUnusedTargets final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    auto* body = op.getBody();
    const auto used = qc::detail::getUsedQubitIndices(*body);
    if (used.size() == op.getNumTargets()) {
      return failure();
    }

    const auto targets = llvm::map_to_vector(
        used, [&](const size_t index) { return op.getTargets()[index]; });
    CtrlOp::create(rewriter, op.getLoc(), op.getControls(), targets,
                   [&](ValueRange args) {
                     qc::detail::inlineNarrowedBody(*body, op.getTargets(),
                                                    used, args, rewriter);
                   });
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

size_t CtrlOp::getNumBodyUnitaries() {
  return mqt::getNumBodyUnitaries<UnitaryOpInterface>(*getBody());
}

UnitaryOpInterface CtrlOp::getBodyUnitary(const size_t i) {
  return mqt::getBodyUnitary<UnitaryOpInterface>(*getBody(), i);
}

void CtrlOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                   ValueRange controls, ValueRange targets,
                   const function_ref<void(ValueRange)>& body) {
  build(odsBuilder, odsState, controls, targets);
  mqt::buildModifierBody<QubitType>(odsBuilder, odsState, targets.size(),
                                    [&](OpBuilder& builder, Block& block) {
                                      body(block.getArguments());
                                      YieldOp::create(builder,
                                                      odsState.location);
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
  mqt::buildModifierBody<QubitType>(
      odsBuilder, odsState, 1, [&](OpBuilder& builder, Block& block) {
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
  for (auto qubit : getQubits()) {
    if (!uniqueQubits.insert(qubit).second) {
      return emitOpError("duplicate qubit found");
    }
  }

  return success();
}

void CtrlOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results.add<MergeNestedCtrl, ReduceCtrl, EraseEmptyCtrl, DropUnusedTargets>(
      context);
}
