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
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/Utils/Utils.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cstddef>
#include <iterator>

using namespace mlir;
using namespace mlir::qc;

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

    if (op.getNumBodyUnitaries() != 1) {
      return failure();
    }
    auto innerCtrlOp = dyn_cast<CtrlOp>(op.getBodyUnitary(0).getOperation());
    if (!innerCtrlOp) {
      return failure();
    }

    auto outerControls = op.getControls();
    auto outerTargets = op.getTargets();
    auto innerTargets = innerCtrlOp.getTargets();

    SmallVector<Value> controls;
    SmallVector<Value> targets;
    llvm::append_range(controls, outerControls);
    for (auto [arg, qubit] :
         llvm::zip_equal(op.getBody()->getArguments(), outerTargets)) {
      if (llvm::is_contained(innerTargets, arg)) {
        targets.push_back(qubit);
      } else {
        controls.push_back(qubit);
      }
    }

    rewriter.replaceOpWithNewOp<CtrlOp>(
        op, controls, targets, [&](ValueRange targetArgs) {
          auto* innerCtrlBody = innerCtrlOp.getBody();
          IRMapping mapping;
          utils::populateMapping(mapping, *innerCtrlBody, innerTargets,
                                 outerTargets, targets, targetArgs);
          for (auto& op : innerCtrlBody->without_terminator()) {
            rewriter.clone(op, mapping);
          }
        });

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
    if (op.getNumBodyUnitaries() != 1) {
      return failure();
    }
    auto* innerOp = op.getBodyUnitary(0).getOperation();

    // Inline ops from empty control modifiers, IdOp and BarrierOp
    if (op.getNumControls() == 0 || isa<IdOp, BarrierOp>(innerOp)) {
      const auto numTargets = op.getNumTargets();
      auto outerTargets = op.getTargets();
      SmallVector<Value> targets;
      for (auto target : innerOp->getOperands().take_front(numTargets)) {
        targets.push_back(
            utils::getValueFromBlockArgument(target, outerTargets));
      }

      rewriter.moveOpBefore(innerOp, op);
      innerOp->setOperands(0, numTargets, targets);
      rewriter.eraseOp(op);
      return success();
    }

    // The remaining code explicitly handles GPhaseOp and nothing else
    auto gPhaseOp = dyn_cast<GPhaseOp>(innerOp);
    if (!gPhaseOp) {
      return failure();
    }

    // Special case for single control: replace with a single POp
    if (op.getNumControls() == 1) {
      rewriter.replaceOpWithNewOp<POp>(op, op.getControl(0),
                                       gPhaseOp.getTheta());
      return success();
    }

    // Adjust the segment sizes of the control and target operands
    const auto opSegmentsAttrName = CtrlOp::getOperandSegmentSizeAttr();
    auto segmentsAttr =
        op->getAttrOfType<DenseI32ArrayAttr>(opSegmentsAttrName);
    auto newSegments = DenseI32ArrayAttr::get(
        rewriter.getContext(), {segmentsAttr[0] - 1, segmentsAttr[1] + 1});
    op->setAttr(opSegmentsAttrName, newSegments);

    // Add a block argument for the target qubit
    auto arg = op.getBody()->addArgument(QubitType::get(rewriter.getContext()),
                                         op.getLoc());

    // Replace the current GPhaseOp with a PhaseOp
    const OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(gPhaseOp);
    POp::create(rewriter, gPhaseOp.getLoc(), arg, gPhaseOp.getTheta());
    rewriter.eraseOp(gPhaseOp);

    return success();
  }
};

/**
 * @brief Erase control modifiers that do not have any body unitaries.
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

size_t CtrlOp::getNumBodyUnitaries() {
  return llvm::count_if(
      *getBody(), [](Operation& op) { return isa<UnitaryOpInterface>(op); });
}

UnitaryOpInterface CtrlOp::getBodyUnitary(const size_t i) {
  auto unitaries = llvm::make_filter_range(
      *getBody(), [](Operation& op) { return isa<UnitaryOpInterface>(op); });
  auto it = std::next(unitaries.begin(), static_cast<std::ptrdiff_t>(i));
  if (it == unitaries.end()) {
    llvm::reportFatalUsageError("Unitary index out of bounds");
  }
  return cast<UnitaryOpInterface>(*it);
}

Value CtrlOp::getQubit(const size_t i) {
  const auto numControls = getNumControls();
  if (i < numControls) {
    return getControls()[i];
  }
  if (numControls <= i && i < getNumQubits()) {
    return getTarget(i - numControls);
  }
  llvm::reportFatalUsageError("Qubit index out of bounds");
}

Value CtrlOp::getControl(const size_t i) {
  if (i >= getNumControls()) {
    llvm::reportFatalUsageError("Control index out of bounds");
  }
  return getControls()[i];
}

void CtrlOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                   ValueRange controls, ValueRange targets,
                   const function_ref<void(ValueRange)>& body) {
  build(odsBuilder, odsState, controls, targets);
  auto& block = odsState.regions.front()->emplaceBlock();

  auto qubitType = QubitType::get(odsBuilder.getContext());
  for (size_t i = 0; i < targets.size(); ++i) {
    block.addArgument(qubitType, odsState.location);
  }

  const OpBuilder::InsertionGuard guard(odsBuilder);
  odsBuilder.setInsertionPointToStart(&block);
  body(block.getArguments());
  YieldOp::create(odsBuilder, odsState.location);
}

LogicalResult CtrlOp::verify() {
  auto& block = *getBody();
  if (llvm::any_of(*getBody(), [](Operation& op) {
        return isa<AllocOp, DeallocOp, MeasureOp, ResetOp>(op);
      })) {
    return emitOpError("body must not contain non-unitary quantum operations");
  }
  if (!isa<YieldOp>(block.back())) {
    return emitOpError(
        "last operation in body region must be a yield operation");
  }

  SmallPtrSet<Value, 4> uniqueQubits;
  for (const auto& control : getControls()) {
    if (!uniqueQubits.insert(control).second) {
      return emitOpError("duplicate control qubit found");
    }
  }
  for (const auto& target : getTargets()) {
    if (!uniqueQubits.insert(target).second) {
      return emitOpError("duplicate target qubit found");
    }
  }

  return success();
}

void CtrlOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results.add<MergeNestedCtrl, ReduceCtrl, EraseEmptyCtrl>(context);
}
