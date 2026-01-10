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

#include <cstddef>
#include <functional>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qco;

namespace {

/**
 * @brief Merge nested control modifiers into a single one.
 */
struct MergeNestedCtrl final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    // Require at least one positive control
    // Trivial case is handled by RemoveTrivialCtrl
    if (op.getNumControls() == 0) {
      return failure();
    }

    auto bodyCtrlOp =
        llvm::dyn_cast<CtrlOp>(op.getBodyUnitary().getOperation());
    if (!bodyCtrlOp) {
      return failure();
    }

    // Merge controls
    SmallVector<Value> newControls(op.getControlsIn());
    for (const auto control : bodyCtrlOp.getControlsIn()) {
      newControls.push_back(control);
    }

    rewriter.replaceOpWithNewOp<CtrlOp>(op, newControls, op.getTargetsIn(),
                                        bodyCtrlOp.getBodyUnitary());

    return success();
  }
};

/**
 * @brief Remove control modifiers without controls.
 */
struct RemoveTrivialCtrl final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getNumControls() > 0) {
      return failure();
    }

    const OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);

    auto* clonedBody = rewriter.clone(*op.getBodyUnitary().getOperation());
    rewriter.replaceOp(op, clonedBody->getResults());

    return success();
  }
};

/**
 * @brief Inline controlled GPhase operations.
 */
struct CtrlInlineGPhase final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    // Require at least one positive control
    // Trivial case is handled by RemoveTrivialCtrl
    if (op.getNumControls() == 0) {
      return failure();
    }

    auto gPhaseOp =
        llvm::dyn_cast<GPhaseOp>(op.getBodyUnitary().getOperation());
    if (!gPhaseOp) {
      return failure();
    }

    SmallVector<Value> newControls(op.getControlsIn());
    const auto newTarget = newControls.back();
    newControls.pop_back();
    auto ctrlOp = CtrlOp::create(rewriter, op.getLoc(), newControls, newTarget,
                                 [&](ValueRange targets) -> SmallVector<Value> {
                                   auto pOp = POp::create(rewriter, op.getLoc(),
                                                          targets[0],
                                                          gPhaseOp.getTheta());
                                   return {pOp.getQubitOut()};
                                 });

    rewriter.replaceOp(op, ctrlOp.getResults());

    return success();
  }
};

/**
 * @brief Inline controlled identity operations.
 */
struct CtrlInlineId final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    // Require at least one positive control
    // Trivial case is handled by RemoveTrivialCtrl
    if (op.getNumControls() == 0) {
      return failure();
    }

    if (!llvm::isa<IdOp>(op.getBodyUnitary().getOperation())) {
      return failure();
    }

    auto idOp = rewriter.create<IdOp>(op.getLoc(), op.getTargetsIn().front());

    SmallVector<Value> newOperands;
    newOperands.reserve(op.getNumControls() + 1);
    newOperands.append(op.getControlsIn().begin(), op.getControlsIn().end());
    newOperands.push_back(idOp.getQubitOut());
    rewriter.replaceOp(op, newOperands);

    return success();
  }
};

} // namespace

UnitaryOpInterface CtrlOp::getBodyUnitary() {
  return llvm::dyn_cast<UnitaryOpInterface>(&getBody()->front());
}

size_t CtrlOp::getNumQubits() { return getNumTargets() + getNumControls(); }

size_t CtrlOp::getNumTargets() { return getTargetsIn().size(); }

size_t CtrlOp::getNumControls() { return getControlsIn().size(); }

Value CtrlOp::getInputQubit(const size_t i) {
  const auto numControls = getNumControls();
  if (i < numControls) {
    return getControlsIn()[i];
  }
  if (numControls <= i && i < getNumQubits()) {
    return getBodyUnitary().getInputQubit(i - numControls);
  }
  llvm::reportFatalUsageError("Invalid qubit index");
}

Value CtrlOp::getOutputQubit(const size_t i) {
  const auto numControls = getNumControls();
  if (i < numControls) {
    return getControlsOut()[i];
  }
  if (numControls <= i && i < getNumQubits()) {
    return getBodyUnitary().getOutputQubit(i - numControls);
  }
  llvm::reportFatalUsageError("Invalid qubit index");
}

Value CtrlOp::getInputTarget(const size_t i) {
  if (i >= getNumTargets()) {
    llvm::reportFatalUsageError("Target index out of bounds");
  }
  return getTargetsIn()[i];
}

Value CtrlOp::getOutputTarget(const size_t i) {
  if (i >= getNumTargets()) {
    llvm::reportFatalUsageError("Target index out of bounds");
  }
  return getTargetsOut()[i];
}

Value CtrlOp::getInputControl(const size_t i) {
  if (i >= getNumControls()) {
    llvm::reportFatalUsageError("Control index out of bounds");
  }
  return getControlsIn()[i];
}

Value CtrlOp::getOutputControl(const size_t i) {
  if (i >= getNumControls()) {
    llvm::reportFatalUsageError("Control index out of bounds");
  }
  return getControlsOut()[i];
}

Value CtrlOp::getInputForOutput(Value output) {
  for (size_t i = 0; i < getNumControls(); ++i) {
    if (output == getControlsOut()[i]) {
      return getControlsIn()[i];
    }
  }
  for (size_t i = 0; i < getNumTargets(); ++i) {
    if (output == getTargetsOut()[i]) {
      return getTargetsIn()[i];
    }
  }
  llvm::reportFatalUsageError("Given qubit is not an output of the operation");
}

Value CtrlOp::getOutputForInput(Value input) {
  for (size_t i = 0; i < getNumControls(); ++i) {
    if (input == getControlsIn()[i]) {
      return getControlsOut()[i];
    }
  }
  for (size_t i = 0; i < getNumTargets(); ++i) {
    if (input == getTargetsIn()[i]) {
      return getTargetsOut()[i];
    }
  }
  llvm::reportFatalUsageError("Given qubit is not an input of the operation");
}

size_t CtrlOp::getNumParams() { return getBodyUnitary().getNumParams(); }

Value CtrlOp::getParameter(const size_t i) {
  return getBodyUnitary().getParameter(i);
}

void CtrlOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                   ValueRange controls, ValueRange targets,
                   UnitaryOpInterface bodyUnitary) {
  build(odsBuilder, odsState, controls, targets);
  auto& block = odsState.regions.front()->emplaceBlock();
  for (const auto target : targets) {
    block.addArgument(target.getType(), odsState.location);
  }
  auto blockArgs = block.getArguments();

  // Move the unitary op into the block
  const OpBuilder::InsertionGuard guard(odsBuilder);
  odsBuilder.setInsertionPointToStart(&block);
  auto* op = odsBuilder.clone(*bodyUnitary.getOperation());
  for (size_t i = 0; i < targets.size(); ++i) {
    op->replaceUsesOfWith(targets[i], blockArgs[i]);
  }
  YieldOp::create(odsBuilder, odsState.location, op->getResults());
}

void CtrlOp::build(
    OpBuilder& odsBuilder, OperationState& odsState, ValueRange controls,
    ValueRange targets,
    const std::function<SmallVector<Value>(ValueRange)>& bodyBuilder) {
  build(odsBuilder, odsState, controls, targets);
  auto& block = odsState.regions.front()->emplaceBlock();
  for (const auto target : targets) {
    block.addArgument(target.getType(), odsState.location);
  }

  // Move the unitary op into the block
  const OpBuilder::InsertionGuard guard(odsBuilder);
  odsBuilder.setInsertionPointToStart(&block);
  const auto targetsOut = bodyBuilder(block.getArguments());
  YieldOp::create(odsBuilder, odsState.location, targetsOut);
}

LogicalResult CtrlOp::verify() {
  auto& block = *getBody();
  if (block.getOperations().size() != 2) {
    return emitOpError("body region must have exactly two operations");
  }
  if (block.getArguments().size() != getNumTargets()) {
    return emitOpError(
        "number of block arguments must match number of targets");
  }
  for (size_t i = 0; i < getNumTargets(); ++i) {
    if (block.getArgument(i).getType() != getTargetsIn()[i].getType()) {
      return emitOpError("block argument type at index ")
             << i << " does not match target type";
    }
  }
  if (!llvm::isa<UnitaryOpInterface>(block.front())) {
    return emitOpError(
        "first operation in body region must be a unitary operation");
  }
  if (!llvm::isa<YieldOp>(block.back())) {
    return emitOpError(
        "second operation in body region must be a yield operation");
  }
  if (block.back().getNumOperands() != getNumTargets()) {
    return emitOpError("yield operation must yield ")
           << getNumTargets() << " values, but found "
           << block.back().getNumOperands();
  }

  SmallPtrSet<Value, 4> uniqueQubitsIn;
  for (const auto& control : getControlsIn()) {
    if (!uniqueQubitsIn.insert(control).second) {
      return emitOpError("duplicate control qubit found");
    }
  }
  auto bodyUnitary = getBodyUnitary();
  const auto numQubits = bodyUnitary.getNumQubits();
  for (size_t i = 0; i < numQubits; i++) {
    if (!uniqueQubitsIn.insert(bodyUnitary.getInputQubit(i)).second) {
      return emitOpError("duplicate qubit found");
    }
  }
  SmallPtrSet<Value, 4> uniqueQubitsOut;
  for (const auto& control : getControlsOut()) {
    if (!uniqueQubitsOut.insert(control).second) {
      return emitOpError("duplicate control qubit found");
    }
  }
  for (size_t i = 0; i < numQubits; i++) {
    if (!uniqueQubitsOut.insert(bodyUnitary.getOutputQubit(i)).second) {
      return emitOpError("duplicate qubit found");
    }
  }

  if (llvm::isa<BarrierOp>(bodyUnitary.getOperation())) {
    return emitOpError("BarrierOp cannot be controlled");
  }

  return success();
}

void CtrlOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results
      .add<MergeNestedCtrl, RemoveTrivialCtrl, CtrlInlineGPhase, CtrlInlineId>(
          context);
}
