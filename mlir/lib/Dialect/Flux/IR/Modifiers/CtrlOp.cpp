/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Flux/IR/FluxDialect.h"

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
using namespace mlir::flux;

namespace {

/**
 * @brief Merge nested control modifiers into a single one.
 */
struct MergeNestedCtrl final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    auto bodyUnitary = op.getBodyUnitary();
    auto bodyCtrlOp = llvm::dyn_cast<CtrlOp>(bodyUnitary.getOperation());
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
    if (op.getNumPosControls() > 0) {
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
    auto gPhaseOp =
        llvm::dyn_cast<GPhaseOp>(op.getBodyUnitary().getOperation());
    if (!gPhaseOp) {
      return failure();
    }

    SmallVector<Value> newOperands;
    newOperands.reserve(op.getNumPosControls());
    for (size_t i = 0; i < op.getNumPosControls(); ++i) {
      auto pOp = rewriter.create<POp>(op.getLoc(), op.getInputPosControl(i),
                                      gPhaseOp.getTheta());
      newOperands.push_back(pOp.getQubitOut());
    }

    rewriter.replaceOp(op, newOperands);

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
    if (!llvm::isa<IdOp>(op.getBodyUnitary().getOperation())) {
      return failure();
    }

    auto idOp = rewriter.create<IdOp>(op.getLoc(), op.getTargetsIn().front());

    SmallVector<Value> newOperands;
    newOperands.reserve(op.getNumPosControls() + 1);
    newOperands.append(op.getControlsIn().begin(), op.getControlsIn().end());
    newOperands.push_back(idOp.getQubitOut());
    rewriter.replaceOp(op, newOperands);

    return success();
  }
};

} // namespace

UnitaryOpInterface CtrlOp::getBodyUnitary() {
  return llvm::dyn_cast<UnitaryOpInterface>(&getBody().front().front());
}

size_t CtrlOp::getNumQubits() { return getNumTargets() + getNumControls(); }

size_t CtrlOp::getNumTargets() { return getTargetsIn().size(); }

size_t CtrlOp::getNumControls() {
  return getNumPosControls() + getNumNegControls();
}

size_t CtrlOp::getNumPosControls() { return getControlsIn().size(); }

size_t CtrlOp::getNumNegControls() {
  return getBodyUnitary().getNumNegControls();
}

Value CtrlOp::getInputQubit(const size_t i) {
  const auto numPosControls = getNumPosControls();
  if (i < numPosControls) {
    return getControlsIn()[i];
  }
  if (numPosControls <= i && i < getNumQubits()) {
    return getBodyUnitary().getInputQubit(i - numPosControls);
  }
  llvm::reportFatalUsageError("Invalid qubit index");
}

Value CtrlOp::getOutputQubit(const size_t i) {
  const auto numPosControls = getNumPosControls();
  if (i < numPosControls) {
    return getControlsOut()[i];
  }
  if (numPosControls <= i && i < getNumQubits()) {
    return getBodyUnitary().getOutputQubit(i - numPosControls);
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

Value CtrlOp::getInputPosControl(const size_t i) {
  if (i >= getNumPosControls()) {
    llvm::reportFatalUsageError("Control index out of bounds");
  }
  return getControlsIn()[i];
}

Value CtrlOp::getOutputPosControl(const size_t i) {
  if (i >= getNumPosControls()) {
    llvm::reportFatalUsageError("Control index out of bounds");
  }
  return getControlsOut()[i];
}

Value CtrlOp::getInputNegControl(const size_t i) {
  return getBodyUnitary().getInputNegControl(i);
}

Value CtrlOp::getOutputNegControl(const size_t i) {
  return getBodyUnitary().getOutputNegControl(i);
}

Value CtrlOp::getInputForOutput(Value output) {
  for (size_t i = 0; i < getNumPosControls(); ++i) {
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
  for (size_t i = 0; i < getNumPosControls(); ++i) {
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

void CtrlOp::build(OpBuilder& builder, OperationState& state,
                   ValueRange controls, ValueRange targets,
                   UnitaryOpInterface bodyUnitary) {
  build(builder, state, controls, targets);
  auto& block = state.regions.front()->emplaceBlock();

  // Move the unitary op into the block
  const OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&block);
  auto* op = builder.clone(*bodyUnitary.getOperation());
  builder.create<YieldOp>(state.location, op->getResults());
}

void CtrlOp::build(
    OpBuilder& builder, OperationState& state, ValueRange controls,
    ValueRange targets,
    const std::function<ValueRange(OpBuilder&, ValueRange)>& bodyBuilder) {
  build(builder, state, controls, targets);
  auto& block = state.regions.front()->emplaceBlock();

  // Move the unitary op into the block
  const OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&block);
  auto targetsOut = bodyBuilder(builder, targets);
  builder.create<YieldOp>(state.location, targetsOut);
}

LogicalResult CtrlOp::verify() {
  auto& block = getBody().front();
  if (block.getOperations().size() != 2) {
    return emitOpError("body region must have exactly two operations");
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
