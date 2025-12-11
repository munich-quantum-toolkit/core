/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
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
 * @brief Cancel nested inverse modifiers, i.e., `inv(inv(x)) => x`.
 */
struct CancelNestedInv final : OpRewritePattern<InvOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InvOp op,
                                PatternRewriter& rewriter) const override {
    auto innerUnitary = op.getBodyUnitary();
    auto innerInvOp = llvm::dyn_cast<InvOp>(innerUnitary.getOperation());
    if (!innerInvOp) {
      return failure();
    }

    // Remove both inverse operations
    auto innerInnerUnitary = innerInvOp.getBodyUnitary();
    auto* clonedOp = rewriter.clone(*innerInnerUnitary.getOperation());
    rewriter.replaceOp(op, clonedOp->getResults());

    return success();
  }
};

} // namespace

UnitaryOpInterface InvOp::getBodyUnitary() {
  return llvm::dyn_cast<UnitaryOpInterface>(&getBody().front().front());
}

size_t InvOp::getNumQubits() { return getNumTargets() + getNumControls(); }

size_t InvOp::getNumTargets() { return getTargetsIn().size(); }

size_t InvOp::getNumControls() { return getBodyUnitary().getNumControls(); }

Value InvOp::getInputQubit(const size_t i) {
  return getBodyUnitary().getInputQubit(i);
}

Value InvOp::getOutputQubit(const size_t i) {
  return getBodyUnitary().getOutputQubit(i);
}

Value InvOp::getInputTarget(const size_t i) {
  if (i >= getNumTargets()) {
    llvm::reportFatalUsageError("Target index out of bounds");
  }
  return getTargetsIn()[i];
}

Value InvOp::getOutputTarget(const size_t i) {
  if (i >= getNumTargets()) {
    llvm::reportFatalUsageError("Target index out of bounds");
  }
  return getTargetsOut()[i];
}

Value InvOp::getInputControl(const size_t i) {
  return getBodyUnitary().getInputControl(i);
}

Value InvOp::getOutputControl(const size_t i) {
  return getBodyUnitary().getOutputControl(i);
}

Value InvOp::getInputForOutput(Value output) {
  for (size_t i = 0; i < getNumTargets(); ++i) {
    if (output == getTargetsOut()[i]) {
      return getTargetsIn()[i];
    }
  }
  llvm::reportFatalUsageError("Given qubit is not an output of the operation");
}

Value InvOp::getOutputForInput(Value input) {
  for (size_t i = 0; i < getNumTargets(); ++i) {
    if (input == getTargetsIn()[i]) {
      return getTargetsOut()[i];
    }
  }
  llvm::reportFatalUsageError("Given qubit is not an input of the operation");
}

size_t InvOp::getNumParams() { return getBodyUnitary().getNumParams(); }

Value InvOp::getParameter(const size_t i) {
  return getBodyUnitary().getParameter(i);
}

void InvOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                  const ValueRange targets, UnitaryOpInterface bodyUnitary) {
  build(odsBuilder, odsState, targets);
  auto& block = odsState.regions.front()->emplaceBlock();

  // Move the unitary op into the block
  const OpBuilder::InsertionGuard guard(odsBuilder);
  odsBuilder.setInsertionPointToStart(&block);
  auto* op = odsBuilder.clone(*bodyUnitary.getOperation());
  YieldOp::create(odsBuilder, odsState.location, op->getResults());
}

void InvOp::build(
    OpBuilder& odsBuilder, OperationState& odsState, const ValueRange targets,
    const std::function<ValueRange(OpBuilder&, ValueRange)>& bodyBuilder) {
  build(odsBuilder, odsState, targets);
  auto& block = odsState.regions.front()->emplaceBlock();

  // Move the unitary op into the block
  const OpBuilder::InsertionGuard guard(odsBuilder);
  odsBuilder.setInsertionPointToStart(&block);
  auto targetsOut = bodyBuilder(odsBuilder, targets);
  YieldOp::create(odsBuilder, odsState.location, targetsOut);
}

LogicalResult InvOp::verify() {
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
  auto bodyUnitary = getBodyUnitary();
  const auto numQubits = bodyUnitary.getNumQubits();
  for (size_t i = 0; i < numQubits; i++) {
    if (!uniqueQubitsIn.insert(bodyUnitary.getInputQubit(i)).second) {
      return emitOpError("duplicate qubit found");
    }
  }
  SmallPtrSet<Value, 4> uniqueQubitsOut;
  for (size_t i = 0; i < numQubits; i++) {
    if (!uniqueQubitsOut.insert(bodyUnitary.getOutputQubit(i)).second) {
      return emitOpError("duplicate qubit found");
    }
  }

  if (llvm::isa<BarrierOp>(bodyUnitary.getOperation())) {
    return emitOpError("BarrierOp cannot be inverted");
  }

  return success();
}

void InvOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<CancelNestedInv>(context);
}
