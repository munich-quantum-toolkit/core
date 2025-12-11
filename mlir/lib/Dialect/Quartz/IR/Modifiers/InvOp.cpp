/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Quartz/IR/QuartzDialect.h"

#include <cstddef>
#include <functional>
#include <llvm/Support/Casting.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::quartz;

namespace {

/**
 * @brief Cancel nested inverse modifiers, i.e., `inv(inv(x)) => x`.
 */
struct CancelNestedInv final : OpRewritePattern<InvOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(InvOp invOp,
                                PatternRewriter& rewriter) const override {
    auto innerUnitary = invOp.getBodyUnitary();
    auto innerInvOp = llvm::dyn_cast<InvOp>(innerUnitary.getOperation());
    if (!innerInvOp) {
      return failure();
    }

    auto innerInnerUnitary = innerInvOp.getBodyUnitary();
    auto* clonedOp = rewriter.clone(*innerInnerUnitary.getOperation());
    rewriter.replaceOp(invOp, clonedOp->getResults());

    return success();
  }
};

} // namespace

UnitaryOpInterface InvOp::getBodyUnitary() {
  return llvm::dyn_cast<UnitaryOpInterface>(&getBody().front().front());
}

size_t InvOp::getNumQubits() { return getNumTargets() + getNumControls(); }

size_t InvOp::getNumTargets() { return getBodyUnitary().getNumTargets(); }

size_t InvOp::getNumControls() {
  return getNumPosControls() + getNumNegControls();
}

size_t InvOp::getNumPosControls() {
  return getBodyUnitary().getNumPosControls();
}

size_t InvOp::getNumNegControls() {
  return getBodyUnitary().getNumNegControls();
}

Value InvOp::getQubit(const size_t i) { return getBodyUnitary().getQubit(i); }

Value InvOp::getTarget(const size_t i) { return getBodyUnitary().getTarget(i); }

Value InvOp::getPosControl(const size_t i) {
  return getBodyUnitary().getPosControl(i);
}

Value InvOp::getNegControl(const size_t i) {
  return getBodyUnitary().getNegControl(i);
}

size_t InvOp::getNumParams() { return getBodyUnitary().getNumParams(); }

Value InvOp::getParameter(const size_t i) {
  return getBodyUnitary().getParameter(i);
}

void InvOp::build(OpBuilder& builder, OperationState& state,
                  UnitaryOpInterface bodyUnitary) {
  const OpBuilder::InsertionGuard guard(builder);
  auto* region = state.addRegion();
  auto& block = region->emplaceBlock();

  // Move the unitary op into the block
  builder.setInsertionPointToStart(&block);
  builder.clone(*bodyUnitary.getOperation());
  builder.create<YieldOp>(state.location);
}

void InvOp::build(OpBuilder& builder, OperationState& state,
                  const std::function<void(OpBuilder&)>& bodyBuilder) {
  const OpBuilder::InsertionGuard guard(builder);
  auto* region = state.addRegion();
  auto& block = region->emplaceBlock();

  builder.setInsertionPointToStart(&block);
  bodyBuilder(builder);
  builder.create<YieldOp>(state.location);
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
  llvm::SmallPtrSet<Value, 4> uniqueQubits;
  auto bodyUnitary = getBodyUnitary();
  const auto numQubits = bodyUnitary.getNumQubits();
  for (size_t i = 0; i < numQubits; i++) {
    if (!uniqueQubits.insert(bodyUnitary.getQubit(i)).second) {
      return emitOpError("duplicate qubit found");
    }
  }
  return success();
}

void InvOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<CancelNestedInv>(context);
}
