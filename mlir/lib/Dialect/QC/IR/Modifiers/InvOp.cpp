/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QC/IR/QCDialect.h"

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
using namespace mlir::qc;

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

size_t InvOp::getNumControls() { return getBodyUnitary().getNumControls(); }

Value InvOp::getQubit(const size_t i) { return getBodyUnitary().getQubit(i); }

Value InvOp::getTarget(const size_t i) { return getBodyUnitary().getTarget(i); }

Value InvOp::getControl(const size_t i) {
  return getBodyUnitary().getControl(i);
}

size_t InvOp::getNumParams() { return getBodyUnitary().getNumParams(); }

Value InvOp::getParameter(const size_t i) {
  return getBodyUnitary().getParameter(i);
}

void InvOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                  UnitaryOpInterface bodyUnitary) {
  const OpBuilder::InsertionGuard guard(odsBuilder);
  auto* region = odsState.addRegion();
  auto& block = region->emplaceBlock();

  // Move the unitary op into the block
  odsBuilder.setInsertionPointToStart(&block);
  odsBuilder.clone(*bodyUnitary.getOperation());
  YieldOp::create(odsBuilder, odsState.location);
}

void InvOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                  const std::function<void(OpBuilder&)>& bodyBuilder) {
  const OpBuilder::InsertionGuard guard(odsBuilder);
  auto* region = odsState.addRegion();
  auto& block = region->emplaceBlock();

  odsBuilder.setInsertionPointToStart(&block);
  bodyBuilder(odsBuilder);
  YieldOp::create(odsBuilder, odsState.location);
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

  if (llvm::isa<BarrierOp>(bodyUnitary.getOperation())) {
    return emitOpError("BarrierOp cannot be inverted");
  }

  return success();
}

void InvOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<CancelNestedInv>(context);
}
