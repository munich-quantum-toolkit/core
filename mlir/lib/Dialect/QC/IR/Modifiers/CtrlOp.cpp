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

#include <cstddef>
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
using namespace mlir::qc;

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

    llvm::SmallVector<Value> newControls(op.getControls());
    for (const auto control : bodyCtrlOp.getControls()) {
      newControls.push_back(control);
    }

    rewriter.replaceOpWithNewOp<CtrlOp>(op, newControls,
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

    rewriter.clone(*op.getBodyUnitary().getOperation());
    rewriter.eraseOp(op);

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

    SmallVector<Value> newControls(op.getControls());
    const auto newTarget = newControls.back();
    newControls.pop_back();
    CtrlOp::create(rewriter, op.getLoc(), newControls, [&] {
      POp::create(rewriter, op.getLoc(), newTarget, gPhaseOp.getTheta());
    });
    rewriter.eraseOp(op);

    return success();
  }
};

/**
 * @brief Inline controlled Identity operations.
 */
struct CtrlInlineIdentity final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CtrlOp op,
                                PatternRewriter& rewriter) const override {
    // Require at least one positive control
    // Trivial case is handled by RemoveTrivialCtrl
    if (op.getNumControls() == 0) {
      return failure();
    }

    auto identityOp = llvm::dyn_cast<IdOp>(op.getBodyUnitary().getOperation());
    if (!identityOp) {
      return failure();
    }

    rewriter.moveOpBefore(identityOp, op);
    rewriter.replaceOp(op, identityOp->getResults());
    return success();
  };
};

} // namespace

UnitaryOpInterface CtrlOp::getBodyUnitary() {
  return llvm::dyn_cast<UnitaryOpInterface>(&getBody()->front());
}

size_t CtrlOp::getNumQubits() { return getNumTargets() + getNumControls(); }

size_t CtrlOp::getNumTargets() { return getBodyUnitary().getNumTargets(); }

size_t CtrlOp::getNumControls() { return getControls().size(); }

Value CtrlOp::getQubit(const size_t i) {
  const auto numControls = getNumControls();
  if (i < numControls) {
    return getControls()[i];
  }
  if (numControls <= i && i < getNumQubits()) {
    return getBodyUnitary().getQubit(i - numControls);
  }
  llvm::reportFatalUsageError("Invalid qubit index");
}

Value CtrlOp::getTarget(const size_t i) {
  return getBodyUnitary().getTarget(i);
}

Value CtrlOp::getControl(const size_t i) {
  if (i >= getNumControls()) {
    llvm::reportFatalUsageError("Control index out of bounds");
  }
  return getControls()[i];
}

size_t CtrlOp::getNumParams() { return getBodyUnitary().getNumParams(); }

Value CtrlOp::getParameter(const size_t i) {
  return getBodyUnitary().getParameter(i);
}

void CtrlOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                   ValueRange controls, UnitaryOpInterface bodyUnitary) {
  const OpBuilder::InsertionGuard guard(odsBuilder);
  odsState.addOperands(controls);
  auto* region = odsState.addRegion();
  auto& block = region->emplaceBlock();

  // Move the unitary op into the block
  odsBuilder.setInsertionPointToStart(&block);
  odsBuilder.clone(*bodyUnitary.getOperation());
  odsBuilder.create<YieldOp>(odsState.location);
}

void CtrlOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                   ValueRange controls,
                   const llvm::function_ref<void()>& bodyBuilder) {
  const OpBuilder::InsertionGuard guard(odsBuilder);
  odsState.addOperands(controls);
  auto* region = odsState.addRegion();
  auto& block = region->emplaceBlock();

  odsBuilder.setInsertionPointToStart(&block);
  bodyBuilder();
  odsBuilder.create<YieldOp>(odsState.location);
}

LogicalResult CtrlOp::verify() {
  auto& block = *getBody();
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
  for (const auto& control : getControls()) {
    if (!uniqueQubits.insert(control).second) {
      return emitOpError("duplicate control qubit found");
    }
  }
  auto bodyUnitary = getBodyUnitary();
  const auto numQubits = bodyUnitary.getNumQubits();
  for (size_t i = 0; i < numQubits; i++) {
    if (!uniqueQubits.insert(bodyUnitary.getQubit(i)).second) {
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
  results.add<MergeNestedCtrl, RemoveTrivialCtrl, CtrlInlineGPhase,
              CtrlInlineIdentity>(context);
}
