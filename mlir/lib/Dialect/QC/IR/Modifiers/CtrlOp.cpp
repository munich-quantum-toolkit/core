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
#include <llvm/ADT/STLFunctionalExtras.h>
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
    auto* bodyUnitary = op.getBodyUnitary().getOperation();
    auto bodyCtrlOp = llvm::dyn_cast<CtrlOp>(bodyUnitary);
    if (!bodyCtrlOp) {
      return failure();
    }

    // add the inner controls as operands to the outer one
    op->insertOperands(op.getNumOperands(), bodyCtrlOp.getControls());

    // Move the inner unitary op into the outer one's body region and replace
    // the outer one with the inner one's results
    const OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(bodyUnitary);
    auto* innerUnitaryOp = bodyCtrlOp.getBodyUnitary().getOperation();
    rewriter.moveOpBefore(innerUnitaryOp, bodyUnitary);
    rewriter.replaceOp(bodyUnitary, innerUnitaryOp->getResults());

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
    auto* bodyUnitary = op.getBodyUnitary().getOperation();
    // Inline ops from empty control modifiers, IdOp and BarrierOp
    if (op.getNumControls() == 0 || llvm::isa<IdOp, BarrierOp>(bodyUnitary)) {
      rewriter.moveOpBefore(bodyUnitary, op);
      rewriter.replaceOp(op, bodyUnitary->getResults());
      return success();
    }

    // The remaining code explicitly handles GPhaseOp and nothing else
    auto gPhaseOp = llvm::dyn_cast<GPhaseOp>(bodyUnitary);
    if (!gPhaseOp) {
      return failure();
    }

    // Special case for single control: replace with a single POp
    if (op.getNumControls() == 1) {
      rewriter.replaceOpWithNewOp<POp>(op, op.getControl(0),
                                       gPhaseOp.getTheta());
      return success();
    }

    // Remove the last control and replace with a single POp with the removed
    // control as target
    auto controls = op.getControls();
    auto target = controls.back();
    controls = controls.drop_back();
    op->setOperands(controls);

    const OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(gPhaseOp);
    rewriter.replaceOpWithNewOp<POp>(gPhaseOp, target, gPhaseOp.getTheta());

    return success();
  }
};

} // namespace

UnitaryOpInterface CtrlOp::getBodyUnitary() {
  // In principle, the body region should only contain exactly two operations,
  // the actual unitary operation and a yield operation. However, the region may
  // also contain constants and arithmetic operations, e.g., created as part of
  // canonicalization. Thus, the only safe way to access the unitary operation
  // is to get the second operation from the back of the region.
  return llvm::cast<UnitaryOpInterface>(*(++getBody()->rbegin()));
}

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

Value CtrlOp::getControl(const size_t i) {
  if (i >= getNumControls()) {
    llvm::reportFatalUsageError("Control index out of bounds");
  }
  return getControls()[i];
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
  if (block.getOperations().size() < 2) {
    return emitOpError("body region must have at least two operations");
  }
  if (!llvm::isa<YieldOp>(block.back())) {
    return emitOpError(
        "last operation in body region must be a yield operation");
  }
  auto iter = ++block.rbegin();
  if (!llvm::isa<UnitaryOpInterface>(*(iter))) {
    return emitOpError(
        "second to last operation in body region must be a unitary operation");
  }
  for (auto it = ++iter; it != block.rend(); ++it) {
    if (llvm::isa<UnitaryOpInterface>(*it)) {
      return emitOpError("body region may only contain a single unitary op");
    }
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

  return success();
}

void CtrlOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results.add<MergeNestedCtrl, ReduceCtrl>(context);
}
