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
#include "mlir/Dialect/Utils/MatrixUtils.h"

#include <cstddef>
#include <functional>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::quartz;
using namespace mlir::utils;

namespace {

/**
 * @brief Merge nested control modifiers into a single one.
 */
struct MergeNestedCtrl final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CtrlOp ctrlOp,
                                PatternRewriter& rewriter) const override {
    auto bodyUnitary = ctrlOp.getBodyUnitary();
    auto bodyCtrlOp = llvm::dyn_cast<CtrlOp>(bodyUnitary.getOperation());
    if (!bodyCtrlOp) {
      return failure();
    }

    llvm::SmallVector<Value> newControls(ctrlOp.getControls());
    for (const auto control : bodyCtrlOp.getControls()) {
      newControls.push_back(control);
    }

    rewriter.replaceOpWithNewOp<CtrlOp>(ctrlOp, newControls,
                                        bodyCtrlOp.getBodyUnitary());
    rewriter.eraseOp(bodyCtrlOp);

    return success();
  }
};

/**
 * @brief Remove control modifiers without controls.
 */
struct RemoveTrivialCtrl final : OpRewritePattern<CtrlOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CtrlOp ctrlOp,
                                PatternRewriter& rewriter) const override {
    if (ctrlOp.getNumControls() > 0) {
      return failure();
    }
    rewriter.replaceOp(ctrlOp, ctrlOp.getBodyUnitary());
    return success();
  }
};

} // namespace

void CtrlOp::build(OpBuilder& odsBuilder, OperationState& odsState,
                   const ValueRange controls, UnitaryOpInterface bodyUnitary) {
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
                   const ValueRange controls,
                   const std::function<void(OpBuilder&)>& bodyBuilder) {
  const OpBuilder::InsertionGuard guard(odsBuilder);
  odsState.addOperands(controls);
  auto* region = odsState.addRegion();
  auto& block = region->emplaceBlock();
  odsBuilder.setInsertionPointToStart(&block);
  bodyBuilder(odsBuilder);
  odsBuilder.create<YieldOp>(odsState.location);
}

UnitaryOpInterface CtrlOp::getBodyUnitary() {
  return llvm::dyn_cast<UnitaryOpInterface>(&getBody().front().front());
}

size_t CtrlOp::getNumQubits() { return getNumTargets() + getNumControls(); }

size_t CtrlOp::getNumTargets() { return getBodyUnitary().getNumTargets(); }

size_t CtrlOp::getNumControls() {
  return getNumPosControls() + getNumNegControls();
}

size_t CtrlOp::getNumPosControls() { return getControls().size(); }

size_t CtrlOp::getNumNegControls() {
  return getBodyUnitary().getNumNegControls();
}

Value CtrlOp::getQubit(const size_t i) {
  const auto numPosControls = getNumPosControls();
  if (i < numPosControls) {
    return getControls()[i];
  }
  if (numPosControls <= i && i < getNumQubits()) {
    return getBodyUnitary().getQubit(i - numPosControls);
  }
  llvm::reportFatalUsageError("Invalid qubit index");
}

Value CtrlOp::getTarget(const size_t i) {
  return getBodyUnitary().getTarget(i);
}

Value CtrlOp::getPosControl(const size_t i) { return getControls()[i]; }

Value CtrlOp::getNegControl(const size_t i) {
  return getBodyUnitary().getNegControl(i);
}

size_t CtrlOp::getNumParams() { return getBodyUnitary().getNumParams(); }

bool CtrlOp::hasStaticUnitary() { return getBodyUnitary().hasStaticUnitary(); }

Value CtrlOp::getParameter(const size_t i) {
  return getBodyUnitary().getParameter(i);
}

DenseElementsAttr CtrlOp::tryGetStaticMatrix() {
  return getMatrixCtrl(getContext(), getNumPosControls(),
                       getBodyUnitary().tryGetStaticMatrix());
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
  results.add<MergeNestedCtrl, RemoveTrivialCtrl>(context);
}
