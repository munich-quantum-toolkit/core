/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QTensor/IR/QTensorOps.h"
#include "mlir/Dialect/QTensor/IR/QTensorUtils.h"

#include <mlir/Dialect/QCO/IR/QCOOps.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>

using namespace mlir;
using namespace mlir::qtensor;

/// Check whether an extract reads from a tensor allocated in this IR.
static bool originatesFromAlloc(ExtractOp extract) {
  auto current = extract.getTensor();
  const auto extractIndex = extract.getIndex();
  if (!getConstantIntValue(extractIndex)) {
    return false;
  }

  while (auto* definingOp = current.getDefiningOp()) {
    if (isa<AllocOp>(definingOp)) {
      return true;
    }

    if (auto nestedExtract = dyn_cast<ExtractOp>(definingOp)) {
      if (!getConstantIntValue(nestedExtract.getIndex()) ||
          areEquivalentIndices(extractIndex, nestedExtract.getIndex())) {
        return false;
      }
      current = nestedExtract.getTensor();
      continue;
    }

    if (auto insert = dyn_cast<InsertOp>(definingOp)) {
      if (!getConstantIntValue(insert.getIndex()) ||
          areEquivalentIndices(extractIndex, insert.getIndex())) {
        return false;
      }
      current = insert.getDest();
      continue;
    }

    return false;
  }

  return false;
}

namespace {
/// Remove a reset after extracting a freshly allocated qubit.
struct RemoveResetAfterExtract final : OpRewritePattern<qco::ResetOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(qco::ResetOp reset,
                                PatternRewriter& rewriter) const override {
    const auto extract = reset.getQubitIn().getDefiningOp<ExtractOp>();
    if (extract == nullptr || !originatesFromAlloc(extract)) {
      return failure();
    }

    rewriter.replaceOp(reset, reset.getQubitIn());
    return success();
  }
};

/**
 * @brief Fold an insert followed immediately by an extract at the same index.
 */
struct FoldExtractAfterInsertPattern final : OpRewritePattern<ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractOp extract,
                                PatternRewriter& rewriter) const override {
    auto insert = extract.getTensor().getDefiningOp<InsertOp>();
    if (!insert || !insert->hasOneUse() ||
        !isEqualConstantIntOrValue(insert.getIndex(), extract.getIndex())) {
      return failure();
    }

    rewriter.replaceOp(extract, {insert.getDest(), insert.getScalar()});
    rewriter.eraseOp(insert);
    return success();
  }
};

} // namespace

LogicalResult ExtractOp::verify() {
  if (getOperation()->getParentOfType<qco::CtrlOp>() ||
      getOperation()->getParentOfType<qco::InvOp>() ||
      getOperation()->getParentOfType<qco::PowOp>()) {
    return emitOpError("cannot access a qubit tensor inside a QCO modifier");
  }

  auto tensorDim = getTensor().getType().getDimSize(0);
  auto index = getConstantIntValue(getIndex());

  if (index) {
    if (*index < 0) {
      return emitOpError("Index must be non-negative");
    }
    if (!ShapedType::isDynamic(tensorDim) && *index >= tensorDim) {
      return emitOpError("Index exceeds tensor dimension");
    }
  }
  return success();
}

void ExtractOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                            MLIRContext* context) {
  results.add<FoldExtractAfterInsertPattern, RemoveResetAfterExtract>(context);
}
