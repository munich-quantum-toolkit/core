/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"
#include "mlir/Dialect/QTensor/IR/QTensorUtils.h"

#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qco;

/**
 * @brief Check if a `qtensor.extract` operation reads from a `qtensor.alloc`
 * chain.
 *
 * @details In QTensor's linear tensor model, reads/writes on different indices
 * commute. We can therefore skip over `qtensor.insert` on other indices while
 * tracing provenance. A write to the same index invalidates the proof.
 */
static bool originatesFromQTensorAlloc(qtensor::ExtractOp extractOp) {
  auto current = extractOp.getTensor();

  auto extractIndex = extractOp.getIndex();
  if (!getConstantIntValue(extractIndex)) {
    return false;
  }

  while (auto* definingOp = current.getDefiningOp()) {
    if (llvm::isa<qtensor::AllocOp>(definingOp)) {
      return true;
    }

    if (auto nestedExtractOp = llvm::dyn_cast<qtensor::ExtractOp>(definingOp)) {
      auto nestedExtractIndex = nestedExtractOp.getIndex();
      if (!getConstantIntValue(nestedExtractIndex)) {
        return false;
      }
      if (qtensor::areEquivalentIndices(extractIndex, nestedExtractIndex)) {
        return false;
      }
      current = nestedExtractOp.getTensor();
      continue;
    }

    if (auto insertOp = llvm::dyn_cast<qtensor::InsertOp>(definingOp)) {
      auto insertIndex = insertOp.getIndex();
      if (!getConstantIntValue(insertIndex)) {
        return false;
      }
      if (qtensor::areEquivalentIndices(extractIndex, insertIndex)) {
        return false;
      }
      current = insertOp.getDest();
      continue;
    }

    return false;
  }

  return false;
}

namespace {

/**
 * @brief Remove reset operations that immediately follow a `qtensor.extract`
 * operation.
 */
struct RemoveResetAfterExtract final : OpRewritePattern<ResetOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ResetOp op,
                                PatternRewriter& rewriter) const override {
    // Check if the predecessor is an ExtractOp
    auto extractOp = op.getQubitIn().getDefiningOp<qtensor::ExtractOp>();
    if (!extractOp) {
      return failure();
    }

    // Check if the tensor originates from an AllocOp
    if (!originatesFromQTensorAlloc(extractOp)) {
      return failure();
    }

    // Remove the ResetOp
    rewriter.replaceOp(op, op.getQubitIn());
    return success();
  }
};

} // namespace

OpFoldResult ResetOp::fold(FoldAdaptor /*adaptor*/) {
  if (getQubitIn().getDefiningOp<AllocOp>()) {
    return getQubitIn();
  }

  return {};
}

void ResetOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                          MLIRContext* context) {
  results.add<RemoveResetAfterExtract>(context);
}
