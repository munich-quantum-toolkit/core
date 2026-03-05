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

#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qtensor;

namespace {

struct ConvertInsertOpToTensorOp : public OpRewritePattern<qtensor::InsertOp> {
  using OpRewritePattern<qtensor::InsertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(qtensor::InsertOp insertOp,
                                PatternRewriter& rewriter) const final {
    rewriter.replaceOpWithNewOp<tensor::InsertOp>(
        insertOp, insertOp.getScalar(), insertOp.getDest(),
        insertOp.getIndices());
    return success();
  }
};

struct InsertFromExtractOp : public OpRewritePattern<InsertOp> {
  using OpRewritePattern<InsertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOp insertOp,
                                PatternRewriter& rewriter) const final {
    auto extractOp = insertOp.getScalar().getDefiningOp<qtensor::ExtractOp>();
    if (!extractOp) {
      return failure();
    }
    if (insertOp.getDest() != extractOp.getOutTensor()) {
      return failure();
    }
    if (insertOp.getIndices() != extractOp.getIndices()) {
      return failure();
    }

    rewriter.replaceOp(insertOp, extractOp.getTensor());
    rewriter.eraseOp(extractOp);
    return success();
  }
};

} // namespace

void InsertOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                           MLIRContext* context) {
  results.add<InsertFromExtractOp, ConvertInsertOpToTensorOp>(context);
}
