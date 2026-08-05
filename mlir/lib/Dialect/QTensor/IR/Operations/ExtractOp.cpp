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

#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>

using namespace mlir;
using namespace mlir::qtensor;

namespace {
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
  results.add<FoldExtractAfterInsertPattern>(context);
}
