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
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cassert>
#include <cstdint>

using namespace mlir;
using namespace mlir::qtensor;

void FromElementsOp::build(OpBuilder& builder, OperationState& result,
                           ValueRange elements) {
  assert(!elements.empty() && "expected at least one element");
  Type resultType = RankedTensorType::get(
      {static_cast<int64_t>(elements.size())}, elements.front().getType());
  build(builder, result, resultType, elements);
}

namespace {

struct ConvertFromElementsOpToTensorOp
    : public OpRewritePattern<qtensor::FromElementsOp> {
  using OpRewritePattern<qtensor::FromElementsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(qtensor::FromElementsOp fromElementsOp,
                                PatternRewriter& rewriter) const final {

    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(
        fromElementsOp, fromElementsOp.getElements());

    return success();
  }
};

} // namespace

void FromElementsOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                                 MLIRContext* context) {
  results.add<ConvertFromElementsOpToTensorOp>(context);
}
