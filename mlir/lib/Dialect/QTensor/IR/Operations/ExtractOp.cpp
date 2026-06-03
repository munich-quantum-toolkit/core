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
#include "mlir/Dialect/QTensor/Utils/TensorIterator.h"

#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>

#include <iterator>

using namespace mlir;
using namespace mlir::qtensor;

namespace {
/**
 * @brief Remove an (extract, insert) pair when the extracted qubit is
 * reinserted unchanged at the same constant index.
 */
struct RemoveExtractInsertPairPattern final : OpRewritePattern<ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractOp extract,
                                PatternRewriter& rewriter) const override {
    // Check: Extract has constant index.
    if (!getConstantIntValue(extract.getIndex())) {
      return failure();
    }

    // Search for an insert operation on the tensor-chain with the same constant
    // index as the matched extract operation.
    TensorIterator it(extract.getOutTensor());
    for (; it != std::default_sentinel; ++it) {
      if (!isa<InsertOp>(it.operation())) {
        continue;
      }

      auto insert = cast<InsertOp>(it.operation());

      // Check: Insert has constant index.
      if (!getConstantIntValue(insert.getIndex())) {
        return failure();
      }

      // Check: Same constant index.
      if (!areEquivalentIndices(insert.getIndex(), extract.getIndex())) {
        continue;
      }

      // Check: The inserted qubit value is the extracted one. If so, the
      // qubit has not been used and both operations can be safely removed.

      if (extract.getResult() == insert.getScalar()) {

        //              ┌──────────┐         ┌─────────┐
        // ... ─tensor─▶│extract(i)│─▶ ... ─▶│insert(i)│─▶result─▶ ...
        //              └────┬─────┘         └────▲────┘
        //                   └──result = scalar───┘
        // ------------------- ⬇ (transformed) ⬇ -------------------
        // ... ─tensor = result─▶ ...

        rewriter.replaceOp(insert, insert.getDest());
        rewriter.replaceOp(extract, {extract.getTensor(), nullptr});
        return success();
      }
    }

    return failure();
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
  results.add<RemoveExtractInsertPairPattern>(context);
}
