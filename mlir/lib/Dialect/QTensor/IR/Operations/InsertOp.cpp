/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 MQSC GmbH
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
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <iterator>

using namespace mlir;
using namespace mlir::qtensor;

/**
 * @brief Checks whether removing an extract-insert pair is linearity-safe.
 */
static bool isRemovableExtractInsertPair(InsertOp insert, ExtractOp extract) {
  return insert.getScalar() == extract.getResult() &&
         areEquivalentIndices(insert.getIndex(), extract.getIndex());
}

/**
 * @brief Folds an insert operation after a matching extract operation into the
 * original tensor.
 */
static Value foldInsertAfterExtract(InsertOp insert) {
  auto extract = insert.getScalar().getDefiningOp<ExtractOp>();
  if (!extract) {
    return nullptr;
  }

  if (insert.getDest() != extract.getOutTensor()) {
    return nullptr;
  }

  if (!isRemovableExtractInsertPair(insert, extract)) {
    return nullptr;
  }

  return extract.getTensor();
}

namespace {
/**
 * @brief Remove an (insert, extract) pair when the inserted qubit has been
 * extracted previously with the same constant index.
 * @pre Assumes each qubit is extracted and inserted with the same index.
 */
struct RemoveInsertExtractPairPattern final : OpRewritePattern<InsertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOp insert,
                                PatternRewriter& rewriter) const override {
    // Check: Insert has constant index.
    if (!getConstantIntValue(insert.getIndex())) {
      return failure();
    }

    // Search for an extract operation on the tensor-chain with the same
    // constant index as the matched insert operation.
    TensorIterator it(insert.getResult());
    for (; it != std::default_sentinel; ++it) {
      if (!isa<ExtractOp>(it.operation())) {
        continue;
      }

      auto extract = cast<ExtractOp>(it.operation());

      // Check: Extract has constant index.
      if (!getConstantIntValue(extract.getIndex())) {
        return failure();
      }

      // Check: Same constant index.
      if (!areEquivalentIndices(extract.getIndex(), insert.getIndex())) {
        continue;
      }

      //                 ┌─────────┐                 ┌──────────┐
      // ... ─t = dest──▶│insert(i)│─▶ ... ─▶tensor─▶│extract(i)│─outTensor─▶...
      //                 └────▲────┘                 └────┬─────┘
      //          ... ─scalar─┘                           └result─▶ ...
      // ------------------------- ⬇ (transformed) ⬇ -------------------------
      // ... ─t = outTensor─▶ ...
      // ... ─scalar = result─▶ ... (Assumption applied.)

      rewriter.replaceOp(extract, {extract.getTensor(), insert.getScalar()});
      rewriter.replaceOp(insert, insert.getDest());

      return success();
    }

    return failure();
  }
};

/**
 * @brief If possible, move insert after extract in tensor chain.
 * @pre Assumes that the extract and insertion index of any qubit is equivalent.
 */
struct BubbleDownInsertPattern final : OpRewritePattern<InsertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOp insert,
                                PatternRewriter& rewriter) const override {
    if (!getConstantIntValue(insert.getIndex())) {
      return failure();
    }

    auto next = std::next(TensorIterator(insert.getResult()));
    if (next == std::default_sentinel) {
      return failure();
    }

    if (!isa<ExtractOp>(next.operation())) {
      return failure();
    }

    auto extract = cast<ExtractOp>(next.operation());
    if (!getConstantIntValue(extract.getIndex())) {
      return failure();
    }

    if (areEquivalentIndices(extract.getIndex(), insert.getIndex())) {
      return failure();
    }

    // i != j
    //                ┌─────────┐                  ┌──────────┐
    // ... ─t = dest─▶│insert(i)│─result = tensor─▶│extract(j)│─outTensor─▶ ...
    //                └─────────┘                  └──────────┘
    // -------------------------- ⬇ (transformed) ⬇ --------------------------
    //                  ┌──────────┐                   ┌─────────┐
    // ... ─t = tensor─▶│extract(j)│─outTensor = dest─▶│insert(i)│─result─▶ ...
    //                  └──────────┘                   └─────────┘

    const Value t = insert.getDest();
    const Value outTensor = extract.getOutTensor();
    const Value result = insert.getResult();

    rewriter.moveOpAfter(insert, extract);
    rewriter.modifyOpInPlace(extract,
                             [&] { extract.getTensorMutable().assign(t); });
    rewriter.modifyOpInPlace(
        insert, [&] { insert.getDestMutable().assign(outTensor); });
    rewriter.replaceAllUsesExcept(outTensor, result, insert);

    return success();
  }
};
} // namespace

LogicalResult InsertOp::verify() {
  auto dstDim = getDest().getType().getDimSize(0);
  auto index = getConstantIntValue(getIndex());

  if (index) {
    if (*index < 0) {
      return emitOpError("Index must be non-negative");
    }
    if (!ShapedType::isDynamic(dstDim) && *index >= dstDim) {
      return emitOpError("Index exceeds tensor dimension");
    }
  }

  return success();
}

OpFoldResult InsertOp::fold(FoldAdaptor /*adaptor*/) {
  if (auto result = foldInsertAfterExtract(*this)) {
    return result;
  }
  return {};
}

void InsertOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                           MLIRContext* context) {
  results.add<RemoveInsertExtractPairPattern, BubbleDownInsertPattern>(context);
}
