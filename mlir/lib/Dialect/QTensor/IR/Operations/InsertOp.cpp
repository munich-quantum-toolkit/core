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
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

using namespace mlir;
using namespace mlir::qtensor;

/**
 * @brief Checks whether removing an extract-insert pair is linearity-safe.
 */
static bool isRemovableExtractInsertPair(InsertOp insert, ExtractOp extract) {
  return insert.getScalar() == extract.getResult() &&
         isEqualConstantIntOrValue(insert.getIndex(), extract.getIndex());
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
 * @brief Commutes a directly chained insert and extract at provably distinct
 * constant indices.
 */
struct CommuteAdjacentInsertExtractPattern final : OpRewritePattern<InsertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOp insert,
                                PatternRewriter& rewriter) const override {
    if (!insert.getResult().hasOneUse()) {
      return failure();
    }
    auto extract = dyn_cast<ExtractOp>(*insert.getResult().getUsers().begin());
    if (!extract || insert->getBlock() != extract->getBlock()) {
      return failure();
    }

    const auto insertIndex = getConstantIntValue(insert.getIndex());
    const auto extractIndex = getConstantIntValue(extract.getIndex());
    if (!insertIndex || !extractIndex || insertIndex == extractIndex) {
      return failure();
    }

    Value tensorBeforeInsert = insert.getDest();
    Value tensorAfterExtract = extract.getOutTensor();
    Value tensorAfterInsert = insert.getResult();

    rewriter.moveOpAfter(insert, extract);
    rewriter.modifyOpInPlace(extract, [&] {
      extract.getTensorMutable().assign(tensorBeforeInsert);
    });
    rewriter.modifyOpInPlace(
        insert, [&] { insert.getDestMutable().assign(tensorAfterExtract); });
    rewriter.replaceAllUsesExcept(tensorAfterExtract, tensorAfterInsert,
                                  insert);
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
  results.add<CommuteAdjacentInsertExtractPattern>(context);
}
