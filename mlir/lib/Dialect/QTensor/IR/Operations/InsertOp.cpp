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

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
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

/**
 * @brief Finds the extract operation corresponding to a given insert operation.
 *
 * @details The function traverses the tensor chain of the insert operation
 * until it finds the matching extract operation.
 */
static ExtractOp findMatchingExtractInTensorChain(InsertOp op) {
  TensorIterator it(op.getResult());
  for (; !isa<AllocOp>(it.operation()); --it) {
    if (auto nestedInsert = dyn_cast<InsertOp>(it.operation())) {
      if (!getConstantIntValue(nestedInsert.getIndex())) {
        return nullptr;
      }

      // A more recent write to the same index shadows all older extracts
      if (areEquivalentIndices(nestedInsert.getIndex(), op.getIndex())) {
        return nullptr;
      }
      continue;
    }

    if (auto extract = dyn_cast<ExtractOp>(it.operation())) {
      if (!getConstantIntValue(extract.getIndex())) {
        return nullptr;
      }

      if (areEquivalentIndices(extract.getIndex(), op.getIndex())) {
        return extract;
      }
      continue;
    }
  }

  return nullptr;
}

namespace {

/**
 * @brief Remove matching extract-insert pairs.
 */
struct RemoveExtractInsertPair final : OpRewritePattern<InsertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOp op,
                                PatternRewriter& rewriter) const override {
    auto extract = findMatchingExtractInTensorChain(op);
    if (!extract) {
      return failure();
    }

    if (!isRemovableExtractInsertPair(op, extract)) {
      return failure();
    }

    rewriter.replaceOp(op, op.getDest());
    rewriter.replaceOp(extract, {extract.getTensor(), nullptr});

    return success();
  }
};

/**
 * @brief Replace extracted qubit with previously inserted qubit and remove both
 * the insert as well as the extract operation.
 */
struct RemoveExtractAfterInsert final : OpRewritePattern<InsertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOp op,
                                PatternRewriter& rewriter) const override {
    for (TensorIterator it(op.getResult()); it != std::default_sentinel; ++it) {
      if (!isa<ExtractOp>(it.operation())) {
        continue;
      }

      auto extract = cast<ExtractOp>(it.operation());
      if (extract.getIndex() != op.getIndex()) {
        continue;
      }

      rewriter.replaceAllUsesWith(extract.getResult(), op.getScalar());
      rewriter.replaceAllUsesWith(extract.getOutTensor(), extract.getTensor());
      rewriter.replaceAllUsesWith(op.getResult(), op.getDest());

      rewriter.eraseOp(extract);
      rewriter.eraseOp(op);

      return success();
    }

    return failure();
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
  results.add<RemoveExtractInsertPair, RemoveExtractAfterInsert>(context);
}
