/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/Support/Casting.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qtensor;

LogicalResult InsertOp::verify() {
  auto destType = getDest().getType();
  if (!llvm::isa<qco::QubitType>(getScalar().getType())) {
    return emitOpError("Scalar must be of qubit type");
  }
  if (!llvm::isa<qco::QubitType>(destType.getElementType())) {
    return emitOpError("Elements of dest tensor must be of qubit type");
  }
  auto index = getConstantIntValue(getIndex());
  auto size = destType.getDimSize(0);
  if (index) {
    if (index < 0) {
      return emitOpError("Index must be non-negative");
    }
    if (index >= size) {
      return emitOpError("Index exceeds tensor dimension");
    }
  }
  return success();
}

/**
 * @brief If an InsertOp consumes an ExtractOp with identical indices,
 * return the tensor from the extractOp directly.
 */
static Value foldInsertAfterExtract(InsertOp insertOp) {
  auto extractOp = insertOp.getScalar().getDefiningOp<ExtractOp>();
  if (!extractOp) {
    return nullptr;
  }
  if (insertOp.getDest() != extractOp.getOutTensor()) {
    return nullptr;
  }

  auto insertIndex = insertOp.getIndex();
  auto extractIndex = extractOp.getIndex();

  // Check if SSA values of the indices are the same
  if (insertIndex == extractIndex) {
    return extractOp.getTensor();
  }

  auto insertIndexValue = getConstantIntValue(insertIndex);
  auto extractIndexValue = getConstantIntValue(extractIndex);

  // Check if the indices are constant and equal
  if (!insertIndexValue || !extractIndexValue) {
    return nullptr;
  }
  if (*insertIndexValue != *extractIndexValue) {
    return nullptr;
  }

  return extractOp.getTensor();
}

OpFoldResult InsertOp::fold(FoldAdaptor /*adaptor*/) {
  if (auto result = foldInsertAfterExtract(*this)) {
    return result;
  }

  return {};
}

namespace {

struct InsertAfterInsertOp : public OpRewritePattern<InsertOp> {
  using OpRewritePattern<InsertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOp insertOp,
                                PatternRewriter& rewriter) const final {
    auto prevInsertOp = insertOp.getDest().getDefiningOp<InsertOp>();
    if (!prevInsertOp) {
      return failure();
    }
    auto insertIndex = insertOp.getIndex();
    auto prevInsertIndex = prevInsertOp.getIndex();

    if (!isSameIndex(insertIndex, prevInsertIndex)) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<InsertOp>(insertOp, insertOp.getScalar(),
                                          prevInsertOp.getDest(), insertIndex);
    rewriter.eraseOp(prevInsertOp);
    return success();
  }
};
} // namespace

void InsertOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                           MLIRContext* context) {
  results.add<InsertAfterInsertOp>(context);
}
