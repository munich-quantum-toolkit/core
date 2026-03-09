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
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cstdint>

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
  if (destType.getRank() != static_cast<int64_t>(getIndices().size())) {
    return emitOpError("incorrect number of indices for insert");
  }
  return success();
}

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

/**
 * @brief If an InsertOp consumes an ExtractOp with identical indices,
 * return the tensor from the extractOp directly.
 */
struct InsertFromExtractOp : public OpRewritePattern<InsertOp> {
  using OpRewritePattern<InsertOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOp insertOp,
                                PatternRewriter& rewriter) const final {
    auto extractOp = insertOp.getScalar().getDefiningOp<ExtractOp>();
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
