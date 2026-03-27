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

#include <llvm/Support/Casting.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qtensor;

static ExtractOp findExtractOp(InsertOp op) {

  auto* definingOp = op.getDest().getDefiningOp();
  if (llvm::isa<ExtractOp>(definingOp)) {
    return llvm::cast<ExtractOp>(definingOp);
  }
  if (llvm::isa<InsertOp>(definingOp)) {
    auto nestedInsertOp = llvm::cast<InsertOp>(definingOp);
    return findExtractOp(nestedInsertOp);
  }
  return nullptr;
}

namespace {

struct RemoveExtractInsertPair final : OpRewritePattern<InsertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOp op,
                                PatternRewriter& rewriter) const override {
    auto extractOp = findExtractOp(op);
    if (!extractOp) {
      return failure();
    }

    if (op.getScalar() != extractOp.getResult()) {
      return failure();
    }

    if (op.getIndex() != extractOp.getIndex()) {
      return failure();
    }

    // TODO: Improve this
    auto qubit = qco::AllocOp::create(rewriter, op.getLoc());
    rewriter.replaceOp(extractOp, {extractOp.getTensor(), qubit.getResult()});
    qco::DeallocOp::create(rewriter, op.getLoc(), qubit.getResult());

    rewriter.replaceOp(op, op.getDest());

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

void InsertOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                           MLIRContext* context) {
  results.add<RemoveExtractInsertPair>(context);
}
