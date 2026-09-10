/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::qtensor;

namespace {
/// Remove both operations so forwarding the tensor preserves linearity.
struct FoldInsertAfterExtract final : OpRewritePattern<InsertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOp insert,
                                PatternRewriter& rewriter) const override {
    auto extract = insert.getScalar().getDefiningOp<ExtractOp>();
    if (!extract || insert.getDest() != extract.getOutTensor() ||
        !isEqualConstantIntOrValue(insert.getIndex(), extract.getIndex())) {
      return failure();
    }
    rewriter.replaceOp(insert, extract.getTensor());
    rewriter.eraseOp(extract);
    return success();
  }
};

/// Commutes a directly chained insert and extract at provably distinct
/// constant indices.
struct CommuteAdjacentInsertExtractPattern final : OpRewritePattern<InsertOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(InsertOp insert,
                                PatternRewriter& rewriter) const override {
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
  if (getOperation()->getParentOfType<qco::CtrlOp>() ||
      getOperation()->getParentOfType<qco::InvOp>() ||
      getOperation()->getParentOfType<qco::PowOp>()) {
    return emitOpError("cannot access a qubit tensor inside a QCO modifier");
  }

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
  results.add<FoldInsertAfterExtract, CommuteAdjacentInsertExtractPattern>(
      context);
}
