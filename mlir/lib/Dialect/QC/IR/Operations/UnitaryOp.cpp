/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQT/Utils/DenseUnitary.h"
#include "mlir/Dialect/QC/IR/QCOps.h"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qc;

namespace {

struct EraseIdentityUnitary final : OpRewritePattern<UnitaryOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UnitaryOp op,
                                PatternRewriter& rewriter) const override {
    if (!mqt::isExactIdentityMatrix(op.getMatrix())) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

LogicalResult UnitaryOp::verify() {
  return mqt::verifyDenseUnitaryMatrix(getOperation(), getMatrix(),
                                       getQubits());
}

void UnitaryOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                            MLIRContext* context) {
  results.add<EraseIdentityUnitary>(context);
}
