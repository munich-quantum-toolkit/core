/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QC/IR/QCOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::qc;

namespace {

struct HoistStaticQubit final : OpRewritePattern<StaticOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(StaticOp op,
                                PatternRewriter& rewriter) const override {
    auto funcOp = op->getParentOfType<func::FuncOp>();
    if (!funcOp ||
        op->getParentWithTrait<OpTrait::IsIsolatedFromAbove>() != funcOp ||
        op->getBlock() == &funcOp.getBody().front()) {
      return failure();
    }
    rewriter.moveOpBefore(op, &funcOp.getBody().front(),
                          funcOp.getBody().front().begin());
    return success();
  }
};

} // namespace

void StaticOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                           MLIRContext* context) {
  results.add<HoistStaticQubit>(context);
}
