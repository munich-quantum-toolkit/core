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
#include "mlir/Dialect/QCO/QCOUtils.h"
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

using namespace mlir;
using namespace mlir::qco;

namespace {

/**
 * @brief Remove subsequent RCCX operations on the same qubits.
 */
struct RemoveSubsequentRCCX final : OpRewritePattern<RCCXOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RCCXOp op,
                                PatternRewriter& rewriter) const override {
    return removeInversePairThreeTargetZeroParameter<RCCXOp>(op, rewriter);
  }
};

} // namespace

void RCCXOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                         MLIRContext* context) {
  results.add<RemoveSubsequentRCCX>(context);
}

DynamicMatrix RCCXOp::getUnitaryMatrix() {
  DynamicMatrix unitary = DynamicMatrix::identity(8);
  unitary(3, 3) = 0.0;
  unitary(5, 5) = -1.0;
  unitary(7, 7) = 0.0;
  unitary(3, 7) = {0.0, -1.0};
  unitary(7, 3) = {0.0, 1.0};
  return unitary;
}
