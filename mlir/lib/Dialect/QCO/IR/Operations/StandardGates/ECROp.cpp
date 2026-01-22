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
#include "mlir/Dialect/QCO/QCOUtils.h"

#include <Eigen/Core>
#include <complex>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <numbers>

using namespace mlir;
using namespace mlir::qco;

namespace {

/**
 * @brief Remove subsequent ECR operations on the same qubits.
 */
struct RemoveSubsequentECR final : OpRewritePattern<ECROp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ECROp op,
                                PatternRewriter& rewriter) const override {
    return removeInversePairTwoTargetZeroParameter<ECROp>(op, rewriter);
  }
};

} // namespace

void ECROp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* context) {
  results.add<RemoveSubsequentECR>(context);
}

Eigen::Matrix4cd ECROp::getUnitaryMatrix() {
  using namespace std::complex_literals;

  constexpr auto m0 = 0i;
  constexpr auto m1 = std::complex<double>{1.0 / std::numbers::sqrt2};
  constexpr auto mi = std::complex<double>{0.0, 1.0 / std::numbers::sqrt2};
  return Eigen::Matrix4cd{{m0, m0, m1, mi},   // row 0
                          {m0, m0, mi, m1},   // row 1
                          {m1, -mi, m0, m0},  // row 2
                          {-mi, m1, m0, m0}}; // row 3
}
