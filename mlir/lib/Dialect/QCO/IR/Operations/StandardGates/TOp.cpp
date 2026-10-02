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
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QCO/Utils/Matrix.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#include <complex>
#include <numbers>

using namespace mlir;
using namespace mlir::qco;

void TOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                      MLIRContext* /*context*/) {
  results.add(&removeInversePairOneTargetZeroParameter<TdgOp, TOp>);
  results.add(&mergeOneTargetZeroParameter<SOp, TOp>);
}

Matrix2x2 TOp::getUnitaryMatrix() {
  const auto m11 = std::polar(1.0, std::numbers::pi / 4);
  return Matrix2x2::fromElements(1, 0,    // row 0
                                 0, m11); // row 1
}
