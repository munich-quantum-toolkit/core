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

#include <numbers>

using namespace mlir;
using namespace mlir::qco;

void HOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                      MLIRContext* /*context*/) {
  results.add(&removeInversePairOneTargetZeroParameter<HOp, HOp>);
}

Matrix2x2 HOp::getUnitaryMatrix() {
  constexpr auto x = 1.0 / std::numbers::sqrt2;
  return Matrix2x2::fromElements(x, x,   // row 0
                                 x, -x); // row 1
}
