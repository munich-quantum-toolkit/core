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

void XOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                      MLIRContext* /*context*/) {
  results.add(&removeInversePairOneTargetZeroParameter<XOp, XOp>);
}

Matrix2x2 XOp::getUnitaryMatrix() {
  return Matrix2x2::fromElements(0, 1,  // row 0
                                 1, 0); // row 1
}
