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

using namespace mlir;
using namespace mlir::qco;

void DCXOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* /*context*/) {
  results.add(+[](DCXOp op, PatternRewriter& rewriter) {
    return removeInversePairTwoTargetZeroParameter<DCXOp>(op, rewriter, false,
                                                          true);
  });
}

Matrix4x4 DCXOp::getUnitaryMatrix() {
  return Matrix4x4::fromElements(1, 0, 0, 0,  // row 0
                                 0, 0, 1, 0,  // row 1
                                 0, 0, 0, 1,  // row 2
                                 0, 1, 0, 0); // row 3
}
