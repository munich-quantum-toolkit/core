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

using namespace mlir;
using namespace mlir::qco;

void SdgOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                        MLIRContext* /*context*/) {
  results.add(&removeInversePairOneTargetZeroParameter<SOp, SdgOp>);
  results.add(&mergeOneTargetZeroParameter<ZOp, SdgOp>);
}

Matrix2x2 SdgOp::getUnitaryMatrix() {
  using namespace std::complex_literals;

  return Matrix2x2::fromElements(1, 0,    // row 0
                                 0, -1i); // row 1
}
