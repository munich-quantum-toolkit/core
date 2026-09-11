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

void SXOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                       MLIRContext* /*context*/) {
  results.add(&removeInversePairOneTargetZeroParameter<SXdgOp, SXOp>);
  results.add(&mergeOneTargetZeroParameter<XOp, SXOp>);
}

Matrix2x2 SXOp::getUnitaryMatrix() {
  constexpr auto diag = std::complex{0.5, 0.5};
  constexpr auto offDiag = std::complex{0.5, -0.5};
  return Matrix2x2::fromElements(diag, offDiag,  // row 0
                                 offDiag, diag); // row 1
}
