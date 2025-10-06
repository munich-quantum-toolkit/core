/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Flux/IR/FluxOps.h"

#include "mlir/Dialect/Flux/IR/FluxDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::flux;

//===----------------------------------------------------------------------===//
// ResetOp Canonicalization
//===----------------------------------------------------------------------===//

void ResetOp::getCanonicalizationPatterns(RewritePatternSet& results,
                                          MLIRContext* context) {
  // Patterns will be added in future phases
}

#define GET_OP_CLASSES
#include "mlir/Dialect/Flux/IR/FluxOps.cpp.inc"
