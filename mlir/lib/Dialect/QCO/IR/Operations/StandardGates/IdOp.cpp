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
#include "mlir/Dialect/QCO/Utils/Matrix.h"

#include <mlir/IR/OperationSupport.h>

using namespace mlir;
using namespace mlir::qco;

OpFoldResult IdOp::fold(FoldAdaptor /*adaptor*/) { return getQubitIn(); }

Matrix2x2 IdOp::getUnitaryMatrix() {
  return Matrix2x2::fromElements(1, 0,  // row 0
                                 0, 1); // row 1
}
