/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Quartz/IR/QuartzDialect.h"
#include "mlir/Dialect/Utils/MatrixUtils.h"

#include <mlir/IR/BuiltinAttributes.h>

using namespace mlir;
using namespace mlir::quartz;
using namespace mlir::utils;

DenseElementsAttr SWAPOp::tryGetStaticMatrix() {
  return getMatrixSWAP(getContext());
}
