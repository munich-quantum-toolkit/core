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

#include <Eigen/Core>

using namespace mlir;
using namespace mlir::qco;

Eigen::Matrix4cd DCXOp::getUnitaryMatrix() {
  return Eigen::Matrix4cd{{1, 0, 0, 0},  // row 0
                          {0, 0, 1, 0},  // row 1
                          {0, 0, 0, 1},  // row 2
                          {0, 1, 0, 0}}; // row 3
}
