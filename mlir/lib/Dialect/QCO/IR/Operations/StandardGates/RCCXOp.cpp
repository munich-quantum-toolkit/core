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

using namespace mlir::qco;

DynamicMatrix RCCXOp::getUnitaryMatrix() {
  DynamicMatrix unitary = DynamicMatrix::identity(8);
  unitary(3, 3) = 0.0;
  unitary(5, 5) = -1.0;
  unitary(7, 7) = 0.0;
  unitary(3, 7) = {0.0, -1.0};
  unitary(7, 3) = {0.0, 1.0};
  return unitary;
}
