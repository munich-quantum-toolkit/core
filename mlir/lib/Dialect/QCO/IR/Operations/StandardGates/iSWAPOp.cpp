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
#include "mlir/Dialect/QCO/Utils/UnitaryMatrix.h"

#include <complex>

using namespace mlir;
using namespace mlir::qco;

Matrix4 iSWAPOp::getUnitaryMatrix() {
  using namespace std::complex_literals;

  return Matrix4::fromElements(1, 0, 0, 0, 0, 0, 1i, 0, 0, 1i, 0, 0, 0, 0, 0,
                               1);
}
