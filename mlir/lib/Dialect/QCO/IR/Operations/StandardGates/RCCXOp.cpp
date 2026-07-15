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

#include <cstddef>

using namespace mlir::qco;

static DynamicMatrix elementaryRCCXUnitary() {
  constexpr Matrix4x4 cx =
      Matrix4x4::fromElements(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                              0.0, 1.0, 0.0, 0.0, 1.0, 0.0);

  DynamicMatrix unitary = DynamicMatrix::identity(8);
  const auto apply1 = [&](const Matrix2x2& gate, const std::size_t qubit) {
    unitary = gate.embedInNqubit(3, qubit) * unitary;
  };
  const auto applyCx = [&](const std::size_t control,
                           const std::size_t target) {
    unitary = cx.embedInNqubit(3, control, target) * unitary;
  };

  apply1(HOp::getUnitaryMatrix(), 2);
  apply1(TOp::getUnitaryMatrix(), 2);
  applyCx(1, 2);
  apply1(TdgOp::getUnitaryMatrix(), 2);
  applyCx(0, 2);
  apply1(TOp::getUnitaryMatrix(), 2);
  applyCx(1, 2);
  apply1(TdgOp::getUnitaryMatrix(), 2);
  apply1(HOp::getUnitaryMatrix(), 2);
  return unitary;
}

DynamicMatrix RCCXOp::getUnitaryMatrix() {
  static const DynamicMatrix unitary = elementaryRCCXUnitary();
  return unitary;
}
