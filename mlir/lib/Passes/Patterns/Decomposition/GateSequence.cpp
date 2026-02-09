/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Passes/Decomposition/GateSequence.h"

#include "ir/operations/OpType.hpp"
#include "mlir/Passes/Decomposition/EulerBasis.h"
#include "mlir/Passes/Decomposition/Gate.h"
#include "mlir/Passes/Decomposition/Helpers.h"
#include "mlir/Passes/Decomposition/UnitaryMatrices.h"

#include <Eigen/Core>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <llvm/ADT/SmallVector.h>

namespace mlir::qco::decomposition {

bool QubitGateSequence::hasGlobalPhase() const {
  return std::abs(globalPhase) > DEFAULT_ATOL;
}

std::size_t QubitGateSequence::complexity() const {
  std::size_t c{};
  for (auto&& gate : gates) {
    c += helpers::getComplexity(gate.type, gate.qubitId.size());
  }
  if (hasGlobalPhase()) {
    // need to add a global phase gate if a global phase needs to be applied
    c += helpers::getComplexity(qc::GPhase, 0);
  }
  return c;
}

Eigen::Matrix4cd QubitGateSequence::getUnitaryMatrix() const {
  Eigen::Matrix4cd unitaryMatrix = Eigen::Matrix4cd::Identity();
  for (auto&& gate : gates) {
    auto gateMatrix = getTwoQubitMatrix(gate);
    unitaryMatrix = gateMatrix * unitaryMatrix;
  }
  unitaryMatrix *= helpers::globalPhaseFactor(globalPhase);
  assert(helpers::isUnitaryMatrix(unitaryMatrix));
  return unitaryMatrix;
}

} // namespace mlir::qco::decomposition
