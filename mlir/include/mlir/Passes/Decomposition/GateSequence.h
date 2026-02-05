/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "EulerBasis.h"
#include "Gate.h"
#include "Helpers.h"
#include "UnitaryMatrices.h"
#include "ir/operations/OpType.hpp"

#include <Eigen/Core>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <llvm/ADT/STLExtras.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

namespace mlir::qco::decomposition {
/**
 * Gate sequence of single-qubit and/or two-qubit gates.
 */
struct QubitGateSequence {
  /**
   * Container sorting the gate sequence in order.
   */
  llvm::SmallVector<Gate, 8> gates;

  /**
   * Global phase adjustment required for the sequence.
   */
  fp globalPhase{};
  /**
   * @return true if the global phase adjustment is not zero.
   */
  [[nodiscard]] bool hasGlobalPhase() const {
    return std::abs(globalPhase) > DEFAULT_ATOL;
  }

  /**
   * Calculate complexity of sequence according to getComplexity().
   */
  [[nodiscard]] std::size_t complexity() const {
    // TODO: add more sophisticated metric to determine complexity of
    // series/sequence
    // TODO: caching mechanism
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

  /**
   * Calculate overall unitary matrix of the sequence.
   */
  [[nodiscard]] matrix4x4 getUnitaryMatrix() const {
    matrix4x4 unitaryMatrix = matrix4x4::Identity();
    for (auto&& gate : gates) {
      auto gateMatrix = getTwoQubitMatrix(gate);
      unitaryMatrix = gateMatrix * unitaryMatrix;
    }
    unitaryMatrix *= std::exp(IM * globalPhase);
    assert(helpers::isUnitaryMatrix(unitaryMatrix));
    return unitaryMatrix;
  }
};
/**
 * Helper type to show that a gate sequence is supposed to only contain
 * single-qubit gates.
 */
using OneQubitGateSequence = QubitGateSequence;
/**
 * Helper type to show that the gate sequence may contain two-qubit gates.
 */
using TwoQubitGateSequence = QubitGateSequence;

} // namespace mlir::qco::decomposition
