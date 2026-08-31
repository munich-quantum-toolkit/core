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

#include "mlir/Compiler/Target.h"

#include <llvm/ADT/DenseMap.h>

#include <cstddef>

namespace mlir {

/**
 * @brief Cached routing costs for a target gate.
 *
 * @details The cache is derived from an immutable compiler target. A native
 * ordered coupling has cost zero, a coupling available only in the opposite
 * direction has cost one, and an unavailable coupling has infinite cost.
 * Nonadjacent costs remain the target's shortest-path distance minus one.
 * Construction is linear in the explicit topology or reported site tuples;
 * unrestricted all-to-all targets do not enumerate every qubit pair.
 */
class TargetGateCosts {
public:
  /// Construct routing costs for a recognized two-qubit gate.
  TargetGateCosts(const CompilerTarget& target, CompilerTarget::GateKind gate);

  /// Return the cached routing cost between two valid target vertices.
  [[nodiscard]] float routingCostBetween(size_t source, size_t target) const;

  /// Return whether every adjacent ordered pair has zero gate cost.
  [[nodiscard]] bool isUniform() const noexcept;

private:
  CompilerTarget target_;
  llvm::DenseMap<CompilerTarget::Coupling, float> costs_;
  float defaultAdjacentCost_ = 0.F;
  bool uniform_ = true;
};

} // namespace mlir
