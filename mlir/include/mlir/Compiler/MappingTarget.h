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

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLFunctionalExtras.h>

#include <cstddef>

namespace mlir {

/// Compiler-target topology with cached directional mapping costs.
///
/// Adjacent native entangler directions have cost zero. A direction that can
/// be synthesized from the reverse native direction has cost one. Symmetric
/// entanglers have cost zero in both directions. Nonadjacent costs are the
/// target's shortest-path distance minus one. Construction visits each target
/// coupling at most once and all cost lookups are constant time.
class MappingTarget {
public:
  explicit MappingTarget(const CompilerTarget& target);

  /// Return the immutable compiler target.
  [[nodiscard]] const CompilerTarget& compilerTarget() const noexcept;

  /// Return the number of target sites.
  [[nodiscard]] size_t numSites() const noexcept;

  /// Return the target topology's maximum degree.
  [[nodiscard]] size_t maxDegree() const noexcept;

  /// Return the shortest-path distance between two valid target vertices.
  [[nodiscard]] size_t distanceBetween(size_t source, size_t target) const;

  /// Invoke @p callback for every neighbour of a valid target vertex.
  void forEachNeighbour(size_t vertex,
                        llvm::function_ref<void(size_t)> callback) const;

  /// Return the routing cost from @p source to @p target.
  [[nodiscard]] float pathCostBetween(size_t source, size_t target) const;

  /// Return whether a two-qubit gate is executable in this order.
  [[nodiscard]] bool isExecutable(size_t source, size_t target) const;

private:
  CompilerTarget target_;
  llvm::DenseSet<CompilerTarget::Coupling> penalizedDirections_;
};

} // namespace mlir
