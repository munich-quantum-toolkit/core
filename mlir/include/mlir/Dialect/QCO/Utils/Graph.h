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

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <optional>
#include <utility>

namespace mlir::qco {

/// A directed graph.
class Graph {
public:
  /// Construct graph from node identifiers.
  explicit Graph(ArrayRef<size_t> nodes) {
    for_each(nodes, [this](const auto u) { std::ignore = adj_[u]; });
  }

  /// Add a directed edge to the internal representation of the graph.
  /// Implicitly adds nodes.
  void addEdge(size_t u, size_t v);

  /// Return the neighbours of a node.
  [[nodiscard]] ArrayRef<size_t> getNeighbours(size_t id) const;

  /// Return the nodes.
  [[nodiscard]] SmallVector<size_t> getNodes() const;

  /// Return the degree of a node.
  [[nodiscard]] size_t getDegree(const size_t id) const {
    return adj_.at(id).size();
  }

  /// Remove the edges from the graph. Keep the nodes.
  void clearEdges();

  /// Return cycle in graph or `std::nullopt` if none exists.
  /// Implements an iterative depth-first search inspired by LLVM's SCC
  /// utilities. For a cycle [A, B, C, A], the function returns [A, B, C].
  [[nodiscard]] std::optional<SmallVector<size_t>> findCycle() const;

private:
  llvm::DenseMap<size_t, SmallVector<size_t>> adj_;
};
} // namespace mlir::qco
