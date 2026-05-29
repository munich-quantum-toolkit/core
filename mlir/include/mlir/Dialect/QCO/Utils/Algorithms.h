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

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <optional>
#include <utility>

namespace mlir::qco {
template <class T> using Vector = SmallVector<T, 0>;
template <class T> using Matrix = Vector<Vector<T>>;

class Graph {
public:
  using IdT = size_t;
  using EdgeSet = llvm::DenseSet<std::pair<IdT, IdT>>;

  /// Construct an empty graph.
  Graph() = default;
  /// Construct graph from edge set.
  explicit Graph(const EdgeSet& edges);
  /// Add a node to the graph.
  void addNode(IdT id);
  /// Add an edge to the graph.
  void addEdge(IdT id, IdT neighbourId);
  /// Add an edge to the graph.
  void addEdge(std::pair<IdT, IdT> edge);
  /// Add multiple edges to the graph.
  void addEdges(SmallVector<std::pair<IdT, IdT>> edges);
  /// Return a set of edges.
  [[nodiscard]] EdgeSet getEdges() const;
  /// Return the edges of a node.
  [[nodiscard]] ArrayRef<IdT> getEdges(size_t id) const;
  /// Return the number of nodes.
  [[nodiscard]] size_t getNumNodes() const { return adj_.size(); }
  /// Return the max degree of the graph.
  [[nodiscard]] size_t getMaxDegree() const;
  /// Return the minimum distance matrix of the graph by implementing the
  /// Floyd-Warshall Algorithm
  /// (https://en.wikipedia.org/wiki/Floyd–Warshall_algorithm) where dist[i][j]
  /// denotes the distance between i and j.
  [[nodiscard]] Matrix<size_t> getDistMatrix() const;
  /// Return cycle in graph or std::nullopt if none exists.
  [[nodiscard]] std::optional<Vector<IdT>> findCycle() const;

private:
  llvm::DenseMap<IdT, Vector<IdT>> adj_;
};
} // namespace mlir::qco
