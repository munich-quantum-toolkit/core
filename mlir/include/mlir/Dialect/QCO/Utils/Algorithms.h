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
#include <utility>

namespace mlir::qco {
template <class T> using Vector = SmallVector<T, 0>;
template <class T> using Matrix = Vector<Vector<T>>;
using EdgeSet = llvm::DenseSet<std::pair<size_t, size_t>>;

class Graph {
public:
  /// Construct an empty graph.
  Graph() = default;
  /// Construct graph from edge set.
  explicit Graph(const EdgeSet& edges);
  /// Add a node to the graph.
  void addNode(size_t id);
  /// Add an edge to the graph.
  void addEdge(size_t id, size_t neighbourId);
  /// Add an edge to the graph.
  void addEdge(std::pair<size_t, size_t> edge);
  /// Add multiple edges to the graph.
  void addEdges(SmallVector<std::pair<size_t, size_t>> edges);
  /// Return a set of edges.
  [[nodiscard]] EdgeSet getEdges() const;
  /// Return the edges of a node.
  [[nodiscard]] ArrayRef<size_t> getEdges(size_t id) const;
  /// Return the number of nodes.
  [[nodiscard]] size_t getNumNodes() const { return nodes_.size(); }
  /// Returns the max degree of the graph.
  [[nodiscard]] size_t getMaxDegree() const;

private:
  llvm::DenseMap<size_t, Vector<size_t>> nodes_;
};

/**
 * @brief Find all shortest paths between two nodes in a graph.
 * @details Has a time complexity of O(n^3).
 *
 * @link Adapted from https://en.wikipedia.org/wiki/Floyd–Warshall_algorithm
 *
 * @param n The number of nodes in the graph.
 * @param edges The set of edges (i, j).
 *
 * @returns The distance matrix dist, where dist[i, j] is defined as the
 * distance between node i and j.
 */
Matrix<size_t> findAllShortestPaths(const Graph& graph);

} // namespace mlir::qco
