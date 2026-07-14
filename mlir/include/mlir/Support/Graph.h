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
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/Debug.h>
#include <mlir/Support/LLVM.h>

#include <cassert>
#include <cstddef>
#include <optional>
#include <utility>

namespace mlir {

/// A directed graph.
class Graph {
public:
  class DistanceMatrix {
    SmallVector<size_t> data_;
    size_t n_{};

  public:
    /// Initialize distance matrix, where all entries are filled with `v`.
    explicit DistanceMatrix(size_t n, size_t v) : n_(n), data_(n * n, v) {}

    /// Return the i-th row.
    MutableArrayRef<size_t> operator[](size_t i) {
      assert(i < n_ && "row index out of bounds");
      return MutableArrayRef<size_t>(data_).slice(i * n_, n_);
    }

    /// Return the i-th row.
    ArrayRef<size_t> operator[](size_t i) const {
      assert(i < n_ && "row index out of bounds");
      return ArrayRef<size_t>(data_).slice(i * n_, n_);
    }
  };

  /// Construct an empty graph.
  Graph() = default;

  /// Construct graph from node identifiers.
  explicit Graph(ArrayRef<size_t> nodes) {
    for_each(nodes, [this](const auto u) { std::ignore = adj_[u]; });
  }

  /// Construct graph from edge set.
  explicit Graph(const llvm::DenseSet<std::pair<size_t, size_t>>& edges) {
    for_each(edges, [this](const auto& e) { addEdge(e.first, e.second); });
  }

  /// Construct graph from node identifiers and edge set.
  explicit Graph(ArrayRef<size_t> nodes,
                 const llvm::DenseSet<std::pair<size_t, size_t>>& edges) {
    for_each(nodes, [this](const auto u) { std::ignore = adj_[u]; });
    for_each(edges, [this](const auto& e) { addEdge(e.first, e.second); });
  }

  /// Add a directed edge to the internal representation of the graph.
  /// Implicitly adds nodes.
  void addEdge(size_t u, size_t v);

  /// Return the neighbours of a node.
  [[nodiscard]] ArrayRef<size_t> getNeighbours(size_t id) const;

  /// Return the nodes.
  [[nodiscard]] SmallVector<size_t> getNodes() const;

  /// Return the number of nodes.
  [[nodiscard]] size_t getNumNodes() const { return adj_.size(); }

  /// Return the degree of a node.
  [[nodiscard]] size_t getDegree(const size_t id) const {
    return adj_.at(id).size();
  }

  /// Return the max degree of the graph.
  [[nodiscard]] size_t getMaxDegree() const;

  /// Return true if the graph has no nodes and edges.
  [[nodiscard]] bool empty() const { return adj_.empty(); }

  /// Clear the graph.
  void clear() { adj_.clear(); }

  /// Remove the edges from the graph. Keep the nodes.
  void clearEdges();

  /// Return the minimum distance matrix of the graph by implementing the
  /// Floyd-Warshall Algorithm
  /// (https://en.wikipedia.org/wiki/Floyd–Warshall_algorithm) where dist[i][j]
  /// denotes the distance between i and j.
  [[nodiscard]] Graph::DistanceMatrix getDistMatrix() const;

  /// Return cycle in graph or `std::nullopt` if none exists.
  /// Implements an iterative depth-first search inspired by LLVM's SCC
  /// utilities. For a cycle [A, B, C, A], the function returns [A, B, C].
  [[nodiscard]] std::optional<SmallVector<size_t>> findCycle() const;

private:
  llvm::DenseMap<size_t, SmallVector<size_t>> adj_;
};
} // namespace mlir
