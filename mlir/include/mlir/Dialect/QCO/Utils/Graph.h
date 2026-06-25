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
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <utility>

namespace mlir::qco {
template <class T> using Vector = SmallVector<T, 0>;
template <class T> using Matrix = Vector<Vector<T>>;

/// A directed graph.
template <class IdT = size_t> class Graph {
public:
  /// Construct an empty graph.
  Graph() = default;

  /// Construct graph from edge set.
  explicit Graph(const llvm::DenseSet<std::pair<IdT, IdT>>& edges) {
    for_each(edges, [this](const auto& e) { addEdge(e); });
  }

  /// Add a directed edge to the internal representation of the graph.
  /// Implicitly adds nodes.
  void addEdge(IdT u, IdT v) {
    if (!adj_.contains(u)) {
      adj_[u] = Vector<IdT>();
    }
    if (!adj_.contains(v)) {
      adj_[v] = Vector<IdT>();
    }
    adj_[u].emplace_back(v);
  }

  /// Add an edge to the graph.
  void addEdge(std::pair<IdT, IdT> edge) { addEdge(edge.first, edge.second); }

  /// Return a set of edges.
  [[nodiscard]] llvm::DenseSet<std::pair<IdT, IdT>> getEdges() const {
    llvm::DenseSet<std::pair<IdT, IdT>> edges;
    for (const auto& [u, nbrs] : adj_) {
      for (const auto& v : nbrs) {
        edges.insert(std::make_pair(u, v));
      }
    }
    return edges;
  }

  /// Return the edges of a node.
  [[nodiscard]] ArrayRef<IdT> getEdges(size_t id) const { return adj_.at(id); }

  /// Return the nodes.
  [[nodiscard]] Vector<IdT> getNodes() const { return to_vector(adj_.keys()); }

  /// Return the number of nodes.
  [[nodiscard]] size_t getNumNodes() const { return adj_.size(); }

  /// Return the degree of a node.
  [[nodiscard]] size_t getDegree(size_t id) { return adj_.at(id).size(); }

  /// Return the max degree of the graph.
  [[nodiscard]] size_t getMaxDegree() const {
    size_t deg = 0;
    for (const auto& [u, nbrs] : adj_) {
      deg = std::max(deg, nbrs.size());
    }
    return deg;
  }

  /// Return true if the graph has no nodes and edges.
  [[nodiscard]] bool empty() const { return adj_.empty(); }

  /// Clear the graph.
  [[nodiscard]] void clear() { adj_.clear(); }

  /// Return the minimum distance matrix of the graph by implementing the
  /// Floyd-Warshall Algorithm
  /// (https://en.wikipedia.org/wiki/Floyd–Warshall_algorithm) where dist[i][j]
  /// denotes the distance between i and j.
  [[nodiscard]] Matrix<size_t> getDistMatrix() const {
    const auto n = getNumNodes();

    SmallVector<std::pair<IdT, IdT>> edges;
    for (const auto& [u, nbrs] : adj_) {
      for (const auto& v : nbrs) {
        edges.emplace_back(u, v);
      }
    }

    Matrix<size_t> dist(n, Vector<size_t>(n, UINT64_MAX));
    for (const auto& [u, v] : edges) {
      dist[u][v] = 1;
    }
    for (size_t v = 0; v < n; ++v) {
      dist[v][v] = 0;
    }

    for (size_t k = 0; k < n; ++k) {
      for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
          if (dist[i][k] == UINT64_MAX || dist[k][j] == UINT64_MAX) {
            continue; // Avoid overflow with "infinite" distances.
          }

          const size_t sum = dist[i][k] + dist[k][j];
          dist[i][j] = std::min(dist[i][j], sum);
        }
      }
    }

    return dist;
  }

  /// Return cycle in graph or `std::nullopt` if none exists.
  /// Implements an iterative depth-first search inspired by LLVM's SCC
  /// utilities. For a cycle [A, B, C, A], the function returns [A, B, C].
  [[nodiscard]] std::optional<Vector<IdT>> findCycle() const {
    enum struct State : uint8_t { Unseen, Seen, Finished };

    struct Frame {
      IdT id;
      size_t neighbourIdx;
    };

    SmallVector<Frame> stack;
    llvm::DenseMap<IdT, IdT> parents;
    llvm::DenseMap<IdT, State> states;

    // Preparation step: Mark all nodes as unseen.
    for_each(adj_.keys(), [&](IdT id) { states[id] = State::Unseen; });

    for (const auto initId : adj_.keys()) {
      // Only start from unseen nodes.
      if (states[initId] != State::Unseen) {
        continue;
      }

      stack.emplace_back(initId, 0);

      while (!stack.empty()) {
        Frame& top = stack.back();

        // If we haven't seen this node before, mark it as seen.
        if (states[top.id] == State::Unseen) {
          states[top.id] = State::Seen;
        }

        auto it = adj_.find(top.id);
        assert(it != adj_.end() && "expected node id in adjacency map");
        const auto nbrs = it->getSecond();

        // Once all neighbours have been visited (indicated by the index
        // exceeding the number of neighbours - 1), set the frame on node to
        // finished and pop it from the stack.
        if (top.neighbourIdx >= nbrs.size()) {
          states[top.id] = State::Finished;
          stack.pop_back();
          continue;
        }

        // Collect the neighbour and advance the index on the
        // frame for the next iteration.
        const auto nbrId = nbrs[top.neighbourIdx];
        ++top.neighbourIdx;

        if (states[nbrId] == State::Unseen) {
          parents[nbrId] = top.id;
          stack.emplace_back(nbrId, 0);
        } else if (states[nbrId] == State::Seen) {
          SmallVector<IdT> path;
          for (auto curr = top.id; curr != nbrId; curr = parents[curr]) {
            path.emplace_back(curr);
          }
          path.emplace_back(nbrId);
          std::ranges::reverse(path);
          return path;
        }
      }

      // Preparse stack for next iteration.
      stack.clear();
    }

    return std::nullopt;
  }

private:
  llvm::DenseMap<IdT, Vector<IdT>> adj_;
};
} // namespace mlir::qco
