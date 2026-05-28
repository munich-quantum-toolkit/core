/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Utils/Algorithms.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace mlir::qco {
Graph::Graph(const EdgeSet& edges) {
  for (const auto& [u, v] : edges) {
    if (!nodes_.contains(u)) {
      nodes_[u] = Vector<size_t>();
    }
    nodes_[u].emplace_back(v);
  }
}
void Graph::addNode(size_t id) {
  const auto r = nodes_.try_emplace(id, SmallVector<size_t, 0>{});
  assert(r.second && "addNode: didn't insert node");
}

void Graph::addEdge(size_t id, size_t neighbourId) {
  assert(nodes_.contains(id) && "addEdge: missing node id");
  nodes_[id].emplace_back(neighbourId);
}

void Graph::addEdge(std::pair<size_t, size_t> edge) {
  addEdge(edge.first, edge.second);
}

void Graph::addEdges(SmallVector<std::pair<size_t, size_t>> edges) {
  for_each(edges, [this](const auto& edge) { addEdge(edge); });
}

Graph::EdgeSet Graph::getEdges() const {
  EdgeSet set;
  for (const auto& [u, nbrs] : nodes_) {
    for (const auto& v : nbrs) {
      set.insert(std::make_pair(u, v));
    }
  }
  return set;
}

ArrayRef<size_t> Graph::getEdges(size_t id) const { return nodes_.at(id); }

size_t Graph::getMaxDegree() const {
  size_t deg = 0;
  for (const auto& [u, nbrs] : nodes_) {
    deg = std::max(deg, nbrs.size());
  }
  return deg;
}

Matrix<size_t> Graph::getDistMatrix() const {
  const size_t n = getNumNodes();
  const auto edges = getEdges();

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
} // namespace mlir::qco
