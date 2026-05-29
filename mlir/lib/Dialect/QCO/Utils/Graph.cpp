/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Utils/Graph.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace mlir::qco {
Graph::Graph(const EdgeSet& edges) {
  for (const auto& [u, v] : edges) {
    if (!adj_.contains(u)) {
      adj_[u] = Vector<size_t>();
    }
    adj_[u].emplace_back(v);
  }
}
void Graph::addNode(size_t id) {
  const auto r = adj_.try_emplace(id, SmallVector<size_t, 0>{});
  assert(r.second && "expected to insert node");
}

void Graph::addEdge(size_t id, size_t neighbourId) {
  assert(adj_.contains(id) && "addEdge: missing node id");
  adj_[id].emplace_back(neighbourId);
}

void Graph::addEdge(std::pair<size_t, size_t> edge) {
  addEdge(edge.first, edge.second);
}

void Graph::addEdges(SmallVector<std::pair<size_t, size_t>> edges) {
  for_each(edges, [this](const auto& edge) { addEdge(edge); });
}

Graph::EdgeSet Graph::getEdges() const {
  EdgeSet set;
  for (const auto& [u, nbrs] : adj_) {
    for (const auto& v : nbrs) {
      set.insert(std::make_pair(u, v));
    }
  }
  return set;
}

ArrayRef<size_t> Graph::getEdges(size_t id) const { return adj_.at(id); }

size_t Graph::getMaxDegree() const {
  size_t deg = 0;
  for (const auto& [u, nbrs] : adj_) {
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

[[nodiscard]] std::optional<Vector<Graph::IdT>> Graph::findCycle() const {
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

      // Once all neighbours have been visited (indicated by the index exceeding
      // the number of neighbours - 1), set the frame on node to finished and
      // pop it from the stack.
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
        return path;
      }
    }

    // Preparse stack for next iteration.
    stack.clear();
  }

  return std::nullopt;
}
} // namespace mlir::qco
