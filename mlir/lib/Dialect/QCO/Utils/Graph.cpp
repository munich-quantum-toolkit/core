/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Support/Graph.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

namespace mlir {
void Graph::addEdge(size_t u, size_t v) {
  adj_[u].emplace_back(v);
  std::ignore = adj_[v]; // Ensure v exists in the map.
}

ArrayRef<size_t> Graph::getNeighbours(size_t id) const { return adj_.at(id); }
SmallVector<size_t> Graph::getNodes() const { return to_vector(adj_.keys()); }

size_t Graph::getMaxDegree() const {
  size_t deg = 0;
  for (const auto& [u, nbrs] : adj_) {
    deg = std::ranges::max(deg, nbrs.size());
  }
  return deg;
}

void Graph::clearEdges() {
  for_each(adj_, [](auto& kv) { kv.second.clear(); });
}

Graph::DistanceMatrix Graph::getDistMatrix() const {
  const auto n = getNumNodes();

  Graph::DistanceMatrix dist(n, std::numeric_limits<size_t>::max());
  for (const auto& [u, nbrs] : adj_) {
    for (const auto& v : nbrs) {
      dist[u][v] = 1;
    }
  }
  for (size_t v = 0; v < n; ++v) {
    dist[v][v] = 0;
  }

  for (size_t k = 0; k < n; ++k) {
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        if (dist[i][k] == std::numeric_limits<size_t>::max() ||
            dist[k][j] == std::numeric_limits<size_t>::max()) {
          continue; // Avoid overflow with "infinite" distances.
        }

        const size_t sum = dist[i][k] + dist[k][j];
        dist[i][j] = std::min(dist[i][j], sum);
      }
    }
  }

  return dist;
}

std::optional<SmallVector<size_t>> Graph::findCycle() const {
  enum struct State : uint8_t { Unseen, Seen, Finished };

  struct Frame {
    size_t id;
    size_t neighbourIdx;
  };

  SmallVector<Frame> stack;
  llvm::DenseMap<size_t, size_t> parents;
  llvm::DenseMap<size_t, State> states;

  // Preparation step: Mark all nodes as unseen.
  llvm::for_each(adj_.keys(), [&](size_t id) { states[id] = State::Unseen; });

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
        SmallVector<size_t> path;
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

} // namespace mlir
