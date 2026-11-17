/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Compiler/Target.h"
#include "mqt/Dialect/QCO/Transforms/Mapping/Placement.h"
#include "mqt/Dialect/QCO/Utils/Graph.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Threading.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <utility>

namespace mlir::qco {
namespace {
using Adjacency = SmallVector<SmallVector<size_t>, 0>;

struct SearchStep {
  size_t vertex;
  size_t degree;
  SmallVector<size_t> previous;
};

// Search frames borrow immutable candidate lists.
struct SearchFrame {
  ArrayRef<size_t> candidates;
  size_t next = 0;
};
} // namespace

// Prefer vertices with more assigned neighbors, then higher degree.
static SmallVector<SearchStep> planSearch(const Graph& graph,
                                          ArrayRef<size_t> nodes) {
  DenseMap<size_t, size_t> positions;
  for (const auto [position, node] : llvm::enumerate(nodes)) {
    positions[node] = position;
  }
  Adjacency adjacency(nodes.size());
  for (size_t vertex = 0; vertex < nodes.size(); ++vertex) {
    for (const auto neighbor : graph.getNeighbours(nodes[vertex])) {
      adjacency[vertex].push_back(positions.at(neighbor));
    }
    llvm::sort(adjacency[vertex]);
  }
  SmallVector<SearchStep> steps;
  llvm::BitVector selected(nodes.size());
  SmallVector<size_t> bound(nodes.size());
  for (size_t depth = 0; depth < nodes.size(); ++depth) {
    size_t best = nodes.size();
    std::pair<size_t, size_t> bestScore;
    for (size_t vertex = 0; vertex < nodes.size(); ++vertex) {
      const auto score = std::pair{bound[vertex], adjacency[vertex].size()};
      if (!selected[vertex] && (best == nodes.size() || score > bestScore)) {
        best = vertex;
        bestScore = score;
      }
    }
    SearchStep step{
        .vertex = best,
        .degree = adjacency[best].size(),
        .previous = {},
    };
    for (const auto neighbor : adjacency[best]) {
      if (selected[neighbor]) {
        step.previous.push_back(neighbor);
      }
      ++bound[neighbor];
    }
    selected.set(best);
    steps.push_back(std::move(step));
  }
  return steps;
}

static std::optional<SmallVector<size_t>>
searchLayouts(ArrayRef<SearchStep> steps, const Adjacency& adjacency,
              ArrayRef<size_t> sites, bool allToAll, size_t begin, size_t end,
              size_t maxCandidates,
              function_ref<bool(ArrayRef<size_t>)> visit) {
  SmallVector<size_t> layout(steps.size());
  llvm::BitVector used(sites.size());
  SmallVector<SearchFrame> frames(steps.size());
  frames[0] = {.candidates = sites.slice(begin, end - begin), .next = 0};
  size_t depth = 0;
  while (true) {
    auto& frame = frames[depth];
    const auto& step = steps[depth];
    if (frame.next == frame.candidates.size()) {
      if (depth == 0) {
        return std::nullopt;
      }
      used.reset(layout[steps[--depth].vertex]);
      continue;
    }
    if (maxCandidates == 0) {
      return std::nullopt;
    }
    --maxCandidates;
    const auto candidate = frame.candidates[frame.next++];
    if (used[candidate] ||
        (!allToAll && (adjacency[candidate].size() < step.degree ||
                       llvm::any_of(step.previous, [&](size_t neighbor) {
                         return !std::binary_search(
                             adjacency[candidate].begin(),
                             adjacency[candidate].end(), layout[neighbor]);
                       })))) {
      continue;
    }
    layout[step.vertex] = candidate;
    if (depth + 1 == steps.size()) {
      if (!visit || visit(layout)) {
        return layout;
      }
      continue;
    }
    used.set(candidate);
    const auto& next = steps[++depth];
    ArrayRef<size_t> candidates = sites;
    if (!allToAll) {
      for (const auto neighbor : next.previous) {
        const auto& adjacent = adjacency[layout[neighbor]];
        if (adjacent.size() < candidates.size()) {
          candidates = adjacent;
        }
      }
    }
    frames[depth] = {.candidates = candidates, .next = 0};
  }
}

static std::optional<SmallVector<size_t>>
findTopologyLayoutImpl(const Graph& graph, const CompilerTarget& target,
                       MLIRContext& context, size_t maxCandidates,
                       function_ref<bool(ArrayRef<size_t>)> visit = nullptr) {
  auto nodes = graph.getNodes();
  if (nodes.size() > target.numSites()) {
    return std::nullopt;
  }
  llvm::sort(nodes);
  if (nodes.empty()) {
    if (visit) {
      visit({});
    }
    return SmallVector<size_t>{};
  }
  const auto steps = planSearch(graph, nodes);
  const bool allToAll =
      target.connectivityKind() == CompilerTarget::Connectivity::Kind::AllToAll;
  Adjacency adjacency;
  if (!allToAll) {
    adjacency.resize(target.numSites());
    size_t maximumDegree = 0;
    for (size_t vertex = 0; vertex < target.numSites(); ++vertex) {
      target.forEachNeighbour(vertex, [&](size_t neighbor) {
        adjacency[vertex].push_back(neighbor);
      });
      llvm::sort(adjacency[vertex]);
      maximumDegree = std::max(maximumDegree, adjacency[vertex].size());
    }
    if (steps.front().degree > maximumDegree) {
      return std::nullopt;
    }
  }
  const auto sites = llvm::to_vector(llvm::seq<size_t>(0, target.numSites()));
  // Scheduling dominates small searches; a single vertex has no join work.
  // Fixed ranges keep bounded-search results independent of thread count.
  const auto tasks = !visit && sites.size() >= 64 && steps.size() > 1
                         ? std::min(sites.size(), size_t{32})
                         : size_t{1};
  SmallVector<std::optional<SmallVector<size_t>>> results(tasks);
  parallelFor(&context, 0, tasks, [&](size_t task) {
    const auto begin =
        sites.size() / tasks * task + std::min(sites.size() % tasks, task);
    const auto end =
        begin + sites.size() / tasks + (task < sites.size() % tasks ? 1 : 0);
    const auto budget =
        maxCandidates / tasks + (task < maxCandidates % tasks ? 1 : 0);
    results[task] = searchLayouts(steps, adjacency, sites, allToAll, begin, end,
                                  budget, visit);
  });
  for (auto& result : results) {
    if (result) {
      return std::move(result);
    }
  }
  return std::nullopt;
}

std::optional<SmallVector<size_t>>
findTopologyLayout(const Graph& graph, const CompilerTarget& target,
                   MLIRContext& context, size_t maxCandidates) {
  return findTopologyLayoutImpl(graph, target, context, maxCandidates);
}

void enumerateTopologyLayouts(const Graph& graph, const CompilerTarget& target,
                              MLIRContext& context, size_t maxCandidates,
                              function_ref<bool(ArrayRef<size_t>)> visit) {
  std::ignore =
      findTopologyLayoutImpl(graph, target, context, maxCandidates, visit);
}

} // namespace mlir::qco
