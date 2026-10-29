/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/Transforms/Mapping/Placement.h"

#include "mqt/Compiler/Target.h"
#include "mqt/Dialect/QCO/Utils/Graph.h"

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include <cstddef>
#include <optional>
#include <utility>

namespace mlir::qco {
namespace {
using PatternGroup = CompilerTarget::TopologyPatterns::Group;
} // namespace

static bool extendTopologyPattern(const Graph& graph,
                                  const PatternGroup& pattern,
                                  ArrayRef<size_t> nodes,
                                  SmallVectorImpl<size_t>& match) {
  // All templates have a connected prefix, so extend along its frontier.
  const auto depth = match.size();
  if (depth == pattern.arity()) {
    return true;
  }
  ArrayRef<size_t> candidateRange = nodes;
  const auto neighbours = pattern.neighbours(depth);
  for (const auto previous : neighbours) {
    if (previous < depth) {
      candidateRange = graph.getNeighbours(match[previous]);
      break;
    }
  }
  auto candidates = llvm::to_vector(candidateRange);
  llvm::sort(candidates);
  for (const auto candidate : candidates) {
    if (llvm::is_contained(match, candidate) ||
        graph.getDegree(candidate) < neighbours.size()) {
      continue;
    }
    if (llvm::any_of(neighbours, [&](size_t other) {
          return other < depth && !graph.hasEdge(candidate, match[other]);
        })) {
      continue;
    }
    size_t patternFrontier = 0;
    size_t patternRemaining = 0;
    for (const auto next : neighbours) {
      if (next > depth) {
        ++patternRemaining;
        if (llvm::any_of(pattern.neighbours(next),
                         [&](size_t n) { return n < depth; })) {
          ++patternFrontier;
        }
      }
    }
    size_t graphFrontier = 0;
    size_t graphRemaining = 0;
    for (const auto next : graph.getNeighbours(candidate)) {
      if (!llvm::is_contained(match, next)) {
        ++graphRemaining;
        if (llvm::any_of(match,
                         [&](size_t n) { return graph.hasEdge(next, n); })) {
          ++graphFrontier;
        }
      }
    }
    // Non-induced matches may map new pattern vertices into the graph
    // frontier.
    if (patternFrontier > graphFrontier || patternRemaining > graphRemaining) {
      continue;
    }
    match.push_back(candidate);
    if (extendTopologyPattern(graph, pattern, nodes, match)) {
      return true;
    }
    match.pop_back();
  }
  return false;
}

std::optional<SmallVector<size_t, 4>>
matchTopologyPattern(const Graph& graph, const PatternGroup& pattern) {
  if (graph.getNumNodes() < pattern.arity()) {
    return std::nullopt;
  }
  const auto nodes = graph.getNodes();
  SmallVector<size_t, 4> match;
  if (extendTopologyPattern(graph, pattern, nodes, match)) {
    return match;
  }
  return std::nullopt;
}

FailureOr<SmallVector<TopologyPatternMatch>>
decomposeConnectivityGraph(Graph graph, ArrayRef<PatternGroup> available) {
  auto groups = llvm::map_to_vector(
      available, [](const PatternGroup& pattern) { return &pattern; });
  return decomposeConnectivityGraph(std::move(graph), groups);
}

FailureOr<SmallVector<TopologyPatternMatch>>
decomposeConnectivityGraph(Graph graph,
                           ArrayRef<const PatternGroup*> available) {
  SmallVector<TopologyPatternMatch> slices;
  SmallVector<const PatternGroup*> patterns;
  for (const auto* pattern : available) {
    if (pattern != nullptr && pattern->size() != 0) {
      patterns.push_back(pattern);
    }
  }
  llvm::sort(patterns, [](const auto* lhs, const auto* rhs) {
    if (lhs->arity() != rhs->arity()) {
      return lhs->arity() > rhs->arity();
    }
    if (lhs->edges().size() != rhs->edges().size()) {
      return lhs->edges().size() > rhs->edges().size();
    }
    return lhs->type() < rhs->type();
  });
  // Removing edges cannot create a new non-induced match of an exhausted type.
  for (const auto* pattern : patterns) {
    while (auto match = matchTopologyPattern(graph, *pattern)) {
      for (const auto& [a, b] : pattern->edges()) {
        graph.removeEdge((*match)[a], (*match)[b]);
        graph.removeEdge((*match)[b], (*match)[a]);
      }
      slices.push_back(
          {.type = pattern->type(), .vertices = std::move(*match)});
    }
  }
  if (llvm::any_of(graph.getNodes(),
                   [&](size_t n) { return graph.getDegree(n) != 0; })) {
    return failure();
  }
  return slices;
}

} // namespace mlir::qco
