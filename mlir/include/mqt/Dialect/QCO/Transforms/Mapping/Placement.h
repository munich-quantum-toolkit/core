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

#include "mqt/Compiler/Target.h"
#include "mqt/Dialect/QCO/Utils/Graph.h"

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include <cstddef>
#include <optional>

namespace mlir::qco {

struct TopologyPatternMatch {
  CompilerTarget::TopologyPatterns::Type type;
  /// Graph vertices in the target pattern's template order.
  ///
  /// Equal vertex IDs across slices constrain a later layout search.
  SmallVector<size_t, 4> vertices;
};

/// Find one non-induced match using VF2 frontier and feasibility checks.
///
/// The graph must be simple, undirected (symmetric), and loop-free.
/// Returns vertices in template order, independent of graph insertion order.
/// Search depth is bounded by the pattern's arity.
[[nodiscard]] std::optional<SmallVector<size_t, 4>>
matchTopologyPattern(const Graph& graph,
                     const CompilerTarget::TopologyPatterns::Group& pattern);

/// Cover each graph edge once using available pattern types.
///
/// Prefers larger arity, then more edges, then the smaller pattern type.
/// Retains shared boundary vertices while removing matched edges from a copy.
/// Isolated vertices need no slice. Ignores groups without target occurrences.
/// Fails if the available patterns cannot cover all edges. This does not join
/// target occurrences or construct a placement.
[[nodiscard]] FailureOr<SmallVector<TopologyPatternMatch>>
decomposeConnectivityGraph(
    Graph graph, ArrayRef<CompilerTarget::TopologyPatterns::Group> available);

/// Cover graph edges using independently cached pattern groups.
///
/// Uses the same ordering and coverage contract as the contiguous-group
/// overload. Null pointers are ignored, allowing unavailable optional groups.
[[nodiscard]] FailureOr<SmallVector<TopologyPatternMatch>>
decomposeConnectivityGraph(
    Graph graph,
    ArrayRef<const CompilerTarget::TopologyPatterns::Group*> available);

/// Find one injective, non-induced layout within a candidate-assignment budget.
///
/// The graph must be simple, undirected (symmetric), and loop-free. The result
/// maps graph vertices in ascending ID order to dense target vertices, not site
/// IDs. Includes disconnected components and isolated vertices. An empty graph
/// has one empty layout. Native operation availability and calibration are not
/// considered.
///
/// Fuses edge joins and injectivity filtering in a vertex-at-a-time search,
/// without materializing pattern occurrences or intermediate joins. Independent
/// search branches use the context's thread pool; disabling context threading
/// selects serial execution. Small searches also run serially. The search
/// retains at most one layout per worker task and divides maxCandidates across
/// fixed starting ranges, independent of thread count. Returns nullopt when no
/// layout is found, including budget exhaustion; this does not prove
/// infeasibility. This helper does not change IR.
[[nodiscard]] std::optional<SmallVector<size_t>>
findTopologyLayout(const Graph& graph, const CompilerTarget& target,
                   MLIRContext& context, size_t maxCandidates);

/// Find a bounded layout using selected topology patterns and the VF2 cover.
///
/// Uses findTopologyLayout's graph and layout contract. Canonical occurrences
/// share the target's lazy caches; oriented tuple indexes are built per call.
/// Unavailable groups are ignored. Returns nullopt if the selected groups do
/// not cover all edges or the search exhausts maxCandidates. An empty selection
/// supports only edgeless queries or all-to-all targets. Include QubitPair to
/// cover edges that do not fit larger patterns.
///
/// A candidate is one pattern tuple or isolated-vertex assignment. Preparation
/// and decomposition are outside this budget. Fixed search ranges share unused
/// quota in deterministic parallel waves; output is independent of thread
/// count.
[[nodiscard]] std::optional<SmallVector<size_t>> findTopologyPatternLayout(
    const Graph& graph, const CompilerTarget& target, MLIRContext& context,
    size_t maxCandidates,
    ArrayRef<CompilerTarget::TopologyPatterns::Type> selected);

} // namespace mlir::qco
