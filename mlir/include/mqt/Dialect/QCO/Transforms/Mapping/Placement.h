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

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace mlir {
class Operation;
class Region;

namespace qco {

/// Simple undirected interactions, represented by two directed edges.
struct RegionConnectivity {
  struct Qubit {
    Value root;
    /// Tensor slot, or nullopt for a scalar qubit.
    std::optional<int64_t> slot;
  };
  Graph graph{ArrayRef<size_t>{}};
  /// Dense vertex order: block arguments, then roots in program order.
  ///
  /// Captured values become roots on their first use. Includes idle qubits.
  SmallVector<Qubit> qubits;
};

/// Build a connectivity graph without changing IR.
///
/// Requires a single-block region that passes verification and verifyLinearity.
/// Root values remain valid only while their IR exists.
/// Follows RegionBranchOpInterface edges, visiting each reachable single-block
/// region once, without evaluating conditions or unrolling loops.
/// Supports scalar QCO and statically sized qtensors with constant indices.
/// Incoming paths at joins and loop backedges must agree on wire identities,
/// including individual tensor slots.
/// Barriers add no edges; repeated interactions add one undirected edge.
/// Diagnoses dynamic tensor accesses, quantum calls, undecomposed gates with
/// more than two qubits, and unsupported operations carrying qubits or regions.
[[nodiscard]] FailureOr<RegionConnectivity>
buildConnectivityGraph(Region& region);

/// Cached connectivity of an operation with one region, such as func.func.
///
/// Uses buildConnectivityGraph's input contract. MLIR invalidates this analysis
/// unless a pass explicitly preserves it, including the stored root values.
class RegionConnectivityAnalysis {
public:
  explicit RegionConnectivityAnalysis(Operation* operation);

  /// Return the graph, or failure if the region is unsupported.
  [[nodiscard]] const FailureOr<RegionConnectivity>& getConnectivity() const {
    return connectivity_;
  }

private:
  FailureOr<RegionConnectivity> connectivity_ = failure();
};

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
/// Search depth is bounded by the pattern's arity (at most four).
[[nodiscard]] std::optional<SmallVector<size_t, 4>>
matchTopologyPattern(const Graph& graph,
                     const CompilerTarget::TopologyPatterns::Group& pattern);

/// Cover each graph edge once using available pattern types.
///
/// Prefers four-cycles, then stars, then pairs. Retains shared boundary
/// vertices while removing matched edges from a copy. Isolated vertices need no
/// slice. Ignores groups without target occurrences. Fails if the available
/// patterns cannot cover all edges. This does not yet join target occurrences
/// or construct a placement.
[[nodiscard]] FailureOr<SmallVector<TopologyPatternMatch>>
decomposeConnectivityGraph(
    Graph graph, ArrayRef<CompilerTarget::TopologyPatterns::Group> available);

} // namespace qco
} // namespace mlir
