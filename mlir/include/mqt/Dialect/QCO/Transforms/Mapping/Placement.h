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

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <optional>

namespace mlir {
class CompilerTarget;
class MLIRContext;

namespace qco {
class Graph;

/// Find one injective layout within a candidate-assignment budget.
///
/// The graph must be simple, undirected (symmetric), and loop-free. The result
/// maps graph vertices in ascending ID order to dense target vertices, not site
/// IDs. Includes disconnected components and isolated vertices. An empty graph
/// has one empty layout. Native operation availability and calibration are not
/// considered; extra target edges are allowed.
///
/// Search counts every attempted vertex assignment, including rejected
/// candidates. Independent starting ranges use the context's thread pool and
/// fixed budgets, so output is independent of thread count. Returns nullopt
/// when no layout is found, including budget exhaustion; this does not prove
/// infeasibility. This helper does not change IR.
[[nodiscard]] std::optional<SmallVector<size_t>>
findTopologyLayout(const Graph& graph, const CompilerTarget& target,
                   MLIRContext& context, size_t maxCandidates);

} // namespace qco
} // namespace mlir
