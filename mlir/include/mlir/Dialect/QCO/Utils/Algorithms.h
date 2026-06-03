/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 MQSC GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <llvm/ADT/DenseSet.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <utility>

namespace mlir::qco {

using Matrix = SmallVector<SmallVector<size_t, 0>, 0>;
using Edges = llvm::DenseSet<std::pair<size_t, size_t>>;

/**
 * @brief Find all shortest paths between two nodes in a graph.
 * @details Has a time complexity of O(n^3).
 *
 * @link Adapted from https://en.wikipedia.org/wiki/Floyd–Warshall_algorithm
 *
 * @param n The number of nodes in the graph.
 * @param edges The set of edges (i, j).
 *
 * @returns The distance matrix dist, where dist[i, j] is defined as the
 * distance between node i and j.
 */
Matrix findAllShortestPaths(size_t n, const Edges& edges);

} // namespace mlir::qco
