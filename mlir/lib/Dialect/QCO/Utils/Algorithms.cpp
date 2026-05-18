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

#include <mlir/Support/LLVM.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/DenseSet.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace mlir::qco {
Matrix findAllShortestPaths(size_t n, const Edges& edges) {
  Matrix dist(n, SmallVector<size_t>(n, UINT64_MAX));

  for (const auto& [u, v] : edges) {
    dist[u][v] = 1;
  }
  for (std::size_t v = 0; v < n; ++v) {
    dist[v][v] = 0;
  }

  for (std::size_t k = 0; k < n; ++k) {
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        if (dist[i][k] == UINT64_MAX || dist[k][j] == UINT64_MAX) {
          continue; // Avoid overflow with "infinite" distances.
        }

        const std::size_t sum = dist[i][k] + dist[k][j];
        dist[i][j] = std::min(dist[i][j], sum);
      }
    }
  }

  return dist;
}
} // namespace mlir::qco
