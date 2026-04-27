/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QCO/Transforms/Mapping/Architecture.h"

#include <llvm/Support/ErrorHandling.h>
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>

using namespace mlir;
using namespace mlir::qco;

std::string_view Architecture::name() const { return name_; }

std::size_t Architecture::nqubits() const { return nqubits_; }

bool Architecture::areAdjacent(std::size_t u, std::size_t v) const {
  return couplingSet_.contains(std::make_pair(u, v));
}

std::size_t Architecture::distanceBetween(std::size_t u, std::size_t v) const {
  if (dist_[u][v] == UINT64_MAX) {
    report_fatal_error("Floyd-warshall failed to compute the distance "
                       "between qubits " +
                       Twine(u) + " and " + Twine(v));
  }
  return dist_[u][v];
}

SmallVector<std::size_t, 4> Architecture::neighboursOf(std::size_t u) const {
  return neighbours_[u];
}

void Architecture::floydWarshallWithPathReconstruction() {
  for (const auto& [u, v] : couplingSet_) {
    dist_[u][v] = 1;
    prev_[u][v] = u;
  }
  for (std::size_t v = 0; v < nqubits(); ++v) {
    dist_[v][v] = 0;
    prev_[v][v] = v;
  }

  for (std::size_t k = 0; k < nqubits(); ++k) {
    for (std::size_t i = 0; i < nqubits(); ++i) {
      for (std::size_t j = 0; j < nqubits(); ++j) {
        if (dist_[i][k] == UINT64_MAX || dist_[k][j] == UINT64_MAX) {
          continue; // Avoid overflow with "infinite" distances.
        }
        const std::size_t sum = dist_[i][k] + dist_[k][j];
        if (dist_[i][j] > sum) {
          dist_[i][j] = sum;
          prev_[i][j] = prev_[k][j];
        }
      }
    }
  }
}

void Architecture::collectNeighbours() {
  for (const auto& [u, v] : couplingSet_) {
    neighbours_[u].push_back(v);
  }
}

std::size_t Architecture::maxDegree() const {
  std::size_t deg = 0;
  for (const auto& nbrs : neighbours_) {
    deg = std::max(deg, nbrs.size());
  }
  return deg;
}
