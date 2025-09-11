/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Architecture.h"

#include <cstddef>
#include <cstdint>
#include <format>
#include <llvm/ADT/SmallVector.h>
#include <stdexcept>
#include <string>

namespace mqt::ir::opt {
[[nodiscard]] llvm::SmallVector<std::size_t>
Architecture::shortestPathBetween(std::size_t u, std::size_t v) const {
  llvm::SmallVector<std::size_t> path;

  if (prev_[u][v] == UINT64_MAX) {
    return {};
  }

  path.push_back(v);
  while (u != v) {
    v = prev_[u][v];
    path.push_back(v);
  }

  return path;
}

void Architecture::floydWarshallWithPathReconstruction() {
  for (const auto& [u, v] : couplingMap_) {
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
          continue; // avoid overflow with "infinite" distances
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

Architecture getArchitecture(const ArchitectureName& name) {
  switch (name) {
  case ArchitectureName::MQTTest: {
    // 0 -- 1
    // |    |
    // 2 -- 3
    // |    |
    // 4 -- 5

    const Architecture::CouplingMap couplingMap{
        {0, 1}, {1, 0}, {0, 2}, {2, 0}, {1, 3}, {3, 1}, {2, 3},
        {3, 2}, {2, 4}, {4, 2}, {3, 5}, {5, 3}, {4, 5}, {5, 4}};

    return Architecture("MQT-Test", 6, couplingMap);
  }
  }

  throw std::invalid_argument(std::format("Unsupported architecture."));
}
}; // namespace mqt::ir::opt
