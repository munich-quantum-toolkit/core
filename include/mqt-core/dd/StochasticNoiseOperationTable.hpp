/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "dd/statistics/TableStatistics.hpp"
#include "ir/Definitions.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dd {
template <class Edge, std::size_t numberOfStochasticOperations = 64>
class StochasticNoiseOperationTable {
public:
  explicit StochasticNoiseOperationTable(const std::size_t nv) : nvars(nv) {
    resize(nv);
    stats.entrySize = sizeof(Edge);
    stats.numBuckets = nv * numberOfStochasticOperations;
  };

  /// Get a reference to the table
  [[nodiscard]] const auto& getTable() const { return table; }

  /// Get a reference to the statistics
  [[nodiscard]] const auto& getStats() const noexcept { return stats; }

  void resize(std::size_t nq) {
    nvars = nq;
    table.resize(nvars);
  }

  void insert(std::uint8_t kind, qc::Qubit target, const Edge& r) {
    assert(kind <
           numberOfStochasticOperations); // There are new operations in OpType.
                                          // Increase the value of
                                          // numberOfOperations accordingly
    table.at(target).at(kind) = r;
    stats.trackInsert();
  }

  Edge* lookup(std::uint8_t kind, qc::Qubit target) {
    assert(kind <
           numberOfStochasticOperations); // There are new operations in OpType.
                                          // Increase the value of
                                          // numberOfOperations accordingly
    ++stats.lookups;
    Edge* r = nullptr;
    auto& entry = table.at(target).at(kind);
    if (entry.w.r == nullptr) {
      return r;
    }
    ++stats.hits;
    return &entry;
  }

  void clear() {
    if (stats.numEntries > 0) {
      for (auto& t : table) {
        std::fill(t.begin(), t.end(), Edge{});
      }
      stats.numEntries = 0;
    }
  }

private:
  std::size_t nvars;
  std::vector<std::array<Edge, numberOfStochasticOperations>> table;
  TableStatistics stats{};
};
} // namespace dd
