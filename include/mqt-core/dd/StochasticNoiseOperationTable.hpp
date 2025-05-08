/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/**
 * @file StochasticNoiseOperationTable.hpp
 * @brief Data structure for caching computed results of stochastic operations
 */

#pragma once

#include "dd/statistics/TableStatistics.hpp"
#include "ir/Definitions.hpp"
#include "ir/operations/OpType.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dd {
template <class Edge> class StochasticNoiseOperationTable {
public:
  explicit StochasticNoiseOperationTable(
      const std::size_t nv,
      const size_t numberOfStochasticOperations = qc::OpType::OpTypeEnd)
      : nvars(nv), numberOfStochasticOperations_(numberOfStochasticOperations),
        table(nv, std::vector<Edge>(numberOfStochasticOperations)) {
    stats.entrySize = sizeof(Edge);
    stats.numBuckets = nv * numberOfStochasticOperations;
  }

  /// Get a reference to the table
  [[nodiscard]] const auto& getTable() const { return table; }

  /// Get a reference to the statistics
  [[nodiscard]] const auto& getStats() const noexcept { return stats; }

  void resize(const std::size_t nq) {
    nvars = nq;
    table.resize(nvars, std::vector<Edge>(numberOfStochasticOperations_));
  }

  void insert(std::uint8_t kind, qc::Qubit target, const Edge& r) {
    assert(kind <
           numberOfStochasticOperations_); // There are new operations in
                                           // OpType. Increase the value of
                                           // numberOfOperations accordingly
    table.at(target).at(kind) = r;
    stats.trackInsert();
  }

  Edge* lookup(std::uint8_t kind, qc::Qubit target) {
    assert(kind <
           numberOfStochasticOperations_); // There are new operations in
                                           // OpType. Increase the value of
                                           // numberOfOperations accordingly
    ++stats.lookups;
    auto& entry = table.at(target).at(kind);
    if (entry.w.r == nullptr) {
      return nullptr;
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
  size_t numberOfStochasticOperations_;
  std::vector<std::vector<Edge>> table;
  TableStatistics stats{};
};
} // namespace dd
