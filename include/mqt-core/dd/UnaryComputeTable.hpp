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
 * @file UnaryComputeTable.hpp
 * @brief Data structure for caching computed results of unary operations
 */

#pragma once

#include "dd/statistics/TableStatistics.hpp"

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <vector>

namespace dd {

/**
 * @brief Data structure for caching computed results of unary operations
 * @tparam OperandType type of the operation's operand
 * @tparam ResultType type of the operation's result
 */
template <class OperandType, class ResultType, std::size_t NBUCKET = 32768>
class UnaryComputeTable {
public:
  /// Default constructor
  explicit UnaryComputeTable(const size_t numBuckets = 32768U) {
    // numBuckets must be a power of two
    if ((numBuckets & (numBuckets - 1)) != 0) {
      throw std::invalid_argument("Number of buckets must be a power of two.");
    }
    stats.entrySize = sizeof(Entry);
    stats.numBuckets = numBuckets;
    valid = std::vector(numBuckets, false);
    table = std::vector<Entry>(numBuckets);
  }

  /// An entry in the compute table
  struct Entry {
    OperandType operand;
    ResultType result;
  };

  /// Get a reference to the underlying table
  [[nodiscard]] const auto& getTable() const { return table; }

  /// Get a reference to the statistics
  [[nodiscard]] const auto& getStats() const noexcept { return stats; }

  /// Compute the hash value for a given operand
  [[nodiscard]] std::size_t hash(const OperandType& a) const {
    const auto mask = stats.numBuckets - 1;
    return std::hash<OperandType>{}(a)&mask;
  }

  /**
   * @brief Insert a new entry into the compute table
   * @details Any existing entry for the resulting hash value will be replaced.
   * @param operand The operand
   * @param result The result of the operation
   */
  void insert(const OperandType& operand, const ResultType& result) {
    const auto key = hash(operand);
    if (valid[key]) {
      ++stats.collisions;
    } else {
      stats.trackInsert();
      valid[key] = true;
    }
    table[key] = {operand, result};
  }

  /**
   * @brief Look up a result in the compute table
   * @param operand The operand
   * @return A pointer to the result if it is found, otherwise nullptr.
   */
  ResultType* lookup(const OperandType& operand) {
    ResultType* result = nullptr;
    ++stats.lookups;
    const auto key = hash(operand);

    if (!valid[key]) {
      return result;
    }

    auto& entry = table[key];
    if (entry.operand != operand) {
      return result;
    }

    ++stats.hits;
    return &entry.result;
  }

  /**
   * @brief Clear the compute table
   * @details Sets all entries to invalid.
   */
  void clear() { valid = std::vector(NBUCKET, false); }

private:
  /// The actual table storing the entries
  std::vector<Entry> table;
  /// Dynamic bitset to mark valid entries
  std::vector<bool> valid;
  /// Statistics of the compute table
  TableStatistics stats{};
};
} // namespace dd
