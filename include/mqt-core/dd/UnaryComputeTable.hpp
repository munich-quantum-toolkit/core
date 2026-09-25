/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file UnaryComputeTable.hpp
/// Data structure for caching computed results of unary operations

#pragma once

#include "dd/DDDefinitions.hpp"
#include "dd/statistics/TableStatistics.hpp"

#include "support/Diagnostics.hpp"

#include "mlir/Support/LogicalResult.h"

#include <bit>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <vector>

namespace dd {

/// Data structure for caching computed results of unary operations
/// @tparam OperandType type of the operation's operand
/// @tparam ResultType type of the operation's result
template <class OperandType, class ResultType> class UnaryComputeTable {
public:
  /// Default number of buckets for the compute table
  static constexpr std::size_t DEFAULT_NUM_BUCKETS = 32768U;

  UnaryComputeTable() : UnaryComputeTable(DEFAULT_NUM_BUCKETS) {}

  [[nodiscard]] static mlir::FailureOr<UnaryComputeTable>
  create(const size_t numBuckets) {
    if (!std::has_single_bit(numBuckets)) {
      return ::mqt::emitError("Number of buckets must be a power of two.",
                              ::mqt::ErrorCategory::InvalidArgument);
    }
    return UnaryComputeTable(numBuckets);
  }

private:
  friend class Package;
  explicit UnaryComputeTable(const size_t numBuckets) {
    stats.entrySize = sizeof(Entry);
    stats.numBuckets = numBuckets;
    valid = std::vector(numBuckets, false);
    table = std::vector<Entry>(numBuckets);
  }

public:
  /// An entry in the compute table
  struct Entry {
    OperandType operand{};
    ResultType result{};
  };

  /// Get a reference to the underlying table
  [[nodiscard]] const auto& getTable() const { return table; }

  /// Get a reference to the statistics
  [[nodiscard]] const auto& getStats() const noexcept { return stats; }

  /// Compute the hash value for a given operand
  [[nodiscard]] std::size_t hash(const OperandType& a) const {
    const auto key = std::hash<OperandType>{}(a);
    const auto mask = stats.numBuckets - 1;
    /// Mix aligned addresses before reducing to the power-of-two bucket count.
    return (std::is_pointer_v<OperandType> ? murmur64(key) : key) & mask;
  }

  /// Insert a new entry into the compute table
  ///
  /// Any existing entry for the resulting hash value will be replaced.
  /// @param operand The operand
  /// @param result The result of the operation
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

  /// Look up a result in the compute table
  /// @param operand The operand
  /// @return A pointer to the result if it is found, otherwise nullptr.
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

  /// Clear the compute table
  ///
  /// Sets all entries to invalid.
  void clear() {
    valid.assign(valid.size(), false);
    stats.reset();
  }

private:
  /// The actual table storing the entries
  std::vector<Entry> table;
  /// Dynamic bitset to mark valid entries
  std::vector<bool> valid;
  /// Statistics of the compute table
  TableStatistics stats{};
};
} // namespace dd
