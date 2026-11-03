/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file ComputeTable.hpp
/// Data structure for caching computed results of binary operations

#pragma once

#include "dd/DDDefinitions.hpp"
#include "dd/statistics/TableStatistics.hpp"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <bit>
#include <cstddef>
#include <system_error>
#include <vector>

namespace dd {

/// Data structure for caching computed results of binary operations
///
/// @tparam LeftOperandType type of the operation's left operand
/// @tparam RightOperandType type of the operation's right operand
/// @tparam ResultType type of the operation's result
template <class LeftOperandType, class RightOperandType, class ResultType>
class ComputeTable {
public:
  /// Default number of buckets for the compute table
  static constexpr std::size_t DEFAULT_NUM_BUCKETS = 16384U;

  ComputeTable() : ComputeTable(DEFAULT_NUM_BUCKETS) {}

  [[nodiscard]] static llvm::Expected<ComputeTable>
  create(const size_t numBuckets) {
    if (!std::has_single_bit(numBuckets)) {
      return llvm::createStringError(
          std::make_error_code(std::errc::invalid_argument),
          "Number of buckets must be a power of two.");
    }
    return ComputeTable(numBuckets);
  }

private:
  friend class Package;
  explicit ComputeTable(const size_t numBuckets) {
    stats.entrySize = sizeof(Entry);
    stats.numBuckets = numBuckets;
  }

public:
  /// An entry in the compute table
  ///
  /// A triple consisting of the left operand, the right operand, and
  /// the result of a binary operation.
  struct Entry {
    LeftOperandType leftOperand{};
    RightOperandType rightOperand{};
    ResultType result{};
  };

  /// Compute the hash value for a given pair of operands
  ///
  /// @param leftOperand The left operand
  /// @param rightOperand The right operand
  /// @return The hash value
  [[nodiscard]] std::size_t hash(const LeftOperandType& leftOperand,
                                 const RightOperandType& rightOperand) const {
    const auto hash = combineHash(hashComputeOperand(leftOperand),
                                  hashComputeOperand(rightOperand));
    const auto mask = stats.numBuckets - 1;
    return hash & mask;
  }

  /// Get the underlying storage, empty until the first insertion.
  [[nodiscard]] const auto& getTable() const { return table; }

  /// Get a reference to the statistics
  [[nodiscard]] const auto& getStats() const noexcept { return stats; }

  /// Insert a new entry into the compute table
  ///
  /// Any existing entry for the resulting hash value will be replaced.
  /// @param leftOperand The left operand
  /// @param rightOperand The right operand
  /// @param result The result of the operation
  void insert(const LeftOperandType& leftOperand,
              const RightOperandType& rightOperand, const ResultType& result) {
    allocate();
    const auto key = hash(leftOperand, rightOperand);
    if (valid[key]) {
      ++stats.collisions;
    } else {
      stats.trackInsert();
      valid[key] = true;
    }
    table[key] = {leftOperand, rightOperand, result};
  }

  /// Look up a result in the compute table
  ///
  /// @param leftOperand The left operand
  /// @param rightOperand The right operand
  /// @return A pointer to the result if it is found, otherwise nullptr.
  ResultType* lookup(const LeftOperandType& leftOperand,
                     const RightOperandType& rightOperand) {
    ResultType* result = nullptr;
    ++stats.lookups;
    if (table.empty()) {
      return result;
    }
    const auto key = hash(leftOperand, rightOperand);
    if (!valid[key]) {
      return result;
    }

    auto& entry = table[key];
    if (entry.leftOperand != leftOperand) {
      return result;
    }
    if (entry.rightOperand != rightOperand) {
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

  /// Replace the bucket storage and discard cached results.
  ///
  /// The capacity must be a power of two. Invalidates lookup result pointers.
  /// Invalid capacity leaves the table unchanged.
  [[nodiscard]] llvm::Error resize(const size_t numBuckets) {
    auto replacement = create(numBuckets);
    if (!replacement) {
      return replacement.takeError();
    }
    table.swap(replacement->table);
    valid.swap(replacement->valid);
    stats.numBuckets = numBuckets;
    stats.reset();
    return llvm::Error::success();
  }

  /// Print the statistics of the compute table
  ///
  /// @param os The output stream to print to
  /// @return The output stream
  llvm::raw_ostream&
  printStatistics(llvm::raw_ostream& os = llvm::outs()) const {
    return os << stats;
  }

private:
  void allocate() {
    if (!table.empty()) {
      return;
    }
    std::vector<Entry> entries(stats.numBuckets);
    std::vector<bool> occupied(stats.numBuckets, false);
    table.swap(entries);
    valid.swap(occupied);
  }

  /// The actual table storing the entries
  std::vector<Entry> table;
  /// Dynamic bitset to mark valid entries
  std::vector<bool> valid;
  /// Statistics of the compute table
  TableStatistics stats{};
};
} // namespace dd
