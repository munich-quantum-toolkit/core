/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/**
 * @file ComputeTable.hpp
 * @brief Data structure for caching computed results of binary operations
 */

#pragma once

#include "dd/Node.hpp"
#include "dd/statistics/TableStatistics.hpp"
#include "ir/Definitions.hpp"

#include <cstddef>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace dd {

/**
 * @brief Data structure for caching computed results of binary operations
 * @tparam LeftOperandType type of the operation's left operand
 * @tparam RightOperandType type of the operation's right operand
 * @tparam ResultType type of the operation's result
 */
template <class LeftOperandType, class RightOperandType, class ResultType>
class ComputeTable {
public:
  /**
   * Default constructor
   * @param numBuckets Number of hash table buckets. Must be a power of two.
   */
  explicit ComputeTable(const size_t numBuckets = 16384U) {
    // numBuckets must be a power of two
    if ((numBuckets & (numBuckets - 1)) != 0) {
      throw std::invalid_argument("Number of buckets must be a power of two.");
    }
    stats.entrySize = sizeof(Entry);
    stats.numBuckets = numBuckets;
    valid = std::vector(numBuckets, false);
    table = std::vector<Entry>(numBuckets);
  }

  /**
   * @brief An entry in the compute table
   * @details A triple consisting of the left operand, the right operand, and
   * the result of a binary operation.
   */
  struct Entry {
    LeftOperandType leftOperand;
    RightOperandType rightOperand;
    ResultType result;
  };

  /**
   * @brief Compute the hash value for a given pair of operands
   * @param leftOperand The left operand
   * @param rightOperand The right operand
   * @return The hash value
   */
  [[nodiscard]] std::size_t hash(const LeftOperandType& leftOperand,
                                 const RightOperandType& rightOperand) const {
    auto h1 = std::hash<LeftOperandType>{}(leftOperand);
    if constexpr (std::is_same_v<LeftOperandType, dNode*>) {
      if (!dNode::isTerminal(leftOperand)) {
        h1 = qc::combineHash(
            h1, dd::dNode::getDensityMatrixTempFlags(leftOperand->flags));
      }
    }
    auto h2 = std::hash<RightOperandType>{}(rightOperand);
    if constexpr (std::is_same_v<RightOperandType, dNode*>) {
      if (!dNode::isTerminal(rightOperand)) {
        h2 = qc::combineHash(
            h2, dd::dNode::getDensityMatrixTempFlags(rightOperand->flags));
      }
    }
    const auto hash = qc::combineHash(h1, h2);
    const auto mask = stats.numBuckets - 1;
    return hash & mask;
  }

  /// Get a reference to the underlying table
  [[nodiscard]] const auto& getTable() const { return table; }

  /// Get a reference to the statistics
  [[nodiscard]] const auto& getStats() const noexcept { return stats; }

  /**
   * @brief Insert a new entry into the compute table
   * @details Any existing entry for the resulting hash value will be replaced.
   * @param leftOperand The left operand
   * @param rightOperand The right operand
   * @param result The result of the operation
   */
  void insert(const LeftOperandType& leftOperand,
              const RightOperandType& rightOperand, const ResultType& result) {
    const auto key = hash(leftOperand, rightOperand);
    if (valid[key]) {
      ++stats.collisions;
    } else {
      stats.trackInsert();
      valid[key] = true;
    }
    table[key] = {leftOperand, rightOperand, result};
  }

  /**
   * @brief Look up a result in the compute table
   * @param leftOperand The left operand
   * @param rightOperand The right operand
   * @param useDensityMatrix Whether a density matrix is expected
   * @return A pointer to the result if it is found, otherwise nullptr.
   */
  ResultType* lookup(const LeftOperandType& leftOperand,
                     const RightOperandType& rightOperand,
                     [[maybe_unused]] const bool useDensityMatrix = false) {
    ResultType* result = nullptr;
    ++stats.lookups;
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

    if constexpr (std::is_same_v<RightOperandType, dNode*> ||
                  std::is_same_v<RightOperandType, dCachedEdge>) {
      // Since density matrices are reduced representations of matrices, a
      // density matrix may not be returned when a matrix is required and vice
      // versa
      if (!dNode::isTerminal(entry.result.p) &&
          dNode::isDensityMatrixNode(entry.result.p->flags) !=
              useDensityMatrix) {
        return result;
      }
    }
    ++stats.hits;
    return &entry.result;
  }

  /**
   * @brief Clear the compute table
   * @details Sets all entries to invalid.
   */
  void clear() { valid = std::vector(stats.numBuckets, false); }

  /**
   * @brief Print the statistics of the compute table
   * @param os The output stream to print to
   * @return The output stream
   */
  std::ostream& printStatistics(std::ostream& os = std::cout) const {
    return os << stats;
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
