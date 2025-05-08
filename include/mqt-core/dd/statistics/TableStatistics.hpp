/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "dd/statistics/Statistics.hpp"

#include <cstddef>
#include <nlohmann/json.hpp>

namespace dd {

/// A utility class for storing statistics of a table
struct TableStatistics : public Statistics {
  /// The size of a single entry
  std::size_t entrySize = 0U;
  /// The number of buckets in the table
  std::size_t numBuckets = 0U;
  /// The number of entries in the table
  std::size_t numEntries = 0U;
  /// The peak number of entries in the table
  std::size_t peakNumEntries = 0U;

  /// The number of collisions
  std::size_t collisions = 0U;
  /// The number of successful lookups
  std::size_t hits = 0U;
  /// The number of lookups
  std::size_t lookups = 0U;
  /// The number of inserts
  std::size_t inserts = 0U;

  /// Track a new insert
  void trackInsert() noexcept;

  /// Reset all statistics (except for peak values)
  void reset() noexcept override;

  /**
   * @brief Get the hit ratio of the table.
   * @details The hit ratio is the ratio of lookups that were successful.
   * @returns The hit ratio of the table.
   */
  [[nodiscard]] double hitRatio() const noexcept;

  /**
   * @brief Get the collision ratio of the table.
   * @details A collision occurs when the hash function maps two different
   * entries to the same bucket. The collision ratio is the ratio of lookups
   * that resulted in a collision.
   * @returns The collision ratio of the table.
   */
  [[nodiscard]] double colRatio() const noexcept;

  /**
   * @brief Get the load factor of the table.
   * @details The load factor is the ratio of entries to buckets.
   * @return The load factor of the table.
   */
  [[nodiscard]] double loadFactor() const noexcept;

  /// Convert the entry size to MiB
  [[nodiscard]] double getEntrySizeMiB() const noexcept;

  /// Get the amount of memory required for the table in MiB
  [[nodiscard]] double getMemoryMiB() const noexcept;

  /// Get a JSON representation of the statistics
  [[nodiscard]] nlohmann::json json() const override;
};

} // namespace dd
