/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file RealNumberUniqueTable.hpp
/// Unique table for canonical decision-diagram real numbers.

#pragma once

#include "dd/DDDefinitions.hpp"
#include "dd/MemoryManager.hpp"
#include "dd/statistics/UniqueTableStatistics.hpp"

#include <cstddef>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace dd {

struct RealNumber;

/// Canonical real numbers with stable addresses and absolute-tolerance lookup.
///
/// Binary cells partition the search space without rounding stored values.
/// Hashed buckets avoid concentrating small values and values above one in a
/// single sorted list. Lookup returns the nearest entry within tolerance,
/// preferring the smaller magnitude on a tie. Zero, one, and sqrt(1/2) have
/// priority. The global tolerance must be finite and non-negative.
class RealNumberUniqueTable {
  /// Initial power-of-two bucket count.
  static constexpr size_t NBUCKET = 65536U;
  static constexpr size_t MAX_BUCKETS = 1048576U;

  /// The initial garbage collection limit.
  ///
  /// The initial garbage collection limit is the number of entries that
  /// must be present in the table before garbage collection is triggered.
  /// Increasing this number reduces the number of garbage collections, but
  /// increases the memory usage.
  static constexpr std::size_t INITIAL_GC_LIMIT = 65536U;

public:
  /// The default constructor
  /// @param manager The memory manager to use for allocating new numbers.
  /// @param initialGCLim The initial garbage collection limit.
  explicit RealNumberUniqueTable(MemoryManager& manager,
                                 std::size_t initialGCLim = INITIAL_GC_LIMIT);

  /// Maps a non-negative value to its bucket in the current index.
  /// The bucket depends on the table capacity and indexed tolerance.
  [[nodiscard]] size_t hash(fp val) const noexcept;

  /// Ordinary absolute-tolerance bucket heads; growth and tolerance changes
  /// invalidate bucket iterators and reorder chains. Entry addresses survive
  /// until collection or reset.
  [[nodiscard]] const auto& getTable() const noexcept { return table; }

  /// Get combined entry, lookup, and bucket statistics for both indexes.
  /// Collision counts describe the ordinary absolute-tolerance index.
  [[nodiscard]] const auto& getStats() const noexcept { return stats; }

  /// Lookup a number in the table
  ///
  /// This function is used to lookup and insert them into the table if
  /// they are not yet present. Since the table only ever stores non-negative
  /// numbers, the lookup is a three-step process. First the sign is stripped
  /// off the number and stored, then the non-negative value is looked up in the
  /// table and an aligned pointer to the respective entry is returned. Finally,
  /// If the sign of the original number was negative, the pointer is adjusted.
  /// @param val The floating point number to look up.
  /// @return A pointer to an entry corresponding to that number.
  [[nodiscard]] RealNumber* lookup(fp val);

  /// Preserve a matrix root's range while retaining nonzero constant priority.
  /// Other values are interned exactly in a separate index so they cannot
  /// become representatives in the ordinary absolute-tolerance table.
  [[nodiscard]] RealNumber* lookupRoot(fp val);

  /// Check whether the table possibly needs garbage collection.
  /// @returns Whether the number of entries in the table has reached the
  /// garbage collection limit.
  [[nodiscard]] bool possiblyNeedsCollection() const noexcept;

  /// Perform garbage collection.
  ///
  /// This function performs garbage collection. It first checks whether
  /// garbage collection is necessary. If not, it does nothing. Otherwise, it
  /// iterates over all entries in the table and removes all numbers whose
  /// pointers are unmarked. If the force flag is set, garbage collection is
  /// performed even if it is not strictly necessary.
  /// Based on how many entries are returned to the available list, the garbage
  /// collection limit is dynamically adjusted.
  /// @param force Whether to force garbage collection.
  /// @returns The number of entries returned to the available list.
  std::size_t garbageCollect(bool force = false) noexcept;

  /// Clear the table.
  ///
  /// Discards bucket heads and resets counters and the collection limit.
  /// Entry storage remains owned by the memory manager.
  void clear() noexcept;

  /// Print the table.
  void print() const;

  /// Print the bucket distribution of the table.
  /// @param os The output stream to print to.
  /// @returns The output stream.
  std::ostream& printBucketDistribution(std::ostream& os = std::cout);

  /// Count the marked entries in the table
  [[nodiscard]] std::size_t countMarkedEntries() const noexcept;

private:
  /// Typedef for a bucket in the table.
  using Bucket = RealNumber*;
  /// Typedef for the table.
  using Table = std::vector<Bucket>;

  /// Intrusive bucket chains; rehashing preserves entry addresses and flags.
  Table table = Table(NBUCKET);
  /// Matrix root weights share the memory manager and mark/sweep ownership.
  /// Exact keys keep lookup independent of tolerance and avoid tiny-value
  /// chains.
  std::unordered_map<fp, RealNumber*> exactRoots;

  /// A power-of-two cell width between eight and sixteen times the tolerance
  /// keeps each tolerance interval within the central and adjacent cells.
  int cellExponent = -1074;
  fp indexedTolerance = -1.;

  void rehash(size_t size, int exponent);
  void updateTolerance();

  /// A pointer to the memory manager for the numbers stored in the table.
  MemoryManager* memoryManager{};

  /// A collection of statistics
  UniqueTableStatistics stats{};

  /// The initial garbage collection limit
  std::size_t initialGCLimit;
  /// The current garbage collection limit
  std::size_t gcLimit = initialGCLimit;

  /// Lookup a non-negative number in the table.
  ///
  /// The table only ever stores non-negative values. Thus, any lookup
  /// must be split between actually looking up the number and adjusting for its
  /// sign. This function looks up a number in the table. If the number is not
  /// found, a new number is created and inserted into the table.
  /// @param val The floating point number to look up. Must be non-negative.
  /// @returns An aligned pointer to the entry corresponding to the number.
  [[nodiscard]] RealNumber* lookupNonNegative(fp val);
};
} // namespace dd
