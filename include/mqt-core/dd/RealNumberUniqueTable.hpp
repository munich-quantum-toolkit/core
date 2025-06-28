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

#include "dd/DDDefinitions.hpp"
#include "dd/MemoryManager.hpp"
#include "dd/statistics/UniqueTableStatistics.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>

namespace dd {

struct RealNumber;

/**
 * @brief Immortal numbers that will never be garbage collected.
 */
namespace immortals {
constexpr std::array<fp, 1> get() { return {.5}; }
constexpr std::size_t size() { return get().size(); }
} // namespace immortals

/**
 * @brief A unique table for real numbers.
 * @details A hash table that stores real numbers. The hash table is implemented
 * as an array of buckets, each of which is a linked list of entries. The hash
 * table has a fixed number of buckets.
 * @note: The implementation assumes that all values are non-negative and in the
 * range [0, 1]. While numbers outside of this range can be stored, they will
 * always be placed in the same bucket and will therefore cause collisions.
 */
class RealNumberUniqueTable {
  /**
   * @brief The number of buckets in the table.
   * @details The number of buckets is fixed and cannot be changed after the
   * table has been created. Increasing the number of buckets reduces the
   * number of collisions, but increases the memory usage.
   * @attention The number of buckets has to be one larger than a power of two.
   * Otherwise, the hash function will not work correctly.
   */
  static constexpr std::size_t NBUCKET = 65537U;

  /**
   * @brief The initial garbage collection limit.
   * @details The initial garbage collection limit is the number of entries that
   * must be present in the table before garbage collection is triggered.
   * Increasing this number reduces the number of garbage collections, but
   * increases the memory usage.
   */
  static constexpr std::size_t INITIAL_GC_LIMIT = 65536U;

public:
  /**
   * @brief The default constructor
   * @param manager The memory manager to use for allocating new numbers.
   * @param initialGCLim The initial garbage collection limit.
   */
  explicit RealNumberUniqueTable(MemoryManager& manager,
                                 std::size_t initialGCLim = INITIAL_GC_LIMIT);

  /**
   * @brief The hash function for the hash table.
   * @details The hash function for the table is a simple linear (clipped) hash
   * function. The hash function is used to map floating point numbers to the
   * buckets of the table.
   * @param val The floating point number to hash. Must be non-negative.
   * @returns The hash value of the floating point number.
   * @note Typically, you would expect the hash to be an unsigned integer. Here,
   * we use a signed integer because it turns out to result in fewer assembly
   * instructions. See https://godbolt.org/z/9v4TEMfdz for a comparison.
   */
  static std::int64_t hash(fp val) noexcept;

  /// Get a reference to the table.
  [[nodiscard]] const auto& getTable() const noexcept { return table; }

  /// Get a reference to the statistics
  [[nodiscard]] const auto& getStats() const noexcept { return stats; }

  /**
   * @brief Lookup a number in the table
   * @details This function is used to lookup and insert them into the table if
   * they are not yet present. Since the table only ever stores non-negative
   * numbers, the lookup is a three-step process. First the sign is stripped off
   * the number and stored, then the non-negative value is looked up in the
   * table and an aligned pointer to the respective entry is returned. Finally,
   * If the sign of the original number was negative, the pointer is adjusted.
   * @param val The floating point number to look up.
   * @return A pointer to an entry corresponding to that number.
   */
  [[nodiscard]] RealNumber* lookup(fp val);

  /**
   * @brief Check whether the table possibly needs garbage collection.
   * @returns Whether the number of entries in the table has reached the garbage
   * collection limit.
   */
  [[nodiscard]] bool possiblyNeedsCollection() const noexcept;

  /**
   * @brief Perform garbage collection.
   * @details This function performs garbage collection. It first checks whether
   * garbage collection is necessary. If not, it does nothing. Otherwise, it
   * iterates over all entries in the table and removes all numbers whose
   * pointers are unmarked. If the force flag is set, garbage collection is
   * performed even if it is not strictly necessary.
   * Based on how many entries are returned to the available list, the garbage
   * collection limit is dynamically adjusted.
   * @param force Whether to force garbage collection.
   * @returns The number of entries returned to the available list.
   */
  std::size_t garbageCollect(bool force = false) noexcept;

  /**
   * @brief Clear the table.
   * @details This function clears the table. It iterates over all entries in
   * the table and sets them to nullptr. It also discards the available list and
   * all but the first chunk of the allocated chunks. Also resets all counters.
   */
  void clear() noexcept;

  /**
   * @brief Print the table.
   */
  void print() const;

  /**
   * @brief Print the bucket distribution of the table.
   * @param os The output stream to print to.
   * @returns The output stream.
   */
  std::ostream& printBucketDistribution(std::ostream& os = std::cout);

  /// Count the marked entries in the table
  [[nodiscard]] std::size_t countMarkedEntries() const noexcept;

private:
  /// Typedef for a bucket in the table.
  using Bucket = RealNumber*;
  /// Typedef for the table.
  using Table = std::array<Bucket, NBUCKET>;

  /**
   * @brief The actual hash table
   * @details The hash table is an array of buckets. Each bucket is a linked
   * list of entries. The linked list is implemented by using the next pointer
   * of the entries.
   */
  Table table{};
  /**
   * @brief The tail table
   * @details The tail table is an array of pointers to the last entry in each
   * bucket. This is used to speed up the insertion of new entries.
   */
  std::array<RealNumber*, NBUCKET> tailTable{};

  /// A pointer to the memory manager for the numbers stored in the table.
  MemoryManager* memoryManager{};

  /// A collection of statistics
  UniqueTableStatistics stats{};

  /// The initial garbage collection limit
  std::size_t initialGCLimit;
  /// The current garbage collection limit
  std::size_t gcLimit = initialGCLimit;

  /**
   * @brief Finds or inserts a value into the bucket indexed by key.
   * @details This function either finds an entry with a value within TOLERANCE
   * of val in the bucket indexed by key or inserts a new entry with value val
   * into the bucket.
   * @param key The index of the bucket to find or insert the value into.
   * @param val The value to find or insert.
   * @returns A pointer to the found or inserted entry.
   */
  RealNumber* findOrInsert(std::int64_t key, fp val);

  /**
   * @brief Inserts a value in the front of the bucket indexed by key.
   * @param key The index of the bucket to insert the value into.
   * @param val The value to insert.
   * @return A pointer to the inserted entry.
   */
  RealNumber* insertFront(std::int64_t key, fp val);

  /**
   * @brief Inserts a value in the back of the bucket indexed by key.
   * @param key The index of the bucket to insert the value into.
   * @param val The value to insert.
   * @return A pointer to the inserted entry.
   */
  RealNumber* insertBack(std::int64_t key, fp val);

  /**
   * @brief Lookup a non-negative number in the table.
   * @details The table only ever stores non-negative values. Thus, any lookup
   * must be split between actually looking up the number and adjusting for its
   * sign. This function looks up a number in the table. If the number is not
   * found, a new number is created and inserted into the table.
   * @param val The floating point number to look up. Must be non-negative.
   * @returns An aligned pointer to the entry corresponding to the number.
   */
  [[nodiscard]] RealNumber* lookupNonNegative(fp val);
};
} // namespace dd
