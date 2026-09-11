/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file MemoryManager.hpp
/// Memory allocation and reuse for decision-diagram nodes.

#pragma once

#include "dd/statistics/MemoryManagerStatistics.hpp"

#include <cassert>
#include <concepts>
#include <cstddef>
#include <vector>

namespace dd {

// forward declarations
struct LLBase;

/// Allocate and reuse objects of one type derived from `LLBase`.
///
/// Objects are stored in contiguous chunks whose capacity grows geometrically.
/// Returned objects enter a free list and are reused before allocating from a
/// chunk, reducing allocation overhead for DD nodes and edge weights.
class MemoryManager {
  MemoryManager(size_t entrySize, std::size_t initialAllocationSize);

public:
  // delete copy construction and assignment
  MemoryManager(const MemoryManager&) = delete;
  MemoryManager& operator=(const MemoryManager&) = delete;

  /// Initial chunk capacity. Larger chunks trade memory for fewer allocations.
  static constexpr std::size_t INITIAL_ALLOCATION_SIZE = 2048U;

  /// Capacity multiplier when allocating the next chunk.
  static constexpr double GROWTH_FACTOR = 2U;

  /// Construct a new MemoryManager object for objects of type T.
  /// @param initialAllocationSize The initial number of entries to allocate
  /// @tparam T The type of the entries
  template <class T>
  static MemoryManager
  create(const std::size_t initialAllocationSize = INITIAL_ALLOCATION_SIZE) {
    return {sizeof(T), initialAllocationSize};
  }

  ~MemoryManager() = default;

  /// Get an entry from the manager.
  ///
  /// If an entry is available for reuse, it is returned. Otherwise, an
  /// entry from the pre-allocated chunks is returned. If no entry is available,
  /// a new chunk is allocated.
  /// @tparam T The type of the entry.
  /// @return A pointer to an entry.
  template <class T>
    requires std::derived_from<T, LLBase>
  [[nodiscard]] T* get() {
    assert(sizeof(T) == entrySize_ && "Cannot get entry of different size");

    return static_cast<T*>(get());
  }

  /// Return an entry to the manager.
  ///
  /// The entry is added to the list of available entries. The entry
  /// must not be used after it has been returned to the manager.
  /// @param entry A reference to an entry that is no longer in use.
  void returnEntry(LLBase& entry) noexcept;

  /// Reset the manager.
  ///
  /// Drops all but the first chunk. If `resizeToTotal` is set to true,
  /// the first chunk is resized to the total number of entries. This increases
  /// memory locality and reduces the number of allocations when the manager is
  /// used again. However, it might also require a huge contiguous block of
  /// memory to be allocated.
  /// @param resizeToTotal If set to true, the first chunk is resized to the
  /// total number of entries.
  void reset(bool resizeToTotal = false) noexcept;

  /// Get a reference to the statistics
  [[nodiscard]] const auto& getStats() const noexcept { return stats; }

private:
  /// Get an entry from the manager
  [[nodiscard]] LLBase* get();

  /// Check whether an entry is available for reuse
  /// @return true if an entry is available for reuse, false otherwise
  [[nodiscard]] bool entryAvailableForReuse() const noexcept;

  /// Get an entry from the list of available entries
  /// @return A pointer to an entry ready for reuse
  [[nodiscard]] LLBase* getEntryFromAvailableList() noexcept;

  /// Check whether an entry is available in the current chunk
  /// @return true if an entry is available in the current chunk, false
  /// otherwise
  [[nodiscard]] bool entryAvailableInChunk() const noexcept;

  /// Allocate a new chunk of memory
  void allocateNewChunk();

  /// Get an entry from the current chunk
  /// @return A pointer to an entry from the current chunk
  [[nodiscard]] LLBase* getEntryFromChunk() noexcept;

  /// The size of an entry in bytes (as reported by `sizeof`)
  size_t entrySize_;

  /// A chunk of memory as a vector of bytes
  using Chunk = std::vector<std::byte>;

  /// A linked list of entries that are available for (re-)use
  ///
  /// The MemoryManager maintains a linked list of entries that are
  /// available for (re-)use. This list is implemented as a singly linked list
  /// using the `next()` method of the entries. The `available` member points to
  /// the first entry in the list. If the list is empty, `available` is
  /// `nullptr`.
  LLBase* available;

  /// The storage for the entries
  ///
  /// The MemoryManager maintains a vector of chunks. Each chunk is a
  /// vector of entries. Entries in a chunk are allocated contiguously.
  std::vector<Chunk> chunks;

  /// Iterator to the next available entry in the current chunk
  ///
  /// This iterator points to the next available entry in the current
  /// chunk. If the current chunk is full, it points to the end of the chunk.
  Chunk::iterator chunkIt;

  /// Iterator to the end of the current chunk
  ///
  /// This iterator points to the end of the current chunk. It is used
  /// to determine whether the current chunk is full.
  Chunk::iterator chunkEndIt;

  /// Memory manager statistics
  MemoryManagerStatistics stats;
};

} // namespace dd
