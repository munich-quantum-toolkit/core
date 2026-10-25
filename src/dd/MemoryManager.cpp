/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/MemoryManager.hpp"

#include "dd/LinkedListBase.hpp"

#include <cassert>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>

#if defined(_MSC_VER) && !defined(__clang__)
#ifdef _M_X64
#include <xmmintrin.h>
#elif defined(_M_ARM64)
#include <intrin.h>
#endif
#endif

namespace dd {

MemoryManager::MemoryManager(size_t entrySize,
                             const std::size_t initialAllocationSize)
    : entrySize_(entrySize), chunks(1), stats(entrySize) {
  if (initialAllocationSize == 0) {
    throw std::invalid_argument("Initial DD allocation size must be positive.");
  }
  if (initialAllocationSize > std::numeric_limits<size_t>::max() / entrySize) {
    throw std::bad_alloc();
  }
  const auto bytes = initialAllocationSize * entrySize;
  chunks[0] = {std::make_unique_for_overwrite<Storage>(bytes), bytes};
  // The first slab must exist before its bounds can be initialized.
  // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
  chunkIt = chunks[0].first.get();
  // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer,cppcoreguidelines-pro-bounds-pointer-arithmetic)
  chunkEndIt = chunks[0].first.get() + chunks[0].second;
  stats.numAllocations = 1U;
  stats.numAllocated = initialAllocationSize;
}

LLBase* MemoryManager::get() {
  if (entryAvailableForReuse()) {
    return getEntryFromAvailableList();
  }

  if (!entryAvailableInChunk()) {
    allocateNewChunk();
  }

  return getEntryFromChunk();
}

void MemoryManager::returnEntry(LLBase& entry) noexcept {
  entry.setNext(available);
  available = &entry;
  stats.trackReturnedEntry();
}

void MemoryManager::reset() {
  const auto numAllocations = stats.numAllocations;
  chunks.resize(1U);

  available = nullptr;
  chunkIt = chunks[0].first.get();
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  chunkEndIt = chunks[0].first.get() + chunks[0].second;

  stats.reset();
  stats.numAllocations = numAllocations;
  stats.numAllocated = chunks[0].second / entrySize_;
}

bool MemoryManager::entryAvailableForReuse() const noexcept {
  return available != nullptr;
}

LLBase* MemoryManager::getEntryFromAvailableList() noexcept {
  assert(entryAvailableForReuse());

  auto* entry = available;
  available = available->next();
  if (available != nullptr) {
#if defined(__clang__) || defined(__GNUC__)
    __builtin_prefetch(available, 0, 3);
#elif defined(_M_ARM64)
    __prefetch(available);
#elif defined(_M_X64)
    _mm_prefetch(reinterpret_cast<char*>(available), _MM_HINT_T0);
#endif
  }
  stats.trackReusedEntries();
  return entry;
}

void MemoryManager::allocateNewChunk() {
  assert(!entryAvailableInChunk());

  const auto previousBytes = chunks.back().second;
  if (previousBytes > std::numeric_limits<size_t>::max() / GROWTH_FACTOR) {
    throw std::bad_alloc();
  }
  const auto bytes = previousBytes * GROWTH_FACTOR;
  const auto numNewEntries = bytes / entrySize_;

  chunks.emplace_back(std::make_unique_for_overwrite<Storage>(bytes), bytes);
  chunkIt = chunks.back().first.get();
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  chunkEndIt = chunks.back().first.get() + chunks.back().second;
  ++stats.numAllocations;
  stats.numAllocated += numNewEntries;
}

LLBase* MemoryManager::getEntryFromChunk() noexcept {
  assert(!entryAvailableForReuse());
  assert(entryAvailableInChunk());

  auto* entry = chunkIt;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  chunkIt += entrySize_;
  // Fresh vector nodes must start with cleared collection flags.
  std::memset(entry, 0, entrySize_);
  stats.trackUsedEntries();
  return reinterpret_cast<LLBase*>(entry);
}

[[nodiscard]] bool MemoryManager::entryAvailableInChunk() const noexcept {
  return chunkIt != chunkEndIt;
}

} // namespace dd
