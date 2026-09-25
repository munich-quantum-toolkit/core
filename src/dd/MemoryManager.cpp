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
#include <memory>

namespace dd {

MemoryManager::MemoryManager(size_t entrySize,
                             const std::size_t initialAllocationSize)
    : entrySize_(entrySize), chunks(1), stats(entrySize) {
  chunks[0] = {std::make_unique_for_overwrite<Storage>(initialAllocationSize *
                                                       entrySize),
               initialAllocationSize * entrySize};
  /// The first slab must exist before its bounds can be initialized.
  /// NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
  chunkIt = chunks[0].first.get();
  /// NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer,cppcoreguidelines-pro-bounds-pointer-arithmetic)
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

void MemoryManager::reset(const bool resizeToTotal) noexcept {
  available = nullptr;

  auto numAllocations = stats.numAllocations;
  chunks.resize(1U);
  if (resizeToTotal && chunks[0].second != stats.numAllocated * entrySize_) {
    chunks[0] = {std::make_unique_for_overwrite<Storage>(stats.numAllocated *
                                                         entrySize_),
                 stats.numAllocated * entrySize_};
    ++numAllocations;
  }

  chunkIt = chunks[0].first.get();
  /// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
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
  stats.trackReusedEntries();
  return entry;
}

void MemoryManager::allocateNewChunk() {
  assert(!entryAvailableInChunk());

  const auto numPrevEntries = chunks.back().second / entrySize_;
  const auto numNewEntries = numPrevEntries * GROWTH_FACTOR;

  chunks.emplace_back(
      std::make_unique_for_overwrite<Storage>(numNewEntries * entrySize_),
      numNewEntries * entrySize_);
  chunkIt = chunks.back().first.get();
  /// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  chunkEndIt = chunks.back().first.get() + chunks.back().second;
  ++stats.numAllocations;
  stats.numAllocated += numNewEntries;
}

LLBase* MemoryManager::getEntryFromChunk() noexcept {
  assert(!entryAvailableForReuse());
  assert(entryAvailableInChunk());

  auto* entry = chunkIt;
  /// NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  chunkIt += entrySize_;
  /// Fresh vector nodes must start with cleared collection flags.
  std::memset(entry, 0, entrySize_);
  stats.trackUsedEntries();
  return reinterpret_cast<LLBase*>(entry);
}

[[nodiscard]] bool MemoryManager::entryAvailableInChunk() const noexcept {
  return chunkIt != chunkEndIt;
}

} // namespace dd
