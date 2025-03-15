/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/MemoryManager.hpp"

#include <cassert>
#include <cstddef>

namespace dd {

MemoryManager::MemoryManager(unsigned entrySize,
                             const std::size_t initialAllocationSize)
    : entrySize(entrySize), available(nullptr),
      chunks(1, chunk_t(initialAllocationSize * entrySize)),
      chunkIt(chunks[0].begin()), chunkEndIt(chunks[0].end()),
      stats(entrySize) {
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

void MemoryManager::returnEntry(LLBase* entry) noexcept {
  assert(entry != nullptr);

  entry->setNext(available);
  available = entry;
  stats.trackReturnedEntry();
}

void MemoryManager::reset(const bool resizeToTotal) noexcept {
  available = nullptr;

  auto numAllocations = stats.numAllocations;
  chunks.resize(1U);
  if (resizeToTotal) {
    chunks[0].resize(stats.numAllocated * entrySize);
    ++numAllocations;
  }

  chunkIt = chunks[0].begin();
  chunkEndIt = chunks[0].end();

  stats.reset();
  stats.numAllocations = numAllocations;
  stats.numAllocated = chunks[0].size() / entrySize;
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

  const auto numPrevEntries = chunks.back().size() / entrySize;
  const auto numNewEntries = static_cast<std::size_t>(
      static_cast<double>(numPrevEntries) * GROWTH_FACTOR);

  chunks.emplace_back(numNewEntries * entrySize);
  chunkIt = chunks.back().begin();
  chunkEndIt = chunks.back().end();
  ++stats.numAllocations;
  stats.numAllocated += numNewEntries;
}

LLBase* MemoryManager::getEntryFromChunk() noexcept {
  assert(!entryAvailableForReuse());
  assert(entryAvailableInChunk());

  auto* entry = &(*chunkIt);
  chunkIt += entrySize;
  stats.trackUsedEntries();
  return reinterpret_cast<LLBase*>(entry);
}

[[nodiscard]] bool MemoryManager::entryAvailableInChunk() const noexcept {
  return chunkIt != chunkEndIt;
}

} // namespace dd
