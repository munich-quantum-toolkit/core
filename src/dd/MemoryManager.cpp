/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
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

namespace dd {

MemoryManager::MemoryManager(size_t entrySize,
                             const std::size_t initialAllocationSize)
    : entrySize_(entrySize), available(nullptr),
      chunks(1, Chunk(initialAllocationSize * entrySize)),
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

void MemoryManager::returnEntry(LLBase& entry) noexcept {
  entry.setNext(available);
  available = &entry;
  stats.trackReturnedEntry();
}

void MemoryManager::reset(const bool resizeToTotal) noexcept {
  available = nullptr;

  auto numAllocations = stats.numAllocations;
  chunks.resize(1U);
  if (resizeToTotal) {
    chunks[0].resize(stats.numAllocated * entrySize_);
    ++numAllocations;
  }

  chunkIt = chunks[0].begin();
  chunkEndIt = chunks[0].end();

  stats.reset();
  stats.numAllocations = numAllocations;
  stats.numAllocated = chunks[0].size() / entrySize_;
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

  const auto numPrevEntries = chunks.back().size() / entrySize_;
  const auto numNewEntries = static_cast<std::size_t>(
      static_cast<double>(numPrevEntries) * GROWTH_FACTOR);

  chunks.emplace_back(numNewEntries * entrySize_);
  chunkIt = chunks.back().begin();
  chunkEndIt = chunks.back().end();
  ++stats.numAllocations;
  stats.numAllocated += numNewEntries;
}

LLBase* MemoryManager::getEntryFromChunk() noexcept {
  assert(!entryAvailableForReuse());
  assert(entryAvailableInChunk());

  auto* entry = &(*chunkIt);
  chunkIt += static_cast<Chunk::difference_type>(entrySize_);
  stats.trackUsedEntries();
  return reinterpret_cast<LLBase*>(entry);
}

[[nodiscard]] bool MemoryManager::entryAvailableInChunk() const noexcept {
  return chunkIt != chunkEndIt;
}

} // namespace dd
