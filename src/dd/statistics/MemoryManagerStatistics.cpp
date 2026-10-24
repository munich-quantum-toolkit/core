/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/statistics/MemoryManagerStatistics.hpp"

#include "StatisticsJson.hpp"

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <string>

namespace dd {

double MemoryManagerStatistics::entryMemoryMIB() const {
  return static_cast<double>(entrySize_) / (1ULL << 20U);
}

std::size_t
MemoryManagerStatistics::getNumAvailableFromChunks() const noexcept {
  return getTotalNumAvailable() - numAvailableForReuse;
}

std::size_t MemoryManagerStatistics::getTotalNumAvailable() const noexcept {
  return numAllocated - numUsed;
}

double MemoryManagerStatistics::getUsageRatio() const noexcept {
  return numAllocated == 0
             ? 0.
             : static_cast<double>(numUsed) / static_cast<double>(numAllocated);
}

double MemoryManagerStatistics::getAllocatedMemoryMiB() const noexcept {
  return static_cast<double>(numAllocated) * entryMemoryMIB();
}

double MemoryManagerStatistics::getUsedMemoryMiB() const noexcept {
  return static_cast<double>(numUsed) * entryMemoryMIB();
}

double MemoryManagerStatistics::getPeakUsedMemoryMiB() const noexcept {
  return static_cast<double>(peakNumUsed) * entryMemoryMIB();
}

void MemoryManagerStatistics::trackUsedEntries(
    const std::size_t numEntries) noexcept {
  numUsed += numEntries;
  peakNumUsed = std::max(peakNumUsed, numUsed);
}

void MemoryManagerStatistics::trackReusedEntries(
    const std::size_t numEntries) noexcept {
  numUsed += numEntries;
  peakNumUsed = std::max(peakNumUsed, numUsed);
  numAvailableForReuse -= numEntries;
}

void MemoryManagerStatistics::trackReturnedEntry() noexcept {
  ++numAvailableForReuse;
  peakNumAvailableForReuse =
      std::max(peakNumAvailableForReuse, numAvailableForReuse);
  --numUsed;
}
void MemoryManagerStatistics::reset() noexcept {
  numAllocations = 0U;
  numAllocated = 0U;
  numUsed = 0U;
  numAvailableForReuse = 0U;
}

std::string MemoryManagerStatistics::toString() const {
  std::string result;
  llvm::raw_string_ostream out(result);
  llvm::json::OStream json(out, 2);
  writeJson(json, *this);
  return result;
}

void writeJson(llvm::json::OStream& json, const MemoryManagerStatistics& s) {
  if (s.peakNumUsed == 0) {
    json.value("unused");
    return;
  }

  json.object([&] {
    writeStatistic(json, "memory_allocated_MiB", s.getAllocatedMemoryMiB());
    writeStatistic(json, "memory_used_MiB", s.getUsedMemoryMiB());
    writeStatistic(json, "memory_used_MiB_peak", s.getPeakUsedMemoryMiB());
    json.attribute("num_allocated", s.numAllocated);
    json.attribute("num_allocations", s.numAllocations);
    json.attribute("num_available_for_reuse", s.numAvailableForReuse);
    json.attribute("num_available_for_reuse_peak", s.peakNumAvailableForReuse);
    json.attribute("num_available_from_chunks", s.getNumAvailableFromChunks());
    json.attribute("num_available_total", s.getTotalNumAvailable());
    json.attribute("num_used", s.numUsed);
    json.attribute("num_used_peak", s.peakNumUsed);
    writeStatistic(json, "usage_ratio", s.getUsageRatio());
  });
}

} // namespace dd
