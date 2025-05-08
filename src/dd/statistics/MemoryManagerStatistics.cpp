/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/statistics/MemoryManagerStatistics.hpp"

#include "dd/statistics/Statistics.hpp"

#include <algorithm>
#include <cstddef>
#include <nlohmann/json.hpp>

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
  return static_cast<double>(numUsed) / static_cast<double>(numAllocated);
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

nlohmann::basic_json<> MemoryManagerStatistics::json() const {
  if (peakNumUsed == 0) {
    return "unused";
  }

  auto j = Statistics::json();
  j["memory_allocated_MiB"] = getAllocatedMemoryMiB();
  j["memory_used_MiB"] = getUsedMemoryMiB();
  j["memory_used_MiB_peak"] = getPeakUsedMemoryMiB();
  j["num_allocated"] = numAllocated;
  j["num_allocations"] = numAllocations;
  j["num_available_for_reuse"] = numAvailableForReuse;
  j["num_available_for_reuse_peak"] = peakNumAvailableForReuse;
  j["num_available_from_chunks"] = getNumAvailableFromChunks();
  j["num_available_total"] = getTotalNumAvailable();
  j["num_used"] = numUsed;
  j["num_used_peak"] = peakNumUsed;
  j["usage_ratio"] = getUsageRatio();
  return j;
}

} // namespace dd
