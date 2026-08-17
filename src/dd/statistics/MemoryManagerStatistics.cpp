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

#include <nlohmann/json.hpp>

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

std::string MemoryManagerStatistics::toString() const {
  return toJson(*this).dump(2U);
}

nlohmann::basic_json<> toJson(const MemoryManagerStatistics& s) {
  if (s.peakNumUsed == 0) {
    return "unused";
  }

  nlohmann::basic_json<> j;
  j["memory_allocated_MiB"] = s.getAllocatedMemoryMiB();
  j["memory_used_MiB"] = s.getUsedMemoryMiB();
  j["memory_used_MiB_peak"] = s.getPeakUsedMemoryMiB();
  j["num_allocated"] = s.numAllocated;
  j["num_allocations"] = s.numAllocations;
  j["num_available_for_reuse"] = s.numAvailableForReuse;
  j["num_available_for_reuse_peak"] = s.peakNumAvailableForReuse;
  j["num_available_from_chunks"] = s.getNumAvailableFromChunks();
  j["num_available_total"] = s.getTotalNumAvailable();
  j["num_used"] = s.numUsed;
  j["num_used_peak"] = s.peakNumUsed;
  j["usage_ratio"] = s.getUsageRatio();
  return j;
}

} // namespace dd
