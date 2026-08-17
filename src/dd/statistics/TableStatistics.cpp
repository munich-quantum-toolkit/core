/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/statistics/TableStatistics.hpp"

#include "StatisticsJson.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <string>

namespace dd {

void TableStatistics::trackInsert() noexcept {
  ++inserts;
  ++numEntries;
  peakNumEntries = std::max(peakNumEntries, numEntries);
}

void TableStatistics::reset() noexcept { numEntries = 0U; }

double TableStatistics::hitRatio() const noexcept {
  if (lookups == 0) {
    return 1.;
  }
  return static_cast<double>(hits) / static_cast<double>(lookups);
}

double TableStatistics::colRatio() const noexcept {
  if (lookups == 0) {
    return 0.;
  }
  return static_cast<double>(collisions) / static_cast<double>(lookups);
}

double TableStatistics::loadFactor() const noexcept {
  if (numBuckets == 0) {
    return 0.;
  }
  return static_cast<double>(numEntries) / static_cast<double>(numBuckets);
}

double TableStatistics::getEntrySizeMiB() const noexcept {
  return static_cast<double>(entrySize) / static_cast<double>(1ULL << 20U);
}

double TableStatistics::getMemoryMiB() const noexcept {
  return static_cast<double>(numBuckets) * getEntrySizeMiB();
}

std::string TableStatistics::toString() const { return toJson(*this).dump(2U); }

nlohmann::basic_json<> toJson(const TableStatistics& s) {
  if (s.lookups == 0) {
    return "unused";
  }

  nlohmann::basic_json<> j;
  j["num_buckets"] = s.numBuckets;
  j["memory_MiB"] = s.getMemoryMiB();
  j["num_entries"] = s.numEntries;
  j["peak_num_entries"] = s.peakNumEntries;
  j["collisions"] = s.collisions;
  j["hits"] = s.hits;
  j["lookups"] = s.lookups;
  j["inserts"] = s.inserts;
  j["hit_ratio"] = s.hitRatio();
  j["col_ratio"] = s.colRatio();
  j["load_factor"] = s.loadFactor();
  return j;
}

} // namespace dd
