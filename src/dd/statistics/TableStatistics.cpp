/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/statistics/TableStatistics.hpp"

#include "dd/statistics/Statistics.hpp"

#include <algorithm>
#include <nlohmann/json.hpp>

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

nlohmann::basic_json<> TableStatistics::json() const {
  if (lookups == 0) {
    return "unused";
  }

  auto j = Statistics::json();
  j["num_buckets"] = numBuckets;
  j["memory_MiB"] = getMemoryMiB();
  j["num_entries"] = numEntries;
  j["peak_num_entries"] = peakNumEntries;
  j["collisions"] = collisions;
  j["hits"] = hits;
  j["lookups"] = lookups;
  j["inserts"] = inserts;
  j["hit_ratio"] = hitRatio();
  j["col_ratio"] = colRatio();
  j["load_factor"] = loadFactor();
  return j;
}

} // namespace dd
