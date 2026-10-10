/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/statistics/UniqueTableStatistics.hpp"

#include "dd/UniqueTable.hpp"
#include "dd/statistics/TableStatistics.hpp"

#include "StatisticsJson.hpp"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstddef>
#include <string>

namespace dd {

void UniqueTableStatistics::reset() noexcept { TableStatistics::reset(); }

std::string UniqueTableStatistics::toString() const {
  return toJson(*this).dump(2U);
}

nlohmann::basic_json<> toJson(const UniqueTableStatistics& s) {
  if (s.lookups == 0) {
    return "unused";
  }

  auto j = toJson(static_cast<const TableStatistics&>(s));
  j["gc_runs"] = s.gcRuns;
  return j;
}
nlohmann::basic_json<> toJson(const UniqueTable& table,
                              const bool includeIndividualTables) {
  const auto& stats = table.getStats();
  if (std::ranges::all_of(stats, [](const UniqueTableStatistics& stat) {
        return stat.peakNumEntries == 0U;
      })) {
    return "unused";
  }

  UniqueTableStatistics totalStats;
  for (const auto& stat : stats) {
    totalStats.entrySize = std::max(totalStats.entrySize, stat.entrySize);
    totalStats.numBuckets += stat.numBuckets;
    totalStats.numEntries += stat.numEntries;
    totalStats.peakNumEntries += stat.peakNumEntries;
    totalStats.collisions += stat.collisions;
    totalStats.hits += stat.hits;
    totalStats.lookups += stat.lookups;
    totalStats.inserts += stat.inserts;
    totalStats.gcRuns = std::max(totalStats.gcRuns, stat.gcRuns);
  }

  nlohmann::basic_json<> j;
  j["total"] = toJson(totalStats);
  if (includeIndividualTables) {
    std::size_t v = 0U;
    for (const auto& stat : stats) {
      j[std::to_string(v)] = toJson(stat);
      ++v;
    }
  }
  return j;
}
} // namespace dd
