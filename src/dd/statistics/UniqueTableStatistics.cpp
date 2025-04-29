/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/statistics/UniqueTableStatistics.hpp"

#include "dd/statistics/TableStatistics.hpp"

#include <algorithm>
#include <nlohmann/json.hpp>

namespace dd {

void UniqueTableStatistics::trackActiveEntry() noexcept {
  ++numActiveEntries;
  peakNumActiveEntries = std::max(peakNumActiveEntries, numActiveEntries);
}

void UniqueTableStatistics::reset() noexcept {
  TableStatistics::reset();
  numActiveEntries = 0U;
}

nlohmann::basic_json<> UniqueTableStatistics::json() const {
  if (lookups == 0) {
    return "unused";
  }

  auto j = TableStatistics::json();
  j["num_active_entries"] = numActiveEntries;
  j["peak_num_active_entries"] = peakNumActiveEntries;
  j["gc_runs"] = gcRuns;
  return j;
}
} // namespace dd
