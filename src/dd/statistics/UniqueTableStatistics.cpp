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

#include "dd/statistics/TableStatistics.hpp"

#include "StatisticsJson.hpp"

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <optional>
#include <string>

namespace dd {

void UniqueTableStatistics::reset() noexcept { TableStatistics::reset(); }

std::string UniqueTableStatistics::toString() const {
  std::string result;
  llvm::raw_string_ostream out(result);
  llvm::json::OStream json(out, 2);
  writeJson(json, *this);
  return result;
}

void writeJson(llvm::json::OStream& json, const UniqueTableStatistics& s,
               const std::optional<size_t> activeEntries) {
  if (s.lookups == 0) {
    json.value("unused");
    return;
  }
  json.object([&] {
    writeTableStatistics(json, s);
    json.attribute("gc_runs", s.gcRuns);
    if (activeEntries) {
      json.attribute("num_active_entries", *activeEntries);
    }
  });
}
} // namespace dd
