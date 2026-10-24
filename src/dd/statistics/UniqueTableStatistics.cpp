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

#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"

#include <string>

namespace dd {

void UniqueTableStatistics::reset() noexcept { TableStatistics::reset(); }

std::string UniqueTableStatistics::toString() const {
  return llvm::formatv("{0:2}", toJson(*this)).str();
}

llvm::json::Value toJson(const UniqueTableStatistics& s) {
  if (s.lookups == 0) {
    return "unused";
  }

  auto j = toJson(static_cast<const TableStatistics&>(s));
  (*j.getAsObject())["gc_runs"] = s.gcRuns;
  return j;
}
} // namespace dd
