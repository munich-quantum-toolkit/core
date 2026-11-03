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

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

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

std::string TableStatistics::toString() const {
  std::string result;
  llvm::raw_string_ostream out(result);
  llvm::json::OStream json(out, 2);
  writeJson(json, *this);
  return result;
}

void writeJson(llvm::json::OStream& json, const TableStatistics& s) {
  if (s.lookups == 0) {
    json.value("unused");
    return;
  }
  json.object([&] { writeTableStatistics(json, s); });
}

void writeTableStatistics(llvm::json::OStream& json, const TableStatistics& s) {
  json.attribute("num_buckets", s.numBuckets);
  writeStatistic(json, "memory_MiB", s.getMemoryMiB());
  json.attribute("num_entries", s.numEntries);
  json.attribute("peak_num_entries", s.peakNumEntries);
  json.attribute("collisions", s.collisions);
  json.attribute("hits", s.hits);
  json.attribute("lookups", s.lookups);
  json.attribute("inserts", s.inserts);
  writeStatistic(json, "hit_ratio", s.hitRatio());
  writeStatistic(json, "col_ratio", s.colRatio());
  writeStatistic(json, "load_factor", s.loadFactor());
}

} // namespace dd
