/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file StatisticsJson.hpp
/// Internal JSON rendering for decision-diagram statistics.
///
/// The private JSON representation is not part of the public statistics API.

#pragma once

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <optional>

namespace dd {

class UniqueTable;
struct MemoryManagerStatistics;
struct TableStatistics;
struct UniqueTableStatistics;

/// Write a finite statistic without depending on the numeric locale.
inline void writeStatistic(llvm::json::OStream& json, llvm::StringRef name,
                           const double value) {
  json.attributeBegin(name);
  json.rawValue([&](llvm::raw_ostream& out) { out << llvm::APFloat(value); });
  json.attributeEnd();
}

/// Write the memory-manager statistics, or "unused" if nothing was used.
void writeJson(llvm::json::OStream& json, const MemoryManagerStatistics& s);

/// Write the fields shared by table and unique-table statistics.
void writeTableStatistics(llvm::json::OStream& json, const TableStatistics& s);

/// Write the table statistics, or "unused" if the table was never queried.
void writeJson(llvm::json::OStream& json, const TableStatistics& s);

/// Write unique-table statistics, including active entries for package reports.
void writeJson(llvm::json::OStream& json, const UniqueTableStatistics& s,
               std::optional<size_t> activeEntries = std::nullopt);

/// Write the statistics of every table in a unique table.
///
/// @param json The output stream
/// @param table The unique table
/// @param includeIndividualTables Whether to add an entry per variable
/// @param activeEntries The number of active entries in the table
void writeJson(llvm::json::OStream& json, const UniqueTable& table,
               bool includeIndividualTables, size_t activeEntries);

} // namespace dd
