/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file StatisticsJson.hpp
 * @brief Internal JSON rendering for decision-diagram statistics.
 * @details This header is not installed. It keeps nlohmann_json out of the
 * public headers while the statistics reports stay JSON-formatted.
 */

#pragma once

#include <nlohmann/json.hpp>

namespace dd {

class UniqueTable;
struct MemoryManagerStatistics;
struct TableStatistics;
struct UniqueTableStatistics;

/// Render the memory-manager statistics, or "unused" if nothing was used.
[[nodiscard]] nlohmann::basic_json<> toJson(const MemoryManagerStatistics& s);

/// Render the table statistics, or "unused" if the table was never queried.
[[nodiscard]] nlohmann::basic_json<> toJson(const TableStatistics& s);

/// Render the unique-table statistics, or "unused" if it was never queried.
[[nodiscard]] nlohmann::basic_json<> toJson(const UniqueTableStatistics& s);

/**
 * @brief Render the statistics of every table in a unique table.
 * @param table The unique table
 * @param includeIndividualTables Whether to add an entry per variable
 * @return The rendered statistics, or "unused" if no table holds entries
 */
[[nodiscard]] nlohmann::basic_json<> toJson(const UniqueTable& table,
                                            bool includeIndividualTables);

} // namespace dd
