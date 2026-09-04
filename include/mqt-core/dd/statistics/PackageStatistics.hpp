/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "dd/Package.hpp"

#include <iostream>
#include <string>

namespace dd {

/**
 * @brief Computes an estimate for the memory usage of active DDs.
 * @details The estimate is based on the number of active entries which are
 * computed by temporarily marking all nodes reachable from the current root
 * set and subsequently counting them in the unique tables. It accounts for the
 * memory used by DD nodes, DD edges, and real numbers.
 * @param package The package instance
 * @return The estimated memory usage in MiB
 */
[[nodiscard]] double computeActiveMemoryMiB(Package& package);

/**
 * @brief Computes an estimate for the peak memory usage of DDs.
 * @details The estimate is based on the peak number of used entries in the
 * respective memory managers. It accounts for the memory used by DD nodes, DD
 * edges, and real numbers.
 * @param package The package instance
 * @return The estimated memory usage in MiB
 */
[[nodiscard]] double computePeakMemoryMiB(const Package& package);

/**
 * @brief Get key statistics about the data structures used by the DD package.
 * @return A JSON-formatted string representation of the statistics
 */
[[nodiscard]] std::string getDataStructureStatisticsString();

/**
 * @brief Get key statistics about the data structures held by @p package.
 * @param package The package instance
 * @param includeIndividualTables Whether to report every unique table
 * @return A JSON-formatted string representation of the statistics
 */
[[nodiscard]] std::string
getStatisticsString(Package& package, bool includeIndividualTables = false);

void printStatistics(Package& package, std::ostream& os = std::cout);

} // namespace dd
