/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "dd/Package.hpp"

#include <iostream>
#include <nlohmann/json_fwd.hpp>
#include <string>

namespace dd {

/**
 * @brief Computes an estimate for the memory usage of active DDs.
 * @details The estimate is based on the number of active entries in the
 * respective unique tables. It accounts for the memory used by DD nodes, DD
 * edges, and real numbers.
 * @param package The package instance
 * @return The estimated memory usage in MiB
 */
[[nodiscard]] double computeActiveMemoryMiB(const Package& package);

/**
 * @brief Computes an estimate for the peak memory usage of DDs.
 * @details The estimate is based on the peak number of used entries in the
 * respective memory managers. It accounts for the memory used by DD nodes, DD
 * edges, and real numbers.
 * @param package The package instance
 * @return The estimated memory usage in MiB
 */
[[nodiscard]] double computePeakMemoryMiB(const Package& package);

[[nodiscard]] nlohmann::basic_json<>
getStatistics(const Package& package, bool includeIndividualTables = false);

/**
 * @brief Get some key statistics about data structures used by the DD package
 * @return A JSON representation of the statistics
 */
[[nodiscard]] nlohmann::basic_json<> getDataStructureStatistics();

[[nodiscard]] std::string getStatisticsString(const Package& package);

void printStatistics(const Package& package, std::ostream& os = std::cout);

} // namespace dd
