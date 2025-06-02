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

#include "dd/statistics/TableStatistics.hpp"

#include <cstddef>
#include <nlohmann/json_fwd.hpp>

namespace dd {
/// \brief A class for storing statistics of a unique table
struct UniqueTableStatistics : public TableStatistics {
  /**
   * @brief The total number of active entries.
   */
  std::size_t numActiveEntries = 0U;
  /// The peak number of active entries in the table
  std::size_t peakNumActiveEntries = 0U;
  /// The number of garbage collection runs
  std::size_t gcRuns = 0U;

  /// Track a new active entry
  void trackActiveEntry() noexcept;

  /// Reset all statistics (except for the peak values)
  void reset() noexcept override;

  /// Get a JSON representation of the statistics
  [[nodiscard]] nlohmann::json json() const override;
};

} // namespace dd
