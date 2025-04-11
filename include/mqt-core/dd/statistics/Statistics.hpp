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

#include <nlohmann/json.hpp>
#include <ostream>
#include <string>

namespace dd {

struct Statistics {
  Statistics() = default;
  Statistics(const Statistics&) = default;
  Statistics(Statistics&&) = default;
  Statistics& operator=(const Statistics&) = default;
  Statistics& operator=(Statistics&&) = default;
  virtual ~Statistics() = default;

  /// Reset all statistics (except for peak values)
  virtual void reset() noexcept {};

  /// Get a JSON representation of the statistics
  [[nodiscard]] virtual nlohmann::json json() const;

  /// Get a pretty-printed string representation of the statistics
  [[nodiscard]] virtual std::string toString() const;

  /**
   * @brief Write a string representation to an output stream
   * @param os The output stream
   * @param stats The statistics
   * @return The output stream
   */
  friend std::ostream& operator<<(std::ostream& os, const Statistics& stats) {
    return os << stats.toString();
  }
};

} // namespace dd
