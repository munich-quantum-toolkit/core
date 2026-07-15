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

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

namespace qasm3 {

/// Source location information for error reporting and diagnostics.
struct DebugInfo {
  size_t line;                       ///< 1-based line number.
  size_t column;                     ///< 1-based column number.
  std::string filename;              ///< Source file name.
  std::shared_ptr<DebugInfo> parent; ///< Enclosing location.

  DebugInfo(const size_t l, const size_t c, std::string file,
            std::shared_ptr<DebugInfo> parentDebugInfo = nullptr)
      : line(l), column(c), filename(std::move(file)),
        parent(std::move(parentDebugInfo)) {}

  /// Returns a human-readable `"file:line:col"` string.
  [[nodiscard]] std::string toString() const {
    return filename + ":" + std::to_string(line) + ":" + std::to_string(column);
  }
};

} // namespace qasm3
