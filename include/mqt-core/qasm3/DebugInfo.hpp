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

struct DebugInfo {
  size_t line;
  size_t column;
  std::string filename;
  std::shared_ptr<DebugInfo> parent;

  DebugInfo(const size_t l, const size_t c, std::string file,
            std::shared_ptr<DebugInfo> parentDebugInfo = nullptr)
      : line(l), column(c), filename(std::move(std::move(file))),
        parent(std::move(parentDebugInfo)) {}

  [[nodiscard]] std::string toString() const {
    return filename + ":" + std::to_string(line) + ":" + std::to_string(column);
  }
};

} // namespace qasm3
