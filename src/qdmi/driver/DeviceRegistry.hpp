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

#include "qdmi/driver/Driver.hpp"

#include <string>
#include <vector>

namespace qdmi::detail {

/// Discovers configured QDMI devices without loading their libraries.
class DeviceRegistry {
public:
  DeviceRegistry();

  [[nodiscard]] const std::vector<qdmi::DeviceDefinition>& definitions() const {
    return definitions_;
  }

  [[nodiscard]] const std::vector<std::string>& disabledIds() const {
    return disabledIds_;
  }

private:
  std::vector<qdmi::DeviceDefinition> definitions_;
  std::vector<std::string> disabledIds_;
};

} // namespace qdmi::detail
