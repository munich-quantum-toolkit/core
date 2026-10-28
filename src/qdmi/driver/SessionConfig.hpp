/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file SessionConfig.hpp
/// Merges QDMI device session configuration overrides.

#pragma once

#include "Driver.hpp"

#include <optional>

namespace qdmi::detail {
template <class T>
inline void applyOverride(std::optional<T>& value,
                          const std::optional<T>& overrideValue) {
  if (overrideValue) {
    value = overrideValue;
  }
}

/// Apply present overrides, including explicitly empty values.
[[nodiscard]] inline auto
mergeSessionConfig(DeviceSessionConfig merged,
                   const DeviceSessionConfig& overrides)
    -> DeviceSessionConfig {
  applyOverride(merged.baseUrl, overrides.baseUrl);
  applyOverride(merged.token, overrides.token);
  applyOverride(merged.authFile, overrides.authFile);
  applyOverride(merged.authUrl, overrides.authUrl);
  applyOverride(merged.username, overrides.username);
  applyOverride(merged.password, overrides.password);
  applyOverride(merged.deviceConfiguration, overrides.deviceConfiguration);
  applyOverride(merged.custom1, overrides.custom1);
  applyOverride(merged.custom2, overrides.custom2);
  applyOverride(merged.custom3, overrides.custom3);
  applyOverride(merged.custom4, overrides.custom4);
  applyOverride(merged.custom5, overrides.custom5);
  return merged;
}
} // namespace qdmi::detail
