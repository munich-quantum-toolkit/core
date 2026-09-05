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

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace qdmi::detail {

/// Rejects IDs that the QDMI string-property ABI cannot represent.
void validateDeviceId(std::string_view id);

/// Stages one low-precedence package manifest before the driver is frozen.
auto stagePackageManifest(const std::filesystem::path& path) -> int;

/// Freezes and returns the staged package manifests.
[[nodiscard]] auto freezePackageManifests()
    -> std::vector<std::filesystem::path>;

/// Reopens package-manifest staging after driver construction fails.
void rollbackPackageManifestFreeze();

/// Parses one strict JSON object with the manifest session grammar.
auto parseDeviceSessionJson(const char* data, size_t size,
                            DeviceSessionConfig& config) -> int;

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
