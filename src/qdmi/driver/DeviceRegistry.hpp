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

#include "mlir/Support/LogicalResult.h"

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace qdmi::detail {

/// Rejects IDs that the QDMI string-property ABI cannot represent.
mlir::LogicalResult validateDeviceId(std::string_view id);

/// Stages one low-precedence device manifest before the driver is frozen.
auto stageDeviceManifest(const std::filesystem::path& path) -> int;

/// Freezes and returns the staged device manifests.
[[nodiscard]] auto freezeDeviceManifests()
    -> std::vector<std::filesystem::path>;

/// Reopens device-manifest staging after driver construction fails.
void rollbackDeviceManifestFreeze();

/// Parses one strict JSON object with the manifest session grammar.
auto parseDeviceSessionJson(const char* data, size_t size,
                            DeviceSessionConfig& config) -> int;

/// Discovers configured QDMI devices without loading their libraries.
class DeviceRegistry {
public:
  [[nodiscard]] static mlir::FailureOr<DeviceRegistry> discover();

  [[nodiscard]] const std::vector<qdmi::DeviceDefinition>& definitions() const {
    return definitions_;
  }

  [[nodiscard]] const std::vector<std::string>& disabledIds() const {
    return disabledIds_;
  }

private:
  DeviceRegistry() = default;
  std::vector<qdmi::DeviceDefinition> definitions_;
  std::vector<std::string> disabledIds_;
};

} // namespace qdmi::detail
