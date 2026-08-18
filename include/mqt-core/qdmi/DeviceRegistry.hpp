/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file DeviceRegistry.hpp
 * @brief Side-effect-free QDMI device discovery and registration.
 */

#pragma once

#include "qdmi/SessionConfig.hpp"

#include <filesystem>
#include <string>
#include <unordered_set>
#include <vector>

namespace qdmi {

/// Stable metadata used to open fresh sessions for one QDMI device.
struct DeviceDefinition {
  std::string id;
  std::filesystem::path library;
  std::string prefix;
  DeviceSessionConfig session;
};

/// Discovers and combines device definitions without loading native code.
class DeviceRegistry {
public:
  /// Discover definitions from the standard configuration sources.
  DeviceRegistry();

  /// Create an isolated registry from explicit definitions.
  explicit DeviceRegistry(std::vector<DeviceDefinition> definitions);

  /// Return enabled definitions in stable registration order.
  [[nodiscard]] const std::vector<DeviceDefinition>& definitions() const {
    return definitions_;
  }

  /// Return enabled stable IDs without loading native code.
  [[nodiscard]] std::vector<std::string> deviceIds() const;

  /// Register or replace a complete definition.
  void registerDevice(DeviceDefinition definition, bool replace = false);

  /// Register a fallback unless the ID exists or is explicitly disabled.
  [[nodiscard]] bool registerDeviceIfAbsent(DeviceDefinition definition);

private:
  std::vector<DeviceDefinition> definitions_;
  std::unordered_set<std::string> disabledIds_;
};

/// Register a definition in the process default registry.
void registerDevice(DeviceDefinition definition, bool replace = false);

/// Register a default fallback unless its ID exists or is disabled.
[[nodiscard]] bool registerDeviceIfAbsent(DeviceDefinition definition);

/// Return IDs from the process default registry without loading native code.
[[nodiscard]] std::vector<std::string> registeredDeviceIds();

} // namespace qdmi
