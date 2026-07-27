/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file DeviceRegistry.hpp
/// @brief Side-effect-free QDMI device discovery and registration.

#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace qdmi {

/// Parameters applied to one device session before it is initialized.
struct SessionParameters {
  /// Base URL of the device service.
  std::optional<std::string> baseUrl;
  /// Authentication token.
  std::optional<std::string> token;
  /// Authentication file path.
  std::optional<std::filesystem::path> authFile;
  /// Authentication service URL.
  std::optional<std::string> authUrl;
  /// Authentication username.
  std::optional<std::string> username;
  /// Authentication password.
  std::optional<std::string> password;
  /// First implementation-defined parameter.
  std::optional<std::string> custom1;
  /// Second implementation-defined parameter.
  std::optional<std::string> custom2;
  /// Third implementation-defined parameter.
  std::optional<std::string> custom3;
  /// Fourth implementation-defined parameter.
  std::optional<std::string> custom4;
  /// Fifth implementation-defined parameter.
  std::optional<std::string> custom5;
};

/// A side-effect-free registration for one QDMI device.
struct DeviceDefinition {
  /// Stable device identifier.
  std::string id;
  /// Path to the QDMI device library.
  std::filesystem::path library;
  /// Symbol prefix used by the QDMI device library.
  std::string prefix;
  /// Default session parameters.
  SessionParameters session;
};

/// Discovers and merges QDMI device definitions without loading libraries.
class DeviceRegistry {
public:
  /// Discovers definitions from the standard configuration sources.
  DeviceRegistry();

  /// Creates a registry from explicit definitions without configuration
  /// discovery.
  explicit DeviceRegistry(std::vector<DeviceDefinition> definitions);

  /// Returns enabled definitions in stable registration order.
  [[nodiscard]] const std::vector<DeviceDefinition>& definitions() const {
    return definitions_;
  }

  /// Registers a complete definition, optionally replacing an existing or
  /// explicitly disabled ID.
  void registerDevice(DeviceDefinition definition, bool replace = false);

  /// Registers a definition unless its ID already exists or is disabled.
  [[nodiscard]] bool registerDeviceIfAbsent(DeviceDefinition definition);

private:
  std::vector<DeviceDefinition> definitions_;
  std::unordered_set<std::string> disabledIds_;
};

} // namespace qdmi
