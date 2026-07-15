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

#include <nlohmann/json.hpp>

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
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

/// A parsed, side-effect-free registration for one QDMI device.
struct DeviceDefinition {
  /// Stable device identifier.
  std::string id;
  /// Path to the QDMI device library.
  std::filesystem::path library;
  /// Supported ABI marker. Only `qdmi-v1` is accepted.
  std::string abi = "qdmi-v1";
  /// Symbol prefix used by the QDMI v1.3 device library.
  std::string prefix;
  /// Default session parameters.
  SessionParameters session;
  /// Configuration source that declared the definition.
  std::filesystem::path source;
  /// Whether this definition is enabled.
  bool enabled = true;
};

/// Controls configuration discovery.
struct ConfigOptions {
  /// Root used to discover packaged manifest fragments.
  std::optional<std::filesystem::path> configRoot;
  /// Explicit configuration file replacing system, user, and project files.
  std::optional<std::filesystem::path> explicitFile;
  /// Base directory for relative paths in inline configuration.
  std::optional<std::filesystem::path> baseDirectory;
  /// Inline JSON configuration layered above discovered files.
  std::optional<nlohmann::json> inlineOverrides;
  /// Runtime definitions layered at the highest precedence.
  std::vector<DeviceDefinition> runtimeOverrides;
  /// Whether discovery excludes packaged built-in manifests.
  bool isolated = false;
};

/// Discovers and merges QDMI device definitions without loading libraries.
class DeviceRegistry {
public:
  explicit DeviceRegistry(const ConfigOptions& options = {});

  /// Returns enabled definitions in stable ID order.
  [[nodiscard]] const std::vector<DeviceDefinition>& definitions() const {
    return definitions_;
  }

  /// Registers a complete definition.
  void registerDevice(DeviceDefinition definition, bool replace = false);

  /// Removes a definition, returning whether it existed.
  bool unregisterDevice(std::string_view id);

private:
  std::vector<DeviceDefinition> definitions_;
};

} // namespace qdmi
