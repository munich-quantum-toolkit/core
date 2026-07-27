/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file DeviceManager.hpp
/// @brief Lazy opening and lifetime management for configured QDMI devices.

#pragma once

#include "qdmi/Device.hpp"
#include "qdmi/DeviceRegistry.hpp"

#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace qdmi {

/// Result of independently opening every enabled definition.
struct OpenAllResult {
  /// Successfully opened devices keyed by stable device ID.
  std::map<std::string, Device> devices;
  /// Error messages for definitions that could not be opened, keyed by ID.
  std::map<std::string, std::string> errors;
};

/// Lazily opens configured QDMI devices.
///
/// Each call to @ref open creates an independent device session. Returned
/// devices and objects derived from them own the library and session state they
/// require and may outlive the manager.
class DeviceManager {
public:
  DeviceManager();
  explicit DeviceManager(DeviceRegistry registry);

  /// Returns the definitions owned by this manager.
  [[nodiscard]] const std::vector<DeviceDefinition>& definitions() const {
    return registry_.definitions();
  }

  /// Opens one configured device by stable ID.
  [[nodiscard]] Device
  open(std::string_view id,
       const SessionParameters& sessionOverrides = SessionParameters{}) const;

  /// Opens a snapshot of all definitions and isolates failures by ID.
  [[nodiscard]] OpenAllResult openAll(
      const SessionParameters& sessionOverrides = SessionParameters{}) const;

private:
  DeviceRegistry registry_;
};

} // namespace qdmi
