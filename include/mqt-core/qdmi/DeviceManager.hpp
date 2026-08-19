/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file DeviceManager.hpp
 * @brief Immutable QDMI registry snapshots and fresh device sessions.
 */

#pragma once

#include "qdmi/Device.hpp"
#include "qdmi/DeviceRegistry.hpp"

#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace qdmi {

/// Successes and independent failures from opening every registered device.
struct OpenAllResult {
  std::map<std::string, Device> devices;
  std::map<std::string, std::string> errors;
};

/**
 * @brief An immutable snapshot that opens fresh QDMI device sessions.
 *
 * A manager is not a singleton. Returned devices and their derived objects own
 * the session and library state they need and may outlive this manager.
 */
class DeviceManager {
public:
  /// Snapshot the process default registry.
  DeviceManager();

  /// Snapshot an explicit registry by value.
  explicit DeviceManager(DeviceRegistry registry);

  [[nodiscard]] const std::vector<DeviceDefinition>& definitions() const {
    return registry_.definitions();
  }

  [[nodiscard]] std::vector<std::string> deviceIds() const {
    return registry_.deviceIds();
  }

  [[nodiscard]] Device
  open(std::string_view id,
       const DeviceSessionConfig& sessionOverrides = {}) const;

  [[nodiscard]] OpenAllResult
  openAll(const DeviceSessionConfig& sessionOverrides = {}) const;

private:
  DeviceRegistry registry_;
};

/// Open one fresh session from the current process default registry.
[[nodiscard]] Device
openDevice(std::string_view id,
           const DeviceSessionConfig& sessionOverrides = {});

} // namespace qdmi
