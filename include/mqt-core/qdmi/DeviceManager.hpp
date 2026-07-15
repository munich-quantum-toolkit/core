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
#include <memory>
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

/// Lazily opens configured QDMI devices and shares loaded libraries.
class DeviceManager {
public:
  explicit DeviceManager(const ConfigOptions& options = {});
  explicit DeviceManager(DeviceRegistry registry);
  ~DeviceManager();

  DeviceManager(DeviceManager&&) noexcept;
  DeviceManager& operator=(DeviceManager&&) noexcept;
  DeviceManager(const DeviceManager&) = delete;
  DeviceManager& operator=(const DeviceManager&) = delete;

  /// Returns a thread-safe snapshot of registered definitions.
  [[nodiscard]] std::vector<DeviceDefinition> definitions() const;

  /// Registers a definition. Concurrent manager operations remain safe.
  void registerDevice(DeviceDefinition definition, bool replace = false);

  /// Unregisters a definition without invalidating opened devices.
  bool unregisterDevice(std::string_view id);

  /// Opens one configured device by stable ID.
  [[nodiscard]] Device
  open(std::string_view id,
       const SessionParameters& sessionOverrides = SessionParameters{});

  /// Opens a snapshot of all definitions and isolates failures by ID.
  [[nodiscard]] OpenAllResult
  openAll(const SessionParameters& sessionOverrides = SessionParameters{});

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace qdmi
