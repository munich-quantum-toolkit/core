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
 * @brief Side-effect-free QDMI discovery and lazy device management.
 */

#pragma once

#include "qdmi/Device.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace qdmi {

/** Parameters applied to one device session before it is initialized. */
struct SessionParameters {
  std::optional<std::string> baseUrl;
  std::optional<std::string> token;
  std::optional<std::filesystem::path> authFile;
  std::optional<std::string> authUrl;
  std::optional<std::string> username;
  std::optional<std::string> password;
  std::optional<std::string> projectId;
  std::optional<std::string> custom1;
  std::optional<std::string> custom2;
  std::optional<std::string> custom3;
  std::optional<std::string> custom4;
  std::optional<std::string> custom5;
};

/** A parsed, side-effect-free registration for one QDMI device. */
struct DeviceDefinition {
  std::string id;
  std::filesystem::path library;
  std::string abi = "qdmi-v1";
  std::string prefix;
  bool enabled = true;
  SessionParameters session;
  std::filesystem::path source;
};

/** Controls configuration discovery. */
struct ConfigOptions {
  std::optional<std::filesystem::path> configRoot;
  std::optional<std::filesystem::path> explicitFile;
  std::optional<std::filesystem::path> baseDirectory;
  bool isolated = false;
  std::optional<nlohmann::json> inlineOverrides;
  std::vector<DeviceDefinition> runtimeOverrides;
};

/** Discovers and merges QDMI device definitions without loading libraries. */
class DeviceRegistry {
public:
  explicit DeviceRegistry(const ConfigOptions& options = {});

  /** Returns enabled definitions in stable ID order. */
  [[nodiscard]] const std::vector<DeviceDefinition>& definitions() const {
    return definitions_;
  }

  /** Registers a complete definition. */
  void registerDevice(DeviceDefinition definition, bool replace = false);

  /** Removes a definition, returning whether it existed. */
  bool unregisterDevice(std::string_view id);

private:
  std::vector<DeviceDefinition> definitions_;
};

/** Result of independently opening every enabled definition. */
struct OpenAllResult {
  std::map<std::string, Device> devices;
  std::map<std::string, std::string> errors;
};

/** Lazily opens configured QDMI devices and shares loaded v1 libraries. */
class DeviceManager {
public:
  explicit DeviceManager(const ConfigOptions& options = {});
  explicit DeviceManager(DeviceRegistry registry);
  ~DeviceManager();

  DeviceManager(DeviceManager&&) noexcept;
  DeviceManager& operator=(DeviceManager&&) noexcept;
  DeviceManager(const DeviceManager&) = delete;
  DeviceManager& operator=(const DeviceManager&) = delete;

  [[nodiscard]] const std::vector<DeviceDefinition>& definitions() const;
  void registerDevice(DeviceDefinition definition, bool replace = false);
  bool unregisterDevice(std::string_view id);
  [[nodiscard]] Device
  open(std::string_view id,
       const SessionParameters& sessionOverrides = SessionParameters{});
  [[nodiscard]] OpenAllResult
  openAll(const SessionParameters& sessionOverrides = SessionParameters{});

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace qdmi
