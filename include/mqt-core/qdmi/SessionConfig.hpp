/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file SessionConfig.hpp
 * @brief QDMI device-session configuration types and helpers.
 */

#pragma once

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace qdmi {

/// Inline JSON used to configure one QDMI device session.
struct InlineDeviceConfiguration {
  std::string json;
};

/// JSON file used to configure one QDMI device session.
struct FileDeviceConfiguration {
  std::filesystem::path path;
};

/// One replaceable source for a runtime device description.
using DeviceConfigurationSource =
    std::variant<InlineDeviceConfiguration, FileDeviceConfiguration>;

/// Optional parameters applied before one QDMI device session is initialized.
struct DeviceSessionConfig {
  std::optional<std::string> baseUrl;
  std::optional<std::string> token;
  std::optional<std::filesystem::path> authFile;
  std::optional<std::string> authUrl;
  std::optional<std::string> username;
  std::optional<std::string> password;
  std::optional<DeviceConfigurationSource> deviceConfiguration;
  std::optional<std::string> custom1;
  std::optional<std::string> custom2;
  std::optional<std::string> custom3;
  std::optional<std::string> custom4;
  std::optional<std::string> custom5;
};

/**
 * @brief Construct session configuration from the public keyword-style fields.
 * @throws std::invalid_argument If both configuration sources are set.
 */
[[nodiscard]] inline DeviceSessionConfig makeDeviceSessionConfig(
    std::optional<std::string> baseUrl, std::optional<std::string> token,
    std::optional<std::filesystem::path> authFile,
    std::optional<std::string> authUrl, std::optional<std::string> username,
    std::optional<std::string> password,
    std::optional<std::string> deviceConfig,
    std::optional<std::filesystem::path> deviceConfigFile,
    std::optional<std::string> custom1, std::optional<std::string> custom2,
    std::optional<std::string> custom3, std::optional<std::string> custom4,
    std::optional<std::string> custom5) {
  if (deviceConfig && deviceConfigFile) {
    throw std::invalid_argument(
        "device_config and device_config_file are mutually exclusive");
  }
  std::optional<DeviceConfigurationSource> configuration;
  if (deviceConfig) {
    configuration = InlineDeviceConfiguration{.json = std::move(*deviceConfig)};
  } else if (deviceConfigFile) {
    configuration =
        FileDeviceConfiguration{.path = std::move(*deviceConfigFile)};
  }
  return {.baseUrl = std::move(baseUrl),
          .token = std::move(token),
          .authFile = std::move(authFile),
          .authUrl = std::move(authUrl),
          .username = std::move(username),
          .password = std::move(password),
          .deviceConfiguration = std::move(configuration),
          .custom1 = std::move(custom1),
          .custom2 = std::move(custom2),
          .custom3 = std::move(custom3),
          .custom4 = std::move(custom4),
          .custom5 = std::move(custom5)};
}

} // namespace qdmi
