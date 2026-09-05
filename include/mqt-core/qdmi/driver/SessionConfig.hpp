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
 * @brief Construction helpers for QDMI device session configuration.
 */

#pragma once

#include "qdmi/driver/Driver.hpp"

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace qdmi {

/**
 * @brief Construct a device session configuration from individual parameters.
 * @throws std::invalid_argument If both an inline device configuration and a
 * device configuration file are set.
 */
[[nodiscard]] inline auto makeDeviceSessionConfig(
    std::optional<std::string> baseUrl, std::optional<std::string> token,
    std::optional<std::filesystem::path> authFile,
    std::optional<std::string> authUrl, std::optional<std::string> username,
    std::optional<std::string> password,
    std::optional<std::string> deviceConfig,
    std::optional<std::filesystem::path> deviceConfigFile,
    std::optional<std::string> custom1, std::optional<std::string> custom2,
    std::optional<std::string> custom3, std::optional<std::string> custom4,
    std::optional<std::string> custom5) -> DeviceSessionConfig {
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
  return {
      .baseUrl = std::move(baseUrl),
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
      .custom5 = std::move(custom5),
  };
}

} // namespace qdmi
