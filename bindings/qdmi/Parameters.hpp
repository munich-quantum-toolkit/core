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

#include "qdmi/QDMI.hpp"
#include "qdmi/common/Common.hpp"

#include "nanobind/nanobind.h"

#include <cstddef>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace mqt::bindings {
using PythonCustomParameter =
    std::variant<std::string, bool, int, double, nanobind::bytes>;

[[nodiscard]] inline std::optional<qdmi::CustomParameter>
toCustomParameter(const std::optional<PythonCustomParameter>& parameter) {
  if (!parameter) {
    return std::nullopt;
  }
  return std::visit(
      [](const auto& value) -> qdmi::CustomParameter {
        if constexpr (std::is_same_v<std::decay_t<decltype(value)>,
                                     nanobind::bytes>) {
          const auto bytes =
              std::as_bytes(std::span(value.c_str(), value.size()));
          return std::vector<std::byte>(bytes.begin(), bytes.end());
        } else {
          return value;
        }
      },
      *parameter);
}

inline qdmi::SessionConfig makeDriverSessionConfig(
    std::optional<std::filesystem::path> driverPath,
    std::optional<std::string> token,
    std::optional<std::filesystem::path> authFile,
    std::optional<std::string> authUrl, std::optional<std::string> username,
    std::optional<std::string> password, std::optional<std::string> projectId,
    const std::optional<PythonCustomParameter>& custom1,
    const std::optional<PythonCustomParameter>& custom2,
    const std::optional<PythonCustomParameter>& custom3,
    const std::optional<PythonCustomParameter>& custom4,
    const std::optional<PythonCustomParameter>& custom5) {
  return {
      .driverPath = std::move(driverPath),
      .token = std::move(token),
      .authFile = std::move(authFile),
      .authUrl = std::move(authUrl),
      .username = std::move(username),
      .password = std::move(password),
      .projectId = std::move(projectId),
      .custom1 = toCustomParameter(custom1),
      .custom2 = toCustomParameter(custom2),
      .custom3 = toCustomParameter(custom3),
      .custom4 = toCustomParameter(custom4),
      .custom5 = toCustomParameter(custom5),
  };
}

} // namespace mqt::bindings
