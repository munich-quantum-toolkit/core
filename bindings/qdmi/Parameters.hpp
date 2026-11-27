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

#include "qdmi/common/Common.hpp"

#include "nanobind/nanobind.h"

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <variant>

namespace mqt::bindings {
using PythonCustomJobParameter =
    std::variant<std::string, bool, int, double, nanobind::bytes>;

[[nodiscard]] inline std::optional<qdmi::CustomJobParameter>
toCustomJobParameter(const std::optional<PythonCustomJobParameter>& parameter) {
  if (!parameter) {
    return std::nullopt;
  }
  return std::visit(
      [](const auto& value) -> qdmi::CustomJobParameter {
        if constexpr (std::is_same_v<std::decay_t<decltype(value)>,
                                     nanobind::bytes>) {
          return std::as_bytes(std::span(value.c_str(), value.size()));
        } else {
          return value;
        }
      },
      *parameter);
}

/// Collect Python job keywords; byte values borrow the Python objects.
[[nodiscard]] inline qdmi::JobParameters
toJobParameters(const std::optional<size_t> shots,
                const std::optional<PythonCustomJobParameter>& custom1,
                const std::optional<PythonCustomJobParameter>& custom2,
                const std::optional<PythonCustomJobParameter>& custom3,
                const std::optional<PythonCustomJobParameter>& custom4,
                const std::optional<PythonCustomJobParameter>& custom5) {
  return {
      .shots = shots,
      .custom =
          {
              toCustomJobParameter(custom1),
              toCustomJobParameter(custom2),
              toCustomJobParameter(custom3),
              toCustomJobParameter(custom4),
              toCustomJobParameter(custom5),
          },
  };
}

} // namespace mqt::bindings
