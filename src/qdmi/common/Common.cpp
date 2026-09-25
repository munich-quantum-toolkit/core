/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file Common.cpp
/// Common definitions and utilities for working with QDMI in C++.

#include "qdmi/common/Common.hpp"

#include "support/Diagnostics.hpp"

#include "qdmi/constants.h"

#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>

#ifdef _WIN32
#include <filesystem>
#include <windows.h>
#endif
#include "mlir/Support/LogicalResult.h"

#include <utility>

namespace qdmi {
namespace detail {
auto environment(const std::string_view name) -> std::optional<std::string> {
#ifdef _WIN32
  std::wstring wideName;
  for (const char character : name) {
    wideName.push_back(
        static_cast<wchar_t>(static_cast<unsigned char>(character)));
  }
  const auto size = GetEnvironmentVariableW(wideName.c_str(), nullptr, 0);
  if (size == 0) {
    return std::nullopt;
  }
  std::wstring value(size, L'\0');
  const auto written = GetEnvironmentVariableW(
      wideName.c_str(), value.data(), static_cast<DWORD>(value.size()));
  if (written == 0 || written >= value.size()) {
    return std::nullopt;
  }
  value.resize(written);
  return pathToString(std::filesystem::path(value));
#else
  const std::string ownedName{name};
  if (const auto* value = std::getenv(ownedName.c_str());
      value != nullptr && *value != '\0') {
    return std::string(value);
  }
  return std::nullopt;
#endif
}
} // namespace detail

mlir::LogicalResult emitError(const int status, std::string message) {
  auto category = ::mqt::ErrorCategory::Runtime;
  switch (status) {
  case QDMI_ERROR_INVALIDARGUMENT:
    category = ::mqt::ErrorCategory::InvalidArgument;
    break;
  case QDMI_ERROR_OUTOFRANGE:
    category = ::mqt::ErrorCategory::OutOfRange;
    break;
  case QDMI_ERROR_OUTOFMEM:
    category = ::mqt::ErrorCategory::OutOfMemory;
    break;
  case QDMI_ERROR_NOTSUPPORTED:
    category = ::mqt::ErrorCategory::NotSupported;
    break;
  default:
    break;
  }
  return ::mqt::emitError(std::move(message), category, status);
}

mlir::LogicalResult checkError(const int result,
                               const std::string_view message) {
  if (result == QDMI_SUCCESS) {
    return mlir::success();
  }
  if (result == QDMI_WARN_GENERAL) {
    ::mqt::emitDiagnostic({
        .message = std::string(message),
        .category = ::mqt::ErrorCategory::Runtime,
        .severity = ::mqt::DiagnosticSeverity::Warning,
        .status = result,
    });
    return mlir::success();
  }
  if (result >= QDMI_ERROR_TIMEOUT && result <= QDMI_ERROR_FATAL) {
    return emitError(result, std::string(message) + ": " +
                                 toString(static_cast<QDMI_STATUS>(result)) +
                                 ".");
  }
  return emitError(result, "Unknown QDMI error code " + std::to_string(result) +
                               ". " + std::string(message));
}
} // namespace qdmi
