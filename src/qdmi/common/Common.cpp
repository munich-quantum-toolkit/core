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
/// @brief Common definitions and utilities for working with QDMI in C++.
/// @note This file will be upstreamed to the QDMI core library in the future.

#include "qdmi/common/Common.hpp"

#include <qdmi/constants.h>

#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#ifdef _WIN32
#include <windows.h>

#include <filesystem>
#endif

namespace qdmi {
namespace detail {
auto environmentUtf8(const std::string_view name)
    -> std::optional<std::string> {
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
  return pathToUtf8(std::filesystem::path(value));
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

auto throwIfError(const int result, const std::string& msg) -> void {
  switch (const auto res = static_cast<QDMI_STATUS>(result)) {
  case QDMI_SUCCESS:
    break;
  case QDMI_WARN_GENERAL:
    std::cerr << "Warning: " << msg << '\n';
    break;
  default:
    std::ostringstream ss;
    ss << msg << ": " << toString(res) << ".";
    switch (res) {
    case QDMI_ERROR_OUTOFMEM:
      throw std::bad_alloc();
    case QDMI_ERROR_OUTOFRANGE:
      throw std::out_of_range(ss.str());
    case QDMI_ERROR_INVALIDARGUMENT:
      throw std::invalid_argument(ss.str());
    case QDMI_ERROR_FATAL:
    case QDMI_ERROR_NOTIMPLEMENTED:
    case QDMI_ERROR_LIBNOTFOUND:
    case QDMI_ERROR_NOTFOUND:
    case QDMI_ERROR_PERMISSIONDENIED:
    case QDMI_ERROR_NOTSUPPORTED:
    case QDMI_ERROR_BADSTATE:
    case QDMI_ERROR_TIMEOUT:
      throw std::runtime_error(ss.str());
    default:
      throw std::runtime_error("Unknown QDMI error code. " + ss.str());
    }
  }
}

} // namespace qdmi
