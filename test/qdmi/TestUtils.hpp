/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file TestUtils.hpp
/// Shared test utilities for QDMI components.

#pragma once

#include "qdmi/common/Common.hpp"

#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace mqt::test {

/// Fail a success-path test with the provider diagnostic.
template <typename T> T value(qdmi::Result<T> result) {
  if (const auto* error = std::get_if<qdmi::Error>(&result)) {
    throw std::runtime_error(error->message);
  }
  return std::get<0>(std::move(result));
}
inline void value(const std::optional<qdmi::Error>& error) {
  if (error) {
    throw std::runtime_error(error->message);
  }
}
template <typename T>
std::optional<int> errorStatus(const qdmi::Result<T>& result) {
  if (const auto* error = std::get_if<qdmi::Error>(&result)) {
    return error->status;
  }
  return std::nullopt;
}
inline std::optional<int> errorStatus(const std::optional<qdmi::Error>& error) {
  return error ? std::optional(error->status) : std::nullopt;
}

/// Temporarily sets or unsets an environment variable and restores its value.
class ScopedEnvironmentVariable {
public:
  ScopedEnvironmentVariable(std::string name,
                            const std::optional<std::string>& value)
      : name_(std::move(name)) {
    if (const auto* previous = std::getenv(name_.c_str());
        previous != nullptr) {
      previous_ = previous;
    }
    set(value);
  }

  ~ScopedEnvironmentVariable() {
    if (!setWithoutChecking(previous_)) {
      std::abort();
    }
  }

  ScopedEnvironmentVariable(const ScopedEnvironmentVariable&) = delete;
  ScopedEnvironmentVariable&
  operator=(const ScopedEnvironmentVariable&) = delete;
  ScopedEnvironmentVariable(ScopedEnvironmentVariable&&) = delete;
  ScopedEnvironmentVariable& operator=(ScopedEnvironmentVariable&&) = delete;

private:
  void set(const std::optional<std::string>& value) const {
    if (!setWithoutChecking(value)) {
      throw std::runtime_error("Failed to set environment variable " + name_);
    }
  }

  [[nodiscard]] bool
  setWithoutChecking(const std::optional<std::string>& value) const {
#ifdef _WIN32
    return _putenv_s(name_.c_str(), value.value_or("").c_str()) == 0;
#else
    const auto* const name = name_.c_str();
    /// NOLINTNEXTLINE(misc-include-cleaner): POSIX environment functions.
    return (value ? setenv(name, value->c_str(), 1) : unsetenv(name)) == 0;
#endif
  }

  std::string name_;
  std::optional<std::string> previous_;
};

} // namespace mqt::test
