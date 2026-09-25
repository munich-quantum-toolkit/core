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

#include "support/Diagnostics.hpp"

#include "gtest/gtest.h"

#include "mlir/Support/LogicalResult.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <random>
#include <string>
#include <utility>

namespace mqt::test {
/// Capture diagnostics before a tested operation, including move-only results.
class DiagnosticCapture {
public:
  std::optional<Diagnostic> error;
  ScopedDiagnosticHandler handler{[this](const Diagnostic& diagnostic) {
    if (diagnostic.severity != DiagnosticSeverity::Error) {
      return mlir::failure();
    }
    if (!error) {
      error = diagnostic;
    }
    return mlir::success();
  }};
};

template <class Result> auto value(Result result) {
  if (failed(result)) {
    ADD_FAILURE() << "Expected a successful result";
    std::abort();
  }
  if constexpr (requires { typename Result::value_type; }) {
    return std::move(*result);
  }
}

template <class Function>
std::optional<Diagnostic> diagnostic(Function&& function) {
  DiagnosticCapture capture;
  auto const result = std::invoke(std::forward<Function>(function));
  EXPECT_EQ(failed(result), capture.error.has_value());
  return std::move(capture.error);
}
template <class Function> std::string errorMessage(Function&& function) {
  auto error = diagnostic(std::forward<Function>(function));
  return error ? error->message : std::string{};
}
template <class Function>
std::optional<ErrorCategory> errorKind(Function&& function) {
  auto error = diagnostic(std::forward<Function>(function));
  return error ? std::optional(error->category) : std::nullopt;
}
template <class Function> std::optional<int> errorStatus(Function&& function) {
  auto error = diagnostic(std::forward<Function>(function));
  return error ? error->status : std::nullopt;
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
      ADD_FAILURE() << "Failed to set environment variable " << name_;
      std::abort();
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

class TemporaryDirectory {
public:
  TemporaryDirectory() {
    path_ = std::filesystem::temp_directory_path() /
            ("mqt-core-qdmi-registry-test-" +
             std::to_string(std::random_device{}()));
    std::filesystem::remove_all(path_);
    std::filesystem::create_directories(path_);
  }

  ~TemporaryDirectory() { std::filesystem::remove_all(path_); }

  [[nodiscard]] const std::filesystem::path& path() const { return path_; }

  [[nodiscard]] std::filesystem::path
  write(const std::filesystem::path& relative,
        const std::string& contents) const {
    const auto path = path_ / relative;
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path);
    output << contents;
    return path;
  }

private:
  std::filesystem::path path_;
};

} // namespace mqt::test
