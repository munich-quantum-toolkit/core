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

#include "gtest/gtest.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <filesystem>
#include <functional>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace mqt::test {
template <class T> T value(llvm::Expected<T> result) {
  if (!result) {
    ADD_FAILURE() << llvm::toString(result.takeError());
    std::abort();
  }
  return std::move(*result);
}

inline void value(llvm::Error error) {
  if (error) {
    ADD_FAILURE() << llvm::toString(std::move(error));
    std::abort();
  }
}

template <class T> T value(llvm::FailureOr<T> result) {
  if (llvm::failed(result)) {
    ADD_FAILURE() << "Expected a successful result";
    std::abort();
  }
  return std::move(*result);
}

inline llvm::Error takeError(llvm::Error error) { return error; }
template <class T> llvm::Error takeError(llvm::Expected<T> result) {
  EXPECT_FALSE(static_cast<bool>(result));
  return result.takeError();
}

template <class Function> std::string errorMessage(Function&& function) {
  return llvm::toString(
      takeError(std::invoke(std::forward<Function>(function))));
}

template <class Function> std::error_code errorCode(Function&& function) {
  return llvm::errorToErrorCode(
      takeError(std::invoke(std::forward<Function>(function))));
}

template <class Function> std::optional<int> errorStatus(Function&& function) {
  const auto code = errorCode(std::forward<Function>(function));
  if (std::string_view(code.category().name()) == "qdmi") {
    return code.value();
  }
  return std::nullopt;
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
    // NOLINTNEXTLINE(misc-include-cleaner): POSIX environment functions.
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
    // Truncate in place to preserve permissions of executable test fixtures.
    std::error_code error;
    const auto utf8 = path.u8string();
    llvm::raw_fd_ostream output(std::string(utf8.begin(), utf8.end()), error);
    llvm::cantFail(llvm::errorCodeToError(error));
    output << contents;
    output.close();
    error = output.error();
    output.clear_error();
    llvm::cantFail(llvm::errorCodeToError(error));
    return path;
  }

private:
  std::filesystem::path path_;
};

} // namespace mqt::test
