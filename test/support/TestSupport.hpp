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

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#ifdef _WIN32
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ConvertUTF.h"
#endif

#include <cstdlib>
#include <filesystem>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace mqt::test {
template <class T>
::testing::AssertionResult succeeded(llvm::Expected<T>& result) {
  if (result) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << llvm::toString(result.takeError());
}

inline ::testing::AssertionResult succeeded(llvm::Error error) {
  if (error) {
    return ::testing::AssertionFailure() << llvm::toString(std::move(error));
  }
  return ::testing::AssertionSuccess();
}

template <class T>
::testing::AssertionResult succeeded(const llvm::FailureOr<T>& result) {
  return llvm::succeeded(result)
             ? ::testing::AssertionSuccess()
             : ::testing::AssertionFailure() << "Expected a successful result";
}

/// Extract a test value; GoogleTest reports exceptions without ending the
/// suite.
template <class T> T value(llvm::Expected<T> result) {
  if (!result) {
    throw std::runtime_error(llvm::toString(result.takeError()));
  }
  return std::move(*result);
}

inline void value(llvm::Error error) {
  if (error) {
    throw std::runtime_error(llvm::toString(std::move(error)));
  }
}

template <class T> T value(llvm::FailureOr<T> result) {
  if (llvm::failed(result)) {
    throw std::runtime_error("Expected a successful result");
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
      : name_(std::move(name)), previous_(llvm::sys::Process::GetEnv(name_)) {
    set(value);
  }

  ~ScopedEnvironmentVariable() {
    EXPECT_TRUE(setWithoutChecking(previous_))
        << "Failed to restore environment variable " << name_;
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
    llvm::SmallVector<wchar_t, 128> name;
    llvm::SmallVector<wchar_t, 128> contents;
    if (llvm::sys::windows::UTF8ToUTF16(name_, name) ||
        llvm::sys::windows::UTF8ToUTF16(value.value_or(""), contents)) {
      return false;
    }
    return _wputenv_s(name.data(), contents.data()) == 0;
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
    llvm::SmallString<128> path;
    if (const auto error =
            llvm::sys::fs::createUniqueDirectory("mqt-core-test", path)) {
      throw std::system_error(error);
    }
    path_ = std::u8string(path.begin(), path.end());
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
    EXPECT_FALSE(error) << "Failed to remove temporary directory: "
                        << error.message();
  }

  TemporaryDirectory(const TemporaryDirectory&) = delete;
  TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;
  TemporaryDirectory(TemporaryDirectory&&) = delete;
  TemporaryDirectory& operator=(TemporaryDirectory&&) = delete;

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
    if (error) {
      output.clear_error();
      throw std::system_error(error);
    }
    output << contents;
    output.close();
    error = output.error();
    output.clear_error();
    if (error) {
      throw std::system_error(error);
    }
    return path;
  }

private:
  std::filesystem::path path_;
};

} // namespace mqt::test
