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

#include "nanobind/nanobind.h"

#include "llvm/Support/Error.h"

#include <functional>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace mqt::bindings {
/// Raise the Python exception matching an owned native error.
[[noreturn]] inline void raiseError(llvm::Error error) {
  std::error_code code;
  llvm::visitErrors(error, [&](const llvm::ErrorInfoBase& info) {
    code = info.convertToErrorCode();
  });
  // Consume inside LLVM, whose RTTI setting may differ from nanobind's.
  const auto message = llvm::toString(std::move(error));
  const nanobind::gil_scoped_acquire acquire;
  if (code == std::errc::invalid_argument) {
    throw nanobind::value_error(message.c_str());
  }
  if (code == std::errc::result_out_of_range) {
    throw nanobind::index_error(message.c_str());
  }
  const char* type = nullptr;
  if (code == std::errc::value_too_large) {
    type = "OverflowError";
  } else if (code == std::errc::not_enough_memory) {
    type = "MemoryError";
  } else if (code == std::errc::io_error) {
    type = "OSError";
  }
  if (type != nullptr) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg): nanobind API.
    nanobind::chain_error(nanobind::builtins()[type], "%s", message.c_str());
    throw nanobind::python_error();
  }
  throw std::runtime_error(message);
}

inline void unwrap(llvm::Error error) {
  if (error) {
    raiseError(std::move(error));
  }
}

template <class T> T unwrap(llvm::Expected<T> result) {
  if (!result) {
    raiseError(result.takeError());
  }
  return std::move(*result);
}

/// Invoke a result-returning function and translate its error at the boundary.
template <class Function, class... Args>
auto invoke(Function&& function, Args&&... args) {
  return unwrap(std::invoke(std::forward<Function>(function),
                            std::forward<Args>(args)...));
}

/// Adapt a mutable member function through invoke() for nanobind.
template <typename C, typename R, typename... Args>
auto bindResult(R (C::*method)(Args...)) {
  return [method](C& self, Args... args) {
    return ::mqt::bindings::invoke(method, self, std::forward<Args>(args)...);
  };
}
/// Adapt a const member function through invoke() for nanobind.
template <typename C, typename R, typename... Args>
auto bindResult(R (C::*method)(Args...) const) {
  return [method](const C& self, Args... args) {
    return ::mqt::bindings::invoke(method, self, std::forward<Args>(args)...);
  };
}
/// Adapt a free function through invoke() for nanobind.
template <typename R, typename... Args>
auto bindResult(R (*function)(Args...)) {
  return [function](Args... args) {
    return ::mqt::bindings::invoke(function, std::forward<Args>(args)...);
  };
}
} // namespace mqt::bindings
