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

#include "nanobind/nanobind.h"

#include "llvm/Support/LogicalResult.h"

#include <functional>
#include <optional>
#include <pyerrors.h>
#include <utility>

namespace mqt::bindings {
/// Acquire the GIL and raise the Python exception matching the diagnostic.
[[noreturn]] inline void raiseDiagnostic(const Diagnostic& diagnostic) {
  const nanobind::gil_scoped_acquire acquire;
  auto* type = PyExc_RuntimeError;
  switch (diagnostic.category) {
  case ErrorCategory::InvalidArgument:
    type = PyExc_ValueError;
    break;
  case ErrorCategory::OutOfRange:
    type = PyExc_IndexError;
    break;
  case ErrorCategory::Overflow:
    type = PyExc_OverflowError;
    break;
  case ErrorCategory::IO:
    type = PyExc_OSError;
    break;
  case ErrorCategory::OutOfMemory:
    type = PyExc_MemoryError;
    break;
  case ErrorCategory::Numerical:
  case ErrorCategory::Runtime:
  case ErrorCategory::NotSupported:
    break;
  }
  PyErr_SetString(type, diagnostic.message.c_str());
  throw nanobind::python_error();
}

/// Invoke a result-returning function, unwrapping success or raising its error.
/// Capture diagnostics before invocation, including functions that release the
/// GIL. Warnings retain the surrounding handler or stderr behavior.
template <class Function, class... Args>
auto invoke(Function&& function, Args&&... args) {
  std::optional<Diagnostic> error;
  ScopedDiagnosticHandler handler([&](const Diagnostic& diagnostic) {
    if (diagnostic.severity != DiagnosticSeverity::Error) {
      return llvm::failure();
    }
    if (!error) {
      error = diagnostic;
    }
    return llvm::success();
  });
  auto result = std::invoke(std::forward<Function>(function),
                            std::forward<Args>(args)...);
  if (llvm::failed(result)) {
    if (error) {
      raiseDiagnostic(*error);
    }
    raiseDiagnostic({
        .message = "Compiler action failed; see diagnostics for details.",
    });
  }
  if constexpr (requires { typename decltype(result)::value_type; }) {
    return std::move(*result);
  }
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
