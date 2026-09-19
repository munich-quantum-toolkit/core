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

#include "dd/Error.hpp"

#include "nanobind/nanobind.h"

#include <optional>
#include <pyerrors.h>
#include <utility>
#include <variant>

namespace mqt::bindings {
[[noreturn]] inline void raiseDDError(const dd::Error& error) {
  const nanobind::gil_scoped_acquire acquire;
  auto* type = PyExc_RuntimeError;
  switch (error.kind) {
  case dd::Error::Kind::InvalidArgument:
    type = PyExc_ValueError;
    break;
  case dd::Error::Kind::OutOfRange:
    type = PyExc_IndexError;
    break;
  case dd::Error::Kind::IO:
    type = PyExc_OSError;
    break;
  case dd::Error::Kind::Numerical:
    break;
  }
  PyErr_SetString(type, error.message.c_str());
  throw nanobind::python_error();
}

template <typename T> T takeDDResult(dd::Result<T> result) {
  if (const auto* error = std::get_if<dd::Error>(&result)) {
    raiseDDError(*error);
  }
  return std::get<T>(std::move(result));
}
inline void takeDDResult(std::optional<dd::Error> error) {
  if (error) {
    raiseDDError(*error);
  }
}

template <typename C, typename R, typename... Args>
auto bindDDResult(R (C::*method)(Args...)) {
  return [method](C& self, Args... args) {
    return takeDDResult((self.*method)(std::forward<Args>(args)...));
  };
}
template <typename C, typename R, typename... Args>
auto bindDDResult(R (C::*method)(Args...) const) {
  return [method](const C& self, Args... args) {
    return takeDDResult((self.*method)(std::forward<Args>(args)...));
  };
}
} // namespace mqt::bindings
