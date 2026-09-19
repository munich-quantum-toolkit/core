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

#include "bench/Error.hpp"

#include "nanobind/nanobind.h"

#include <pyerrors.h>
#include <utility>
#include <variant>

namespace mqt::bindings {

template <typename T> T takeBenchResult(bench::Result<T> result) {
  if (const auto* error = std::get_if<bench::Error>(&result)) {
    if (error->kind == bench::Error::Kind::Overflow) {
      const nanobind::gil_scoped_acquire acquire;
      PyErr_SetString(PyExc_OverflowError, error->message.c_str());
      throw nanobind::python_error();
    }
    throw nanobind::value_error(error->message.c_str());
  }
  return std::get<T>(std::move(result));
}

template <typename C, typename R, typename... Args>
auto bindBenchResult(R (C::*method)(Args...) const) {
  return [method](const C& self, Args... args) {
    return takeBenchResult((self.*method)(std::forward<Args>(args)...));
  };
}

template <typename R, typename... Args>
auto bindBenchResult(R (*function)(Args...)) {
  return [function](Args... args) {
    return takeBenchResult(function(std::forward<Args>(args)...));
  };
}

} // namespace mqt::bindings
