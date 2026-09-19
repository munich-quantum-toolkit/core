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

#include "qdmi/common/Common.hpp"

#include "nanobind/nanobind.h"
#include "qdmi/constants.h"

#include <optional>
#include <pyerrors.h>
#include <utility>
#include <variant>

namespace mqt::bindings {

[[noreturn]] inline void raiseQDMIError(const qdmi::Error& error) {
  namespace nb = nanobind;
  switch (error.status) {
  case QDMI_ERROR_INVALIDARGUMENT:
    throw nb::value_error(error.message.c_str());
  case QDMI_ERROR_OUTOFRANGE:
    throw nb::index_error(error.message.c_str());
  case QDMI_ERROR_OUTOFMEM: {
    const nb::gil_scoped_acquire acquire;
    PyErr_SetString(PyExc_MemoryError, error.message.c_str());
    throw nb::python_error();
  }
  default:
    throw nb::builtin_exception(nb::exception_type::runtime_error,
                                error.message.c_str());
  }
}

template <typename T> T takeQDMIResult(qdmi::Result<T> result) {
  if (const auto* error = std::get_if<qdmi::Error>(&result)) {
    raiseQDMIError(*error);
  }
  return std::get<T>(std::move(result));
}

inline void takeQDMIResult(const std::optional<qdmi::Error>& error) {
  if (error) {
    raiseQDMIError(*error);
  }
}

template <typename C, typename R, typename... Args>
auto bindQDMIResult(R (C::*method)(Args...) const) {
  return [method](const C& self, Args... args) {
    return takeQDMIResult((self.*method)(std::forward<Args>(args)...));
  };
}

} // namespace mqt::bindings
