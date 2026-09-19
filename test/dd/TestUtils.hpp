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

#include <optional>
#include <stdexcept>
#include <utility>
#include <variant>

namespace dd::test {

template <typename T> T value(Result<T> result) {
  if (const auto* error = std::get_if<Error>(&result)) {
    throw std::runtime_error(error->message);
  }
  return std::get<T>(std::move(result));
}

inline void value(const std::optional<Error>& error) {
  if (error) {
    throw std::runtime_error(error->message);
  }
}
inline std::optional<Error::Kind> errorKind(const std::optional<Error>& error) {
  return error ? std::optional(error->kind) : std::nullopt;
}

template <typename T>
std::optional<Error::Kind> errorKind(const Result<T>& result) {
  if (const auto* error = std::get_if<Error>(&result)) {
    return error->kind;
  }
  return std::nullopt;
}

} // namespace dd::test
