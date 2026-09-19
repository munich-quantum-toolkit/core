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

#include <string>
#include <variant>

namespace dd {

/// A recoverable decision-diagram input or operational failure.
struct Error {
  enum class Kind { InvalidArgument, OutOfRange, Numerical, IO };
  std::string message;
  Kind kind = Kind::InvalidArgument;
};

template <class T> using Result = std::variant<T, Error>;

} // namespace dd
