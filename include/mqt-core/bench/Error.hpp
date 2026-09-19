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

namespace mqt::bench {

/// A recoverable benchmark input or evaluation failure.
struct Error {
  enum class Kind { InvalidArgument, Overflow };
  std::string message;
  Kind kind = Kind::InvalidArgument;
};

template <class T> using Result = std::variant<T, Error>;

} // namespace mqt::bench
