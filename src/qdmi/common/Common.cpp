/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file Common.cpp
/// Common definitions and utilities for working with QDMI in C++.

#include "qdmi/common/Common.hpp"

#include "qdmi/constants.h"

#include <iostream>
#include <optional>
#include <string>

namespace qdmi {

std::optional<Error> checkError(const int result, const std::string& message) {
  if (result == QDMI_SUCCESS) {
    return std::nullopt;
  }
  if (result == QDMI_WARN_GENERAL) {
    std::cerr << "Warning: " << message << '\n';
    return std::nullopt;
  }
  if (result >= QDMI_ERROR_TIMEOUT && result <= QDMI_ERROR_FATAL) {
    return Error{
        .status = result,
        .message =
            message + ": " + toString(static_cast<QDMI_STATUS>(result)) + ".",
    };
  }
  return Error{
      .status = result,
      .message =
          "Unknown QDMI error code " + std::to_string(result) + ". " + message,
  };
}

} // namespace qdmi
