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
/// @brief Common definitions and utilities for working with QDMI in C++.
/// @note This file will be upstreamed to the QDMI core library in the future.

#include "qdmi/Common.hpp"

#include <iostream>
#include <qdmi/constants.h>
#include <sstream>
#include <stdexcept>
#include <string>

namespace qdmi {

auto throwIfError(const int result, const std::string& msg) -> void {
  switch (const auto res = static_cast<QDMI_STATUS>(result)) {
  case QDMI_SUCCESS:
    break;
  case QDMI_WARN_GENERAL:
    std::cerr << "Warning: " << msg << '\n';
    break;
  default:
    std::ostringstream ss;
    ss << msg << ": " << toString(res) << ".";
    switch (res) {
    case QDMI_ERROR_OUTOFMEM:
      throw std::bad_alloc();
    case QDMI_ERROR_OUTOFRANGE:
      throw std::out_of_range(ss.str());
    case QDMI_ERROR_INVALIDARGUMENT:
      throw std::invalid_argument(ss.str());
    case QDMI_ERROR_FATAL:
    case QDMI_ERROR_NOTIMPLEMENTED:
    case QDMI_ERROR_LIBNOTFOUND:
    case QDMI_ERROR_NOTFOUND:
    case QDMI_ERROR_PERMISSIONDENIED:
    case QDMI_ERROR_NOTSUPPORTED:
    case QDMI_ERROR_BADSTATE:
    case QDMI_ERROR_TIMEOUT:
      throw std::runtime_error(ss.str());
    default:
      throw std::runtime_error("Unknown QDMI error code. " + ss.str());
    }
  }
}

} // namespace qdmi
