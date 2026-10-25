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

#include "qdmi/common/Diagnostics.hpp"

#include "qdmi/constants.h"

#include "llvm/Support/raw_ostream.h"

#include <stdexcept>
#include <string>

namespace qdmi {
auto throwIfError(const int result, const std::string& msg) -> void {
  switch (const auto res = static_cast<QDMI_STATUS>(result)) {
  case QDMI_SUCCESS:
    break;
  case QDMI_WARN_GENERAL:
    qdmi::diagnostics::warn("{}", msg);
    break;
  default:
    std::string ssStorage;
    llvm::raw_string_ostream ss(ssStorage);
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
