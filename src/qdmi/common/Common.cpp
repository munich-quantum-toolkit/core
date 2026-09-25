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

#include "support/Diagnostics.hpp"

#include "qdmi/constants.h"

#include <string>
#include <string_view>

#include "mlir/Support/LogicalResult.h"

#include <utility>

namespace qdmi {

mlir::LogicalResult emitError(const int status, std::string message) {
  auto category = ::mqt::ErrorCategory::Runtime;
  switch (status) {
  case QDMI_ERROR_INVALIDARGUMENT:
    category = ::mqt::ErrorCategory::InvalidArgument;
    break;
  case QDMI_ERROR_OUTOFRANGE:
    category = ::mqt::ErrorCategory::OutOfRange;
    break;
  case QDMI_ERROR_OUTOFMEM:
    category = ::mqt::ErrorCategory::OutOfMemory;
    break;
  case QDMI_ERROR_NOTSUPPORTED:
    category = ::mqt::ErrorCategory::NotSupported;
    break;
  default:
    break;
  }
  return ::mqt::emitError(std::move(message), category, status);
}

mlir::LogicalResult checkError(const int result,
                               const std::string_view message) {
  if (result == QDMI_SUCCESS) {
    return mlir::success();
  }
  if (result == QDMI_WARN_GENERAL) {
    ::mqt::emitDiagnostic({
        .message = std::string(message),
        .category = ::mqt::ErrorCategory::Runtime,
        .severity = ::mqt::DiagnosticSeverity::Warning,
        .status = result,
    });
    return mlir::success();
  }
  if (result >= QDMI_ERROR_TIMEOUT && result <= QDMI_ERROR_FATAL) {
    return emitError(result, std::string(message) + ": " +
                                 toString(static_cast<QDMI_STATUS>(result)) +
                                 ".");
  }
  return emitError(result, "Unknown QDMI error code " + std::to_string(result) +
                               ". " + std::string(message));
}
} // namespace qdmi
