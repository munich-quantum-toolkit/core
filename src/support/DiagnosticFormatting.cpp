/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "support/DiagnosticFormatting.hpp"

#include "support/Diagnostics.hpp"

#include <format>
#include <string>
#include <string_view>

namespace mqt::diagnostics::detail {
void emitFormatted(DiagnosticSeverity level, std::string_view format,
                   std::format_args args) noexcept {
  try {
    emitDiagnostic({
        .message = std::vformat(format, args),
        .category = ErrorCategory::Runtime,
        .severity = level,
    });
  } catch (...) {
    emitDiagnostic({
        .message = std::string(format),
        .category = ErrorCategory::Runtime,
        .severity = level,
    });
  }
}
} // namespace mqt::diagnostics::detail
