/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/common/Diagnostics.hpp"

#include <format>
#include <string_view>

namespace qdmi::diagnostics::detail {
void emitFormatted(DiagnosticLevel level, std::string_view format,
                   std::format_args args) noexcept {
  try {
    writeDiagnostic(level, std::vformat(format, args));
  } catch (...) {
    writeDiagnostic(level, format);
  }
}
} // namespace qdmi::diagnostics::detail
