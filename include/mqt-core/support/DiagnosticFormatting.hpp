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

#include "support/Diagnostics.hpp"

#include <format>
#include <string_view>
#include <utility>

namespace mqt::diagnostics {
namespace detail {
MQT_CORE_SUPPORT_EXPORT void emitFormatted(DiagnosticSeverity level,
                                           std::string_view format,
                                           std::format_args args) noexcept;
}

/// Formats and writes one best-effort diagnostic to standard error.
///
/// Writes the unformatted format string if formatting fails.
template <class... Args>
void emit(const DiagnosticSeverity level,
          const std::format_string<Args...> format, Args&&... args) noexcept {
  detail::emitFormatted(level, format.get(), std::make_format_args(args...));
}

/// Formats and writes an informational diagnostic to standard error.
template <class... Args>
void info(const std::format_string<Args...> format, Args&&... args) noexcept {
  emit(DiagnosticSeverity::Info, format, std::forward<Args>(args)...);
}

/// Formats and writes a warning diagnostic to standard error.
template <class... Args>
void warn(const std::format_string<Args...> format, Args&&... args) noexcept {
  emit(DiagnosticSeverity::Warning, format, std::forward<Args>(args)...);
}

/// Formats and writes an error diagnostic to standard error.
template <class... Args>
void error(const std::format_string<Args...> format, Args&&... args) noexcept {
  emit(DiagnosticSeverity::Error, format, std::forward<Args>(args)...);
}

} // namespace mqt::diagnostics
