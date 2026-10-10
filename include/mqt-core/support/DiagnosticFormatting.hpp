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
void emitToStderr(DiagnosticSeverity level, std::string_view message) noexcept;
MQT_CORE_SUPPORT_EXPORT void emitFormatted(DiagnosticSeverity level,
                                           std::string_view format,
                                           std::format_args args) noexcept;
} // namespace detail

/// Formats and emits one best-effort diagnostic.
///
/// Writes the unformatted format string directly to stderr if formatting fails.
template <class... Args>
void emit(const DiagnosticSeverity level,
          // make_format_args stores references and requires lvalues.
          // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
          const std::format_string<Args...> format, Args&&... args) noexcept {
  detail::emitFormatted(level, format.get(), std::make_format_args(args...));
}

/// Formats and emits an informational diagnostic.
template <class... Args>
void info(const std::format_string<Args...> format, Args&&... args) noexcept {
  emit(DiagnosticSeverity::Info, format, std::forward<Args>(args)...);
}

/// Formats and emits a warning diagnostic.
template <class... Args>
void warn(const std::format_string<Args...> format, Args&&... args) noexcept {
  emit(DiagnosticSeverity::Warning, format, std::forward<Args>(args)...);
}

/// Formats and emits an error diagnostic.
template <class... Args>
void error(const std::format_string<Args...> format, Args&&... args) noexcept {
  emit(DiagnosticSeverity::Error, format, std::forward<Args>(args)...);
}

} // namespace mqt::diagnostics
