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

#include <cstdint>
#include <cstdio>
#include <format>
#include <string_view>
#include <utility>

namespace qdmi::diagnostics {

/// Severity of a diagnostic.
enum class DiagnosticLevel : uint8_t { Info, Warning, Error };

namespace detail {

[[nodiscard]] constexpr std::string_view
diagnosticLevelName(const DiagnosticLevel level) noexcept {
  switch (level) {
  case DiagnosticLevel::Info:
    return "info";
  case DiagnosticLevel::Warning:
    return "warning";
  case DiagnosticLevel::Error:
    return "error";
  }
  return "error";
}

/// Writes one diagnostic to standard error without allocating memory.
inline void writeDiagnostic(const DiagnosticLevel level,
                            const std::string_view message) noexcept {
#ifdef _WIN32
  _lock_file(stderr);
#else
  // NOLINTNEXTLINE(misc-include-cleaner)
  flockfile(stderr);
#endif
  const auto write = [](const std::string_view part) noexcept {
    if (!part.empty()) {
#ifdef _WIN32
      _fwrite_nolock(part.data(), sizeof(char), part.size(), stderr);
#else
      std::fwrite(part.data(), sizeof(char), part.size(), stderr);
#endif
    }
  };
  write("[mqt-core] [");
  write(diagnosticLevelName(level));
  write("] ");
  write(message);
#ifdef _WIN32
  _fputc_nolock('\n', stderr);
  _fflush_nolock(stderr);
  _unlock_file(stderr);
#else
  std::fputc('\n', stderr);
  std::fflush(stderr);
  // NOLINTNEXTLINE(misc-include-cleaner)
  funlockfile(stderr);
#endif
}

} // namespace detail

/// Formats and writes one best-effort diagnostic to standard error.
///
/// Writes the unformatted format string if formatting fails.
template <class... Args>
void emit(const DiagnosticLevel level, const std::format_string<Args...> format,
          Args&&... args) noexcept {
  try {
    detail::writeDiagnostic(level,
                            std::format(format, std::forward<Args>(args)...));
  } catch (...) {
    detail::writeDiagnostic(level, format.get());
  }
}

/// Formats and writes an informational diagnostic to standard error.
template <class... Args>
void info(const std::format_string<Args...> format, Args&&... args) noexcept {
  emit(DiagnosticLevel::Info, format, std::forward<Args>(args)...);
}

/// Formats and writes a warning diagnostic to standard error.
template <class... Args>
void warn(const std::format_string<Args...> format, Args&&... args) noexcept {
  emit(DiagnosticLevel::Warning, format, std::forward<Args>(args)...);
}

/// Formats and writes an error diagnostic to standard error.
template <class... Args>
void error(const std::format_string<Args...> format, Args&&... args) noexcept {
  emit(DiagnosticLevel::Error, format, std::forward<Args>(args)...);
}

} // namespace qdmi::diagnostics
