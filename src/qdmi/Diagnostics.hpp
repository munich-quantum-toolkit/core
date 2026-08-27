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
#include <initializer_list>
#include <string_view>

namespace qdmi::detail {

enum class DiagnosticLevel : uint8_t { Info, Warning, Error };

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

/// Writes one best-effort diagnostic to the process standard error stream.
///
/// The writer does not allocate memory. This property lets exception handlers
/// report allocation failures without throwing another exception.
inline void
emitDiagnostic(const DiagnosticLevel level,
               const std::initializer_list<std::string_view> parts) noexcept {
#ifdef _WIN32
  _lock_file(stderr);
#else
  // POSIX exposes these declarations through <cstdio>, but include-cleaner
  // does not associate them with the C++ header.
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
  for (const auto part : parts) {
    write(part);
  }
#ifdef _WIN32
  _fputc_nolock('\n', stderr);
  _fflush_nolock(stderr);
#else
  std::fputc('\n', stderr);
  std::fflush(stderr);
#endif
#ifdef _WIN32
  _unlock_file(stderr);
#else
  // NOLINTNEXTLINE(misc-include-cleaner)
  funlockfile(stderr);
#endif
}

} // namespace qdmi::detail
