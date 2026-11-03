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
#include <llvm/Support/FormatVariadic.h>
#include <utility>

namespace qdmi::diagnostics {

/// Severity of a diagnostic.
enum class DiagnosticLevel : uint8_t { Info, Warning, Error };

namespace detail {

/// Writes one formatted diagnostic.
void emit(DiagnosticLevel level,
          const llvm::formatv_object_base& message) noexcept;

} // namespace detail

/// Formats and writes one diagnostic to standard error.
template <class... Args>
void emit(const DiagnosticLevel level, const char* format,
          Args&&... args) noexcept {
  detail::emit(level, llvm::formatv(format, std::forward<Args>(args)...));
}

/// Formats and writes an informational diagnostic to standard error.
template <class... Args>
void info(const char* format, Args&&... args) noexcept {
  emit(DiagnosticLevel::Info, format, std::forward<Args>(args)...);
}

/// Formats and writes a warning diagnostic to standard error.
template <class... Args>
void warn(const char* format, Args&&... args) noexcept {
  emit(DiagnosticLevel::Warning, format, std::forward<Args>(args)...);
}

/// Formats and writes an error diagnostic to standard error.
template <class... Args>
void error(const char* format, Args&&... args) noexcept {
  emit(DiagnosticLevel::Error, format, std::forward<Args>(args)...);
}

} // namespace qdmi::diagnostics
