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

#include "support/mqt_core_support_export.h"

#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>

namespace mqt {
enum class DiagnosticSeverity : uint8_t { Info, Warning, Error };
enum class ErrorCategory : uint8_t {
  InvalidArgument,
  OutOfRange,
  Overflow,
  Numerical,
  IO,
  Runtime,
  OutOfMemory,
  NotSupported
};

struct Diagnostic {
  std::string message;
  ErrorCategory category = ErrorCategory::Runtime;
  DiagnosticSeverity severity = DiagnosticSeverity::Error;
  std::optional<int> status = std::nullopt;
};

/// Unhandled diagnostics are written to stderr.
MQT_CORE_SUPPORT_EXPORT void emitDiagnostic(const Diagnostic& diagnostic);

/// Handlers run synchronously on their installing thread, newest first.
/// Success consumes the diagnostic; failure forwards it. Handlers must not
/// throw. A handler may emit another diagnostic, which starts at the previous
/// handler.
class MQT_CORE_SUPPORT_EXPORT ScopedDiagnosticHandler {
public:
  explicit ScopedDiagnosticHandler(
      std::function<mlir::LogicalResult(const Diagnostic&)> handler);
  ~ScopedDiagnosticHandler();
  ScopedDiagnosticHandler(const ScopedDiagnosticHandler&) = delete;
  ScopedDiagnosticHandler& operator=(const ScopedDiagnosticHandler&) = delete;

private:
  friend void emitDiagnostic(const Diagnostic& diagnostic);
  ScopedDiagnosticHandler* previous_;
  std::function<mlir::LogicalResult(const Diagnostic&)> handler_;
};

/// Emit a diagnostic and return failure without putting a message in the
/// result.
[[nodiscard]] inline mlir::LogicalResult
emitError(std::string message, ErrorCategory category = ErrorCategory::Runtime,
          std::optional<int> status = std::nullopt) {
  emitDiagnostic(
      {std::move(message), category, DiagnosticSeverity::Error, status});
  return mlir::failure();
}
} // namespace mqt
