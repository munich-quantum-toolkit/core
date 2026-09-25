/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "support/Diagnostics.hpp"

#include "mlir/Support/LogicalResult.h"

#include <cassert>
#include <cstdio>
#include <functional>
#include <mutex>
#include <utility>

namespace mqt {
namespace {
/// The mutable per-thread stack stays private to the library, outside DLL
/// exports. NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
thread_local ScopedDiagnosticHandler* currentHandler = nullptr;
} // namespace

ScopedDiagnosticHandler::ScopedDiagnosticHandler(
    std::function<mlir::LogicalResult(const Diagnostic&)> handler)
    : previous_(currentHandler), handler_(std::move(handler)) {
  currentHandler = this;
}
ScopedDiagnosticHandler::~ScopedDiagnosticHandler() {
  assert(currentHandler == this &&
         "diagnostic handlers must leave in stack order");
  currentHandler = previous_;
}

void emitDiagnostic(const Diagnostic& diagnostic) {
  auto* const installed = currentHandler;
  for (auto const* handler = installed; handler != nullptr;
       handler = handler->previous_) {
    currentHandler = handler->previous_;
    const auto consumed = mlir::succeeded(handler->handler_(diagnostic));
    currentHandler = installed;
    if (consumed) {
      return;
    }
  }
  const char* severity = "error";
  if (diagnostic.severity == DiagnosticSeverity::Warning) {
    severity = "warning";
  } else if (diagnostic.severity == DiagnosticSeverity::Info) {
    severity = "info";
  }
  static std::mutex stderrMutex;
  const std::scoped_lock lock(stderrMutex);
  std::fputs("[mqt-core] [", stderr);
  std::fputs(severity, stderr);
  std::fputs("] ", stderr);
  std::fwrite(diagnostic.message.data(), 1, diagnostic.message.size(), stderr);
  std::fputc('\n', stderr);
  std::fflush(stderr);
}
} // namespace mqt
