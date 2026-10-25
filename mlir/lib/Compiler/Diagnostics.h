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

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <system_error>

namespace mlir {
// Capture diagnostics only while a native entry point owns the context.
class CompilerDiagnostics {
public:
  explicit CompilerDiagnostics(MLIRContext* context)
      : handler(context, [this](Diagnostic& diagnostic) {
          if (diagnostic.getSeverity() != DiagnosticSeverity::Error) {
            return failure();
          }
          if (!message.empty()) {
            message.push_back('\n');
          }
          llvm::raw_string_ostream stream(message);
          print(stream, diagnostic);
          for (auto& note : diagnostic.getNotes()) {
            stream << "\nnote: ";
            print(stream, note);
          }
          return success();
        }) {}

  [[nodiscard]] llvm::Error
  takeError(std::error_code code = llvm::inconvertibleErrorCode()) const {
    return llvm::createStringError(
        code, message.empty() ? "Compiler action failed" : message);
  }

private:
  static void print(llvm::raw_ostream& stream, Diagnostic& diagnostic) {
    if (!llvm::isa<UnknownLoc>(diagnostic.getLocation())) {
      stream << diagnostic.getLocation() << ": ";
    }
    diagnostic.print(stream);
  }

  std::string message;
  ScopedDiagnosticHandler handler;
};
} // namespace mlir
