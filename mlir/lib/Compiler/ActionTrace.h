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

#include "mqt/Compiler/CompilationOptions.h"
#include "mqt/Compiler/Programs.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <utility>

namespace mlir {

/// Run a compilation step on `program` with an optional action trace.
///
/// When `options` requests a trace, the function imports OpenQASM source in
/// `program` and traces the actions on the program's context. See the
/// context-based withActionTrace in `mqt/Support/Passes.h` for the
/// tracing behavior.
///
/// @param program The input that `run` compiles.
/// @param options The requested compilation options.
/// @param run The compilation step.
/// @return The error from `run`, from opening or writing the file, or both.
[[nodiscard]] llvm::Error
withActionTrace(CompilerInput& program, const CompilationOptions& options,
                llvm::function_ref<llvm::Error(const CompilationOptions&)> run);

/// Run a compilation step that returns a value with an optional action trace.
///
/// See the overload that returns `llvm::Error` for the tracing behavior.
template <typename T>
[[nodiscard]] llvm::Expected<T> withActionTrace(
    CompilerInput& program, const CompilationOptions& options,
    llvm::function_ref<llvm::Expected<T>(const CompilationOptions&)> run) {
  std::optional<T> value;
  if (auto error = withActionTrace(
          program, options,
          [&](const CompilationOptions& stepOptions) -> llvm::Error {
            auto result = run(stepOptions);
            if (!result) {
              return result.takeError();
            }
            value.emplace(std::move(*result));
            return llvm::Error::success();
          })) {
    return std::move(error);
  }
  return std::move(*value);
}

} // namespace mlir
