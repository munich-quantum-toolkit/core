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

#include <mlir/IR/OwningOpRef.h>

#include <string_view>

namespace mlir {

// Forward declarations
class MLIRContext;
class ModuleOp;

namespace qc::detail {

/**
 * @brief Parse an OpenQASM 3 program and emit the equivalent QC dialect module.
 *
 * @details
 * Implementation detail of `translateQASM3ToQC`; not part of the public API
 * (this header lives with the library sources and is not installed). Prefer the
 * `translateQASM3ToQC` entry points declared in
 * `mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h`, which wrap this in the
 * diagnostic-reporting `try`/`catch`.
 *
 * This is a single-pass, hand-written recursive-descent parser that consumes
 * `qasm3::Scanner` tokens directly and emits QC operations via the
 * `QCProgramBuilder` as it parses, without building an intermediate AST. It
 * depends on the legacy `qasm3` package only for its `Scanner`/`Token` lexer,
 * the `GateInfo`/`NestedEnvironment` helpers, and `DebugInfo`/`CompilerError`
 * for diagnostics.
 *
 * On a malformed program it throws a `qasm3::CompilerError` (or another
 * `std::exception`); callers are expected to translate that into a diagnostic.
 *
 * @param source The OpenQASM 3 program text.
 * @param context The MLIRContext to create the module in.
 * @return The constructed QC dialect module.
 */
[[nodiscard]] OwningOpRef<ModuleOp> parseQASM3(std::string_view source,
                                               MLIRContext* context);

} // namespace qc::detail

} // namespace mlir
