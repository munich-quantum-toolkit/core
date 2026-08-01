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

#include "mlir/Target/OpenQASM/Detail/OpenQASMSyntax.h"
#include "mlir/Target/OpenQASM/Frontend.h"

#include <llvm/Support/SourceMgr.h>

namespace mlir::oq3::frontend::detail {

[[nodiscard]] AnalysisResult
analyzeSyntaxProgram(const SyntaxProgram& syntax,
                     const llvm::SourceMgr& sources,
                     const FrontendOptions& options);

[[nodiscard]] SourceLocation sourceLocation(const llvm::SourceMgr& sources,
                                            llvm::SMLoc location);

} // namespace mlir::oq3::frontend::detail
