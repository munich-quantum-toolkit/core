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

#include "mlir/Target/OpenQASM/Frontend.h"

#include <llvm/ADT/StringRef.h>
#include <mlir/IR/OwningOpRef.h>

namespace llvm {
class SourceMgr;
} // namespace llvm

namespace mlir {
class MLIRContext;
class ModuleOp;

namespace oq3 {

struct OpenQASMTranslationOptions {
  frontend::FrontendOptions frontend;
};

[[nodiscard]] OwningOpRef<ModuleOp>
emitOQ3(const frontend::TypedProgram& program, MLIRContext& context);

[[nodiscard]] OwningOpRef<ModuleOp>
translateOpenQASMToOQ3(llvm::SourceMgr& sourceMgr, MLIRContext& context,
                       const OpenQASMTranslationOptions& options = {});

[[nodiscard]] OwningOpRef<ModuleOp>
translateOpenQASMToOQ3(llvm::StringRef source, MLIRContext& context,
                       const OpenQASMTranslationOptions& options = {});

} // namespace oq3
} // namespace mlir
