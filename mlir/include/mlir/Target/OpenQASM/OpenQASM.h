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

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>

#include <string>

namespace llvm {
class SourceMgr;
} // namespace llvm

namespace mlir {
class MLIRContext;

namespace oq3 {

/**
 * @brief Options for translating OpenQASM source to typed OQ3 IR.
 */
struct OpenQASMTranslationOptions {
  /// Search paths used to resolve non-standard include files.
  llvm::SmallVector<std::string> includeDirectories;
};

/**
 * @brief Translate OpenQASM 3 or compatible OpenQASM 2.0 to typed OQ3 IR.
 * @param sourceMgr Source manager containing the main OpenQASM buffer.
 * @param context MLIR context in which to construct the module.
 * @param options Translation and include-resolution options.
 * @return A verified module, or a null owning reference after a diagnostic.
 */
OwningOpRef<ModuleOp>
translateOpenQASMToOQ3(llvm::SourceMgr& sourceMgr, MLIRContext& context,
                       const OpenQASMTranslationOptions& options = {});

/**
 * @brief Translate an in-memory OpenQASM program to typed OQ3 IR.
 * @param source OpenQASM source text.
 * @param context MLIR context in which to construct the module.
 * @param options Translation and include-resolution options.
 * @return A verified module, or a null owning reference after a diagnostic.
 */
OwningOpRef<ModuleOp>
translateOpenQASMToOQ3(llvm::StringRef source, MLIRContext& context,
                       const OpenQASMTranslationOptions& options = {});

} // namespace oq3
} // namespace mlir
