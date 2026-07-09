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

namespace llvm {
class SourceMgr;
} // namespace llvm

namespace mlir {

// Forward declarations
class MLIRContext;
class ModuleOp;

namespace qc::detail {

/**
 * @brief Import an OpenQASM 3 program into a QC-dialect module.
 *
 * @details
 * Lexes, parses, and lowers @p sourceMgr's main buffer in a single pass. On any
 * error, a diagnostic is emitted through @p context and a null module is
 * returned.
 *
 * @param sourceMgr Source manager whose main buffer holds the program.
 * @param context The MLIRContext to create the module in.
 * @return The imported module, or null on failure.
 */
[[nodiscard]] OwningOpRef<ModuleOp> importQASM3(llvm::SourceMgr& sourceMgr,
                                                MLIRContext* context);

} // namespace qc::detail

} // namespace mlir
