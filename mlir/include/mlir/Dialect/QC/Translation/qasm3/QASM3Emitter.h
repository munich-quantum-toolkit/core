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
 * @brief Translate an OpenQASM 3 program to a QC program.
 *
 * @details
 * Lexes, parses, and lowers @p sourceMgr's main buffer in a single pass. On any
 * error, a diagnostic is emitted through @p context and a null module is
 * returned.
 *
 * @param sourceMgr Source manager containing the OpenQASM 3 program.
 * @param context The MLIRContext to create the module in.
 * @return A module containing the QC program, or `nullptr` if the translation
 * failed.
 */
[[nodiscard]] OwningOpRef<ModuleOp>
translateQASM3ToQC(llvm::SourceMgr& sourceMgr, MLIRContext* context);

} // namespace qc::detail

} // namespace mlir
