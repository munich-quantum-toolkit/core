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

#include <llvm/Support/SourceMgr.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Support/LLVM.h>

namespace mlir {

// Forward declarations
class MLIRContext;
class ModuleOp;

namespace qc {

/**
 * @brief Translate an OpenQASM 3 program to a QC program.
 *
 * @param sourceMgr Source manager containing the OpenQASM 3 program.
 * @param context The MLIRContext to create the module in.
 * @return A module containing the QC program.
 */
[[nodiscard]] OwningOpRef<ModuleOp>
translateQASM3ToQC(llvm::SourceMgr& sourceMgr, MLIRContext* context);

/**
 * @brief Translate an OpenQASM 3 program to a QC program.
 *
 * @param source String containing the OpenQASM 3 program.
 * @param context The MLIRContext to create the module in.
 * @return A module containing the QC program.
 */
[[nodiscard]] OwningOpRef<ModuleOp> translateQASM3ToQC(StringRef source,
                                                       MLIRContext* context);

} // namespace qc

} // namespace mlir
