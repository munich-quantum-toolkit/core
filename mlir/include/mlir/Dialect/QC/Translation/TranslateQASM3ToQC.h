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
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

namespace mlir::qc {

/**
 * @brief Translate an OpenQASM 3 program to a QC program.
 *
 * @param context MLIRContext to create the module in.
 * @param sourceMgr Source manager containing the OpenQASM3 program.
 */
[[nodiscard]] OwningOpRef<ModuleOp>
translateQASM3ToQC(MLIRContext* context, llvm::SourceMgr& sourceMgr);

} // namespace mlir::qc
