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

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>

#include <iosfwd>
#include <string>

namespace mlir::qc {

/**
 * @brief Import a QASM3 program directly into the QC dialect.
 *
 * @details Bypasses qc::QuantumComputation: the parser produces an AST walked
 * by an InstVisitor that emits QC dialect ops via QCProgramBuilder.
 * Returns nullptr on failure (diagnostics written to llvm::errs()).
 */
[[nodiscard]] OwningOpRef<ModuleOp>
translateQASM3ToQC(MLIRContext* context, const std::string& filename);

[[nodiscard]] OwningOpRef<ModuleOp> translateQASM3ToQC(MLIRContext* context,
                                                       std::istream& input);

} // namespace mlir::qc
