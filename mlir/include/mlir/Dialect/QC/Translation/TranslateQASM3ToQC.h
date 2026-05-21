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

#include <iosfwd>
#include <string>

namespace mlir::qc {

/**
 * @brief Translate an OpenQASM3 program to the QC dialect.
 *
 * @param context MLIRContext to create the module in.
 * @param filename Path to the input OpenQASM3 file.
 */
[[nodiscard]] OwningOpRef<ModuleOp>
translateQASM3ToQC(MLIRContext* context, const std::string& filename);

/**
 * @brief Translate an OpenQASM3 program to the QC dialect.
 *
 * @param context MLIRContext to create the module in.
 * @param input Stream containing the OpenQASM3 program.
 */
[[nodiscard]] OwningOpRef<ModuleOp> translateQASM3ToQC(MLIRContext* context,
                                                       std::istream& input);

} // namespace mlir::qc
