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
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>

namespace mlir::qco {

#define GEN_PASS_DECL
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc" // IWYU pragma: export

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc" // IWYU pragma: export

void runQuantumArgumentPromotion(ModuleOp module);
void runAuxiliaryQubitHoisting(ModuleOp module);
void runQuantumFunctionBoundaryCommutation(ModuleOp module,
                                           SymbolTable& symbolTable);

} // namespace mlir::qco
