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
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>

#include <memory>

namespace mlir {
class CompilerTarget;
} // namespace mlir

namespace mlir::qco {

#define GEN_PASS_DECL
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc" // IWYU pragma: export

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc" // IWYU pragma: export

/**
 * @brief Replace qubit-tensor arguments by the scalar qubits a callee uses.
 *
 * @details
 * A tensor argument whose elements are taken out and put back at compile-time
 * constant indices is split into one qubit argument and one qubit result per
 * touched element, so untouched elements never cross the call boundary. Call
 * sites are rewritten to extract before and re-insert after the call.
 *
 * @param moduleOp The module to transform.
 */
void runQuantumArgumentPromotion(ModuleOp moduleOp);

/**
 * @brief Turn qubits that a callee allocates and releases itself into
 * arguments.
 *
 * @details
 * The release point becomes a `qco.reset` handed back as an extra result, so
 * the caller owns the allocation and can reuse one qubit across calls.
 * Externally visible functions, declarations and recursive functions are left
 * alone.
 *
 * @param moduleOp The module to transform.
 */
void runAuxiliaryQubitHoisting(ModuleOp moduleOp);

/**
 * @brief Cancel a self-inverse gate in front of a call against the same gate at
 * the start of the callee.
 *
 * @details
 * Both gates are removed and the call is redirected to a specialized copy of
 * the callee, cached per callee and parameter index.
 *
 * @param moduleOp The module to transform.
 * @param symbolTable The symbol table of @p moduleOp. It is mutated: every
 * specialization created here is inserted into it.
 */
void runQuantumFunctionBoundaryCommutation(ModuleOp moduleOp,
                                           SymbolTable& symbolTable);
/**
 * @brief Create target-independent two-qubit gate fusion.
 */
[[nodiscard]] std::unique_ptr<Pass> createFuseTwoQubitGates();

/**
 * @brief Create post-routing synthesis for one immutable compiler target.
 */
[[nodiscard]] std::unique_ptr<Pass>
createTargetNativeSynthesis(const CompilerTarget& target);

/**
 * @brief Create the final mapped-operation conformance verifier.
 */
[[nodiscard]] std::unique_ptr<Pass>
createVerifyTargetConformance(const CompilerTarget& target);

} // namespace mlir::qco
