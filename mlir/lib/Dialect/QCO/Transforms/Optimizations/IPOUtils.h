/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/**
 * @file
 * @brief Internals shared between the stages of the quantum IPO pass.
 *
 * @details
 * These are implementation details of `quantum-ipo` rather than public API,
 * which is why they live next to the sources instead of in the dialect's
 * include directory. The stage entry points are declared here so that the pass
 * driver can call them; nothing outside this directory should need them.
 */

#pragma once

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Support/LLVM.h>

namespace mlir::qco {

/**
 * @brief Create a detached copy of a function under a new name.
 *
 * @details
 * The copy is not inserted into a symbol table; the caller is responsible for
 * that, which is also what makes the name unique should @p newName already be
 * taken.
 *
 * @param funcOp The function to copy.
 * @param newName The name of the copy.
 * @return The detached copy.
 */
[[nodiscard]] func::FuncOp copyFunction(func::FuncOp funcOp, StringRef newName);

/**
 * @brief Erase the functions a stage left without callers.
 *
 * @details
 * Only the functions in @p candidates are considered, which are the callees a
 * stage redirected calls away from and the specializations it created. A
 * private function no stage touched is left alone even when it is unused,
 * because removing it is the user's decision rather than ours.
 *
 * Erasing one function can orphan another, for example when a specialization is
 * itself specialized further, so this repeats until nothing more is removed.
 *
 * @param symbolTable The symbol table to erase from.
 * @param candidates The functions that may have been orphaned. Erased entries
 * are removed from it, so the remaining handles stay valid.
 */
void eraseOrphanedSpecializations(SymbolTable& symbolTable,
                                  SmallVector<func::FuncOp>& candidates);

/**
 * @brief Specialize callees for what is known at their call sites.
 * @param moduleOp The module to transform.
 */
void runContextSensitiveSpecialization(ModuleOp moduleOp);

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
 * Callees this stage leaves without callers are erased.
 *
 * @param moduleOp The module to transform.
 */
void runQuantumFunctionBoundaryCommutation(ModuleOp moduleOp);

} // namespace mlir::qco
