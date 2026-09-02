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
 * @brief Internals shared between the interprocedural passes.
 *
 * @details
 * These are implementation details of the interprocedural passes rather than
 * public API, which is why they live next to the sources instead of in the
 * dialect's include directory. Nothing outside this directory should need them.
 */

#pragma once

#include <mlir/Dialect/Func/IR/FuncOps.h>
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

} // namespace mlir::qco
