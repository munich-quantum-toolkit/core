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

#include "Decisions.hpp"

#include <llvm/ADT/SmallVector.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/PatternMatch.h>

namespace mlir::qco {

/**
 * @brief Walks entry in program order and, from the constant-propagation facts
 * already computed in solver, collects the rewrites to perform.
 *
 * Pure: touches no IR. Only top-level controlled gates are considered - a
 * CtrlOp nested in another modifier body is left alone. A program point with no
 * lattice, an uninitialised lattice, or an all-top table yields no decision for
 * that op.
 */
[[nodiscard]] SmallVector<Decision> collectDecisions(func::FuncOp entry,
                                                     DataFlowSolver& solver);

/**
 * @brief Applies decisions to the IR via rewriter.
 *
 * Decisions are independent (distinct ops, no nested-body overlap) and use
 * operand indices rather than values, so batch application is
 * order-insensitive.
 */
void applyDecisions(ArrayRef<Decision> decisions, IRRewriter& rewriter);

} // namespace mlir::qco
