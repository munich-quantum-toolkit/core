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

namespace mlir {
class ModuleOp;
} // namespace mlir

/// Compare verified modules, including QCO linearity, with exact types,
/// attributes and control flow. Ignore locations and module symbol order;
/// permit independent SSA operations and supported QCO/qtensor permutations.
/// Preserve effect order except fresh QC/CBit/QCO allocations, linear quantum
/// disposal, and consecutive QC deallocations or QIR runtime releases of the
/// same kind. In particular, QC gates and unknown calls are not freely
/// reordered. Blocks correspond in region order. Matching is greedy, so failure
/// does not prove semantic inequivalence. Numerical tolerances belong in
/// semantic tests.
[[nodiscard]] bool areModulesEquivalentWithPermutations(mlir::ModuleOp,
                                                        mlir::ModuleOp);
