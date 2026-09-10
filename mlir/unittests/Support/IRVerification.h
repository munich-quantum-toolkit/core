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

/// Compare verified modules in the same context with exact operation, block,
/// operand, attribute, and result order. Ignore locations. Forward SSA
/// references across blocks are supported; numerical tolerances belong in
/// semantic tests.
[[nodiscard]] bool areModulesStructurallyEquivalent(mlir::ModuleOp,
                                                    mlir::ModuleOp);

/// Compare verified modules, including QCO linearity, with exact types,
/// attributes and control flow. Ignore locations and module symbol order;
/// permit independent SSA operations and supported QCO/qtensor permutations.
/// Preserve effect order except fresh QC/CBit/QCO allocations, linear quantum
/// disposal, and consecutive QC deallocations or QIR runtime releases of the
/// same kind. In particular, QC gates and unknown calls are not freely
/// reordered. Region yields follow the mapped parent result positions. Blocks
/// correspond in region order. Permutation matching requires cross-block
/// definitions before their uses in region order; strict structural comparison
/// does not. Matching is greedy, so failure does not prove semantic
/// inequivalence.
[[nodiscard]] bool areModulesEquivalentWithPermutations(mlir::ModuleOp,
                                                        mlir::ModuleOp);
