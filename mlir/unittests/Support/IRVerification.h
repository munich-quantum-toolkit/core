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

/// Compare two (quantum) module operations for structural equivalence, allowing
/// some permutations. This function is especially tailored to compare quantum
/// computations.
[[nodiscard]] bool areModulesEquivalentWithPermutations(mlir::ModuleOp,
                                                        mlir::ModuleOp);
