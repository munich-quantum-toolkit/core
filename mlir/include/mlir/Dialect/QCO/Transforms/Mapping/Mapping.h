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

#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <mlir/Pass/Pass.h>

#include <memory>

namespace mlir {

class CompilerTarget;

namespace qco {

/// Create a deterministic placement pass for a compiler target.
std::unique_ptr<Pass> createPlacementPass(const CompilerTarget& target);

/**
 * @brief Create a mapping pass instance for a compiler target.
 * @returns a pass object.
 */
std::unique_ptr<Pass> createMappingPass(const CompilerTarget& target,
                                        MappingPassOptions options);

} // namespace qco
} // namespace mlir
