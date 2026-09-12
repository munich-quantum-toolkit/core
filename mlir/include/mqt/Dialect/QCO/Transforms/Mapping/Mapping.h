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

#include "mqt/Dialect/QCO/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <memory>

namespace mlir {

class CompilerTarget;
struct MappingResult;

namespace qco {

/// Transient state shared only by passes in one layout-tracking pipeline.
struct LayoutTracking;
std::shared_ptr<LayoutTracking>
createLayoutTracking(MappingResult& result,
                     llvm::ArrayRef<int64_t> initialLayout);
std::unique_ptr<Pass>
createLayoutPreparationPass(std::shared_ptr<LayoutTracking> tracking);
std::unique_ptr<Pass>
createLayoutResultPass(std::shared_ptr<LayoutTracking> tracking);
std::unique_ptr<Pass>
createMappingPass(const MappingPassOptions& options,
                  std::shared_ptr<LayoutTracking> tracking);

/// Create a deterministic placement pass for a compiler target.
std::unique_ptr<Pass> createPlacementPass(const CompilerTarget& target);
std::unique_ptr<Pass>
createPlacementPass(const CompilerTarget& target,
                    std::shared_ptr<LayoutTracking> tracking);

} // namespace qco
} // namespace mlir
