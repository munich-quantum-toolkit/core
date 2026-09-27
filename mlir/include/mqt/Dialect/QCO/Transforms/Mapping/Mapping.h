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
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace mlir {

class CompilerTarget;
class ModuleOp;

namespace qco {

/// State shared by preparation and placement during one compilation.
struct LayoutTracking {
  std::vector<size_t> allocationSizes;
  std::vector<int64_t> initialLayout;
  std::vector<size_t> routingPermutation;
  llvm::SmallVector<size_t> sourceToProgram;
};
/// Record input slots during target preparation.
[[nodiscard]] FailureOr<bool> prepareLayout(ModuleOp moduleOp,
                                            const CompilerTarget& target,
                                            LayoutTracking& tracking);
std::unique_ptr<Pass> createMappingPass(const MappingPassOptions& options,
                                        LayoutTracking* tracking);

/// Create a deterministic placement pass for a compiler target.
std::unique_ptr<Pass> createPlacementPass(const CompilerTarget& target);
std::unique_ptr<Pass> createPlacementPass(const CompilerTarget& target,
                                          LayoutTracking* tracking);

} // namespace qco
} // namespace mlir
