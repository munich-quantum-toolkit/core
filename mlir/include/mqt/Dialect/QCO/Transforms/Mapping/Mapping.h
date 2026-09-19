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

#include "mqt/Compiler/TargetCompilation.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace mlir {

class CompilerTarget;
class ModuleOp;

namespace qco {

/// Scratch state owned by one synchronous compileForTargetWithLayout call.
/// Passes borrow it for that invocation only; do not nest or reuse them.
struct LayoutTracking {
  llvm::ArrayRef<int64_t> requested;
  MappingResult result;
  llvm::SmallVector<size_t> sourceToProgram;
};
/// Record input slots during target preparation.
[[nodiscard]] LogicalResult prepareLayout(ModuleOp moduleOp,
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
