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

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>
#include <mlir/Support/LogicalResult.h>

#include <cstddef>
#include <utility>

namespace mlir {

/// Verify that no path below `root` exceeds `maximumDepth` region-owning ops.
[[nodiscard]] inline LogicalResult
verifyRegionNestingDepth(Operation* root, size_t maximumDepth) {
  SmallVector<std::pair<Operation*, size_t>> worklist{{root, 0}};
  while (!worklist.empty()) {
    auto [operation, parentDepth] = worklist.pop_back_val();
    const size_t childDepth =
        parentDepth + static_cast<size_t>(operation->getNumRegions() != 0);
    if (childDepth > maximumDepth) {
      return operation->emitError()
             << "operation nesting exceeds the supported maximum of "
             << maximumDepth << " operations with regions";
    }
    for (Region& region : operation->getRegions()) {
      for (Block& block : region) {
        for (Operation& nested : block) {
          worklist.emplace_back(&nested, childDepth);
        }
      }
    }
  }
  return success();
}

} // namespace mlir
