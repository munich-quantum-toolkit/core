/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/Common/Compat.h"
#include "mlir/Dialect/MQTRef/Transforms/Passes.h"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <utility>

namespace mqt::ir::ref {

#define GEN_PASS_DEF_CONSTANTFOLDING
#include "mlir/Dialect/MQTRef/Transforms/Passes.h.inc"

/**
 * @brief This pass attempts to perform constant folding for some `mqtref`
 * operations.
 */
struct ConstantFolding final : impl::ConstantFoldingBase<ConstantFolding> {

  void runOnOperation() override {
    // Get the current operation being operated on.
    auto op = getOperation();
    auto* ctx = &getContext();

    // Define the set of patterns to use.
    mlir::RewritePatternSet patterns(ctx);
    populateConstantFoldExtractQubitPatterns(patterns);

    // Apply patterns in an iterative and greedy manner.
    if (mlir::failed(APPLY_PATTERNS_GREEDILY(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mqt::ir::ref
