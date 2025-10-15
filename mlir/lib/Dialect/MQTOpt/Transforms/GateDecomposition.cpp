/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <utility>

namespace mqt::ir::opt {

#define GEN_PASS_DEF_GATEDECOMPOSITION
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

/**
 * @brief This pass attempts to cancel consecutive self-inverse operations.
 */
struct GateDecomposition final
    : impl::GateDecompositionBase<GateDecomposition> {

  void runOnOperation() override {
    // Get the current operation being operated on.
    auto op = getOperation();
    auto* ctx = &getContext();

    // Define the set of patterns to use.
    mlir::RewritePatternSet patterns(ctx);
    populateGateDecompositionPatterns(patterns);

    // Configure greedy driver
    mlir::GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);

    // Apply patterns in an iterative and greedy manner.
    if (mlir::failed(mlir::applyPatternsGreedily(op, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace mqt::ir::opt
