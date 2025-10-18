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

namespace mqt::ir::opt::detail {
// Prefer the setter if present (newer MLIR 21/22+), otherwise fall back to the
// public field (older 20/21).
template <typename T>
auto setTopDown(T& cfg, int)
    -> decltype((void)cfg.setUseTopDownTraversal(true), void()) {
  cfg.setUseTopDownTraversal(true);
}
template <typename T> void setTopDown(T& cfg, long) {
  cfg.useTopDownTraversal = true;
}
} // namespace mqt::ir::opt::detail

namespace mqt::ir::opt {

#define GEN_PASS_DEF_SWAPRECONSTRUCTIONANDELISION
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

/**
 * @brief This pattern attempts to remove SWAP gates by re-ordering qubits.
 */
struct SwapReconstructionAndElision final
    : impl::SwapReconstructionAndElisionBase<SwapReconstructionAndElision> {

  void runOnOperation() override {
    // Get the current operation being operated on.
    auto op = getOperation();
    auto* ctx = &getContext();

    mlir::RewritePatternSet patterns(ctx);
    populateSwapReconstructionAndElisionPatterns(patterns);

    mlir::GreedyRewriteConfig config;
    mqt::ir::opt::detail::setTopDown(config,
                                     0); // tries setter first, else field

    if (mlir::failed(
            mlir::applyPatternsGreedily(op, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace mqt::ir::opt
