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

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <utility>

namespace mqt::ir::opt {

#define GEN_PASS_DEF_LIFTMEASUREMENTSPASS
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

/**
 * @brief This pass attempts to lift measurements above certain operations.
 */
struct LiftMeasurementsPass final
    : impl::LiftMeasurementsPassBase<LiftMeasurementsPass> {

  /**
   * @brief Registers dialects required by this pass.
   *
   * Inserts the SCF dialect into the provided dialect registry so the pass can
   * create and manipulate SCF operations during transformation.
   *
   * @param registry Dialect registry to populate with required dialects.
   */
  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    registry.insert<mlir::scf::SCFDialect>();
  }

  /**
   * @brief Applies rewrite patterns to the current operation to lift
   * measurements.
   *
   * Populates a RewritePatternSet with patterns that replace basis-state
   * controls with conditional operations, lift measurement ops above control
   * and gate constructs, and remove dead gates; then applies those patterns
   * greedily to the operation. Signals the pass as failed if pattern
   * application does not succeed.
   */
  void runOnOperation() override {
    // Get the current operation being operated on.
    auto op = getOperation();
    auto* ctx = &getContext();

    // Define the set of patterns to use.
    mlir::RewritePatternSet patterns(ctx);
    populateReplaceBasisStateControlsWithIfPatterns(patterns);
    populateLiftMeasurementsAboveControlsPatterns(patterns);
    populateLiftMeasurementsAboveGatesPatterns(patterns);
    populateDeadGateEliminationPatterns(patterns);

    // Apply patterns in an iterative and greedy manner.
    if (mlir::failed(mlir::applyPatternsGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mqt::ir::opt
