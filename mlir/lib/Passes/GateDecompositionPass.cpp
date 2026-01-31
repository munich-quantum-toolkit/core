/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Passes/Passes.h"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <utility>

namespace mlir::qco {

#define GEN_PASS_DEF_GATEDECOMPOSITIONPASS
#include "mlir/Passes/Passes.h.inc"

/**
 * @brief This pass attempts to collect as many operations as possible into a
 *        4x4 unitary matrix and then decompose it into 1q rotations and 2q
 *        basis gates.
 */
struct GateDecompositionPass final
    : impl::GateDecompositionPassBase<GateDecompositionPass> {

  GateDecompositionPass() = default;
  GateDecompositionPass(const GateDecompositionPass& other)
      : impl::GateDecompositionPassBase<GateDecompositionPass>{other},
        twoQubitCreationTime{this, "twoQubitCreationTime",
                             "Creation time of basis decomposers"},
        numberOfTwoQubitCreations{
            this, "numberOfTwoQubitCreations",
            "Number of times basis decomposers are created"},
        successfulSingleQubitDecompositions{
            this, "successfulSingleQubitDecompositions",
            "Number of times a single-qubit decomposition was applied"},
        totalSingleQubitDecompositions{this, "totalSingleQubitDecompositions",
                                       "Number of times (only) a single-qubit "
                                       "decomposition was calculated"},
        successfulTwoQubitDecompositions{
            this, "successfulTwoQubitDecompositions",
            "Number of times a two-qubit decomposition was applied"},
        totalTwoQubitDecompositions{
            this, "totalTwoQubitDecompositions",
            "Number of times a two-qubit decomposition was calculated"},
        totalCircuitCollections{this, "totalCircuitCollections",
                                "Number of times a sub-circuit was collected"},
        totalTouchedGates{
            this, "totalTouchedGates",
            "Number of gates that were looked at (in sub-circuit collection)"},
        subCircuitComplexityChange{
            this, "subCircuitComplexityChange",
            "Increase or decrease of complexity in sub-circuit"},
        timeInCircuitCollection{this, "timeInCircuitCollection",
                                "Time spent in circuit collection (µs)"},
        timeInSingleQubitDecomposition{
            this, "timeInSingleQubitDecomposition",
            "Time spent in single-qubit decomposition (µs)"},
        timeInTwoQubitDecomposition{
            this, "timeInTwoQubitDecomposition",
            "Time spent in single-qubit decomposition (µs)"} {}
  GateDecompositionPass(GateDecompositionPass&& other) = delete;
  GateDecompositionPass& operator=(const GateDecompositionPass& other) = delete;
  GateDecompositionPass& operator=(GateDecompositionPass&& other) = delete;

  void runOnOperation() override {
    // Get the current operation being operated on.
    auto op = getOperation();
    auto* ctx = &getContext();

    // Define the set of patterns to use.
    mlir::RewritePatternSet patterns(ctx);
    populateGateDecompositionPatterns(
        patterns, twoQubitCreationTime, numberOfTwoQubitCreations,
        successfulSingleQubitDecompositions, totalSingleQubitDecompositions,
        successfulTwoQubitDecompositions, totalTwoQubitDecompositions,
        totalCircuitCollections, totalTouchedGates, subCircuitComplexityChange,
        timeInCircuitCollection, timeInSingleQubitDecomposition,
        timeInTwoQubitDecomposition);

    // Configure greedy driver
    mlir::GreedyRewriteConfig config;
    // start at top of program to maximize collected sub-circuits
    config.setUseTopDownTraversal(true);
    // only optimize existing operations to avoid unnecessary sub-circuit
    // collections of already decomposed gates
    config.setStrictness(GreedyRewriteStrictness::ExistingOps);

    // Apply patterns in an iterative and greedy manner.
    if (mlir::failed(
            mlir::applyPatternsGreedily(op, std::move(patterns), config))) {
      signalPassFailure();
    }
  }

  Statistic twoQubitCreationTime{this, "twoQubitCreationTime",
                                 "Creation time of basis decomposers"};
  Statistic numberOfTwoQubitCreations{
      this, "numberOfTwoQubitCreations",
      "Number of times basis decomposers are created"};
  Statistic successfulSingleQubitDecompositions{
      this, "successfulSingleQubitDecompositions",
      "Number of times a single-qubit decomposition was applied"};
  Statistic totalSingleQubitDecompositions{
      this, "totalSingleQubitDecompositions",
      "Number of times (only) a single-qubit decomposition was calculated"};
  Statistic successfulTwoQubitDecompositions{
      this, "successfulTwoQubitDecompositions",
      "Number of times a two-qubit decomposition was applied"};
  Statistic totalTwoQubitDecompositions{
      this, "totalTwoQubitDecompositions",
      "Number of times a two-qubit decomposition was calculated"};
  Statistic totalCircuitCollections{
      this, "totalCircuitCollections",
      "Number of times a sub-circuit was collected"};
  Statistic totalTouchedGates{
      this, "totalTouchedGates",
      "Number of gates that were looked at (in sub-circuit collection)"};
  Statistic subCircuitComplexityChange{
      this, "subCircuitComplexityChange",
      "Increase or decrease of complexity in sub-circuit"};
  Statistic timeInCircuitCollection{this, "timeInCircuitCollection",
                                    "Time spent in circuit collection (µs)"};
  Statistic timeInSingleQubitDecomposition{
      this, "timeInSingleQubitDecomposition",
      "Time spent in single-qubit decomposition (µs)"};
  Statistic timeInTwoQubitDecomposition{
      this, "timeInTwoQubitDecomposition",
      "Time spent in single-qubit decomposition (µs)"};
};

} // namespace mlir::qco
