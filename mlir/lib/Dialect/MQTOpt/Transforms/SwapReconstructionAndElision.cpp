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
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"

#include <mlir/Dialect/PDL/IR/PDL.h>
#include <mlir/Dialect/PDLInterp/IR/PDLInterp.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LLVM.h>
#include <utility>

namespace mqt::ir::opt {

#define GEN_PASS_DEF_SWAPRECONSTRUCTIONANDELISION
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"
#undef GEN_PASS_DEF_SWAPRECONSTRUCTIONANDELISION
#include "SwapReconstructionAndElision.h.inc"

/**
 * @brief This pattern attempts to remove SWAP gates by re-ordering qubits.
 */
struct SwapReconstructionAndElision final
    : impl::SwapReconstructionAndElisionBase<SwapReconstructionAndElision> {

  void runOnOperation() override {
    // Get the current operation being operated on.
    auto op = getOperation();
    auto* ctx = &getContext();

    // Define the set of patterns to use.
    mlir::RewritePatternSet patterns(ctx);
    populateGeneratedPDLLPatterns(patterns);

    // Apply patterns in an iterative and greedy manner.
    if (mlir::failed(APPLY_PATTERNS_GREEDILY(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }

  void getDependentDialects(::mlir::DialectRegistry& registry) const override {
    registry
        .insert<mlir::pdl::PDLDialect, mlir::pdl_interp::PDLInterpDialect>();
  }
};

} // namespace mqt::ir::opt
