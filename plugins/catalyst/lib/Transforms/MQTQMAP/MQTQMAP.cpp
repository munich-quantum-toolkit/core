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
#include "mqt-core/ir/QuantumComputation.hpp"
#include "sc/Architecture.hpp"
#include "sc/configuration/Configuration.hpp"
#include "sc/heuristic/HeuristicMapper.hpp"

#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <utility>

namespace mqt::ir::transforms {

#define GEN_PASS_DEF_MQTQMAP
#include "mlir/Transforms/MQTQMAP/MQTQMAP.h.inc"
struct MQTQMAP final : impl::MQTQMAPBase<MQTQMAP> {

  qc::QuantumComputation circuit;

  void runOnOperation() override {
    // Get the current operation being operated on.
    auto op = getOperation();
    auto* ctx = &getContext();

    // Define the set of patterns to use.
    mlir::RewritePatternSet patterns(ctx);
    mqt::ir::opt::populateToQuantumComputationPatterns(patterns, circuit);
    // TODO: perform call QMAP here ...
    mqt::ir::opt::populateFromQuantumComputationPatterns(patterns, circuit);

    // Apply patterns in an iterative and greedy manner.
    if (mlir::failed(APPLY_PATTERNS_GREEDILY(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace mqt::ir::transforms
