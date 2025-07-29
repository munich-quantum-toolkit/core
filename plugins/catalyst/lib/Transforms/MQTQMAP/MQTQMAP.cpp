/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/Definitions.hpp"
#include "ir/operations/Operation.hpp"
#include "mlir/Dialect/Common/Compat.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"
#include "mqt-core/ir/QuantumComputation.hpp"
#include "sc/Architecture.hpp"
#include "sc/MappingResults.hpp"
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
    auto op = getOperation();
    auto* ctx = &getContext();

    qc::QuantumComputation circuit;

    // === Step 1: Lower MLIR to QuantumComputation ===
    {
      mlir::RewritePatternSet toPatterns(ctx);
      mqt::ir::opt::populateToQuantumComputationPatterns(toPatterns, circuit);
      if (mlir::failed(
              applyPatternsAndFoldGreedily(op, std::move(toPatterns)))) {
        op->emitError("Failed to lower MLIR to QuantumComputation.");
        signalPassFailure();
        return;
      }
    }

    // === Step 2: Run QMAP mapping on extracted circuit ===
    try {
      const CouplingMap cm = {{0, 1}, {1, 0}, {1, 2}, {2, 1}, {2, 3}, {3, 2},
                              {3, 4}, {4, 3}, {4, 5}, {5, 4}, {5, 6}, {6, 5}};
      Architecture::Properties props{};
      props.setSingleQubitErrorRate(0, "x", 0.9);
      for (int i = 1; i <= 5; ++i) {
        props.setSingleQubitErrorRate(i, "x", 0.5);
      }
      props.setSingleQubitErrorRate(6, "x", 0.1);
      for (auto edge : cm) {
        props.setTwoQubitErrorRate(edge.first, edge.second, 0.01, "cx");
      }

      Architecture arch{7, cm, props};

      Configuration config;
      config.method = Method::Heuristic;
      config.layering = Layering::DisjointQubits;
      config.initialLayout = InitialLayout::Identity;
      config.automaticLayerSplits = false;
      config.iterativeBidirectionalRouting = false;
      config.heuristic = Heuristic::FidelityBestLocation;
      config.lookaheadHeuristic = LookaheadHeuristic::None;
      config.earlyTermination = EarlyTermination::None;
      config.earlyTerminationLimit = 4;
      config.debug = true;

      auto mapper = std::make_unique<HeuristicMapper>(circuit, arch);
      mapper->map(config);
      mapper->dumpResult(std::cout);

      circuit = mapper->moveMappedCircuit();
    } catch (const std::exception& e) {
      llvm::errs() << "QMAP mapping failed: " << e.what() << "\n";
      // optional: signalPassFailure(); return;
    }

    // === Step 3: Lower back to MLIR from mapped circuit ===
    {
      mlir::RewritePatternSet fromPatterns(ctx);
      mqt::ir::opt::populateFromQuantumComputationPatterns(fromPatterns,
                                                           circuit);
      if (mlir::failed(
              applyPatternsAndFoldGreedily(op, std::move(fromPatterns)))) {
        op->emitError("Failed to lower QuantumComputation back to MLIR.");
        signalPassFailure();
      }
    }
  }
};

} // namespace mqt::ir::transforms
