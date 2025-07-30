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

#include <Quantum/IR/QuantumDialect.h>
#include <Quantum/IR/QuantumOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>

namespace mqt::ir::transforms {

#define GEN_PASS_DECL_MQTQMAP
#define GEN_PASS_DEF_MQTQMAP
#include "mlir/Transforms/MQTQMAP/MQTQMAP.h.inc"

Architecture extractArchitectureFromModule(const std::string& couplingMapStr,
                                           mlir::ModuleOp module,
                                           mlir::Operation* op) {
  CouplingMap cm;
  unsigned short numQubits = 0;

  // Walk IR to determine max qubit count
  mlir::WalkResult result =
      module.walk([&](mlir::Operation* op) -> mlir::WalkResult {
        if (auto deviceOp =
                mlir::dyn_cast<catalyst::quantum::DeviceInitOp>(op)) {
          auto attr =
              op->getAttr("device_name").dyn_cast_or_null<mlir::StringAttr>();
          if (!attr) {
            op->emitError("Missing 'device_name' attribute on quantum.device.");
            return mlir::WalkResult::interrupt();
          }
          auto deviceName = attr.getValue();
          if (deviceName == "LightningSimulator") {
            numQubits = 32; // TODO: typically limited by system memory
          } else {
            op->emitError("Unsupported quantum.device type: ") << deviceName;
            return mlir::WalkResult::interrupt();
          }
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      });

  if (result.wasInterrupted() && numQubits == 0) {
    op->emitError("Failed to determine number of qubits.");
    return Architecture{}; // default invalid
  }

  // If no coupling map was specified, create all-to-all map
  if (couplingMapStr.empty()) {
    for (unsigned short i = 0; i < numQubits; ++i) {
      for (unsigned short j = 0; j < numQubits; ++j) {
        if (i != j) {
          cm.emplace(i, j);
        }
      }
    }
  } else {
    try {
      const auto j = nlohmann::json::parse(couplingMapStr);
      for (const auto& pair : j) {
        if (pair.size() != 2 || !pair[0].is_number_integer() ||
            !pair[1].is_number_integer()) {
          op->emitError("Invalid format in coupling-map: each entry must be a "
                        "pair of integers.");
          return Architecture{};
        }
        cm.emplace(pair[0].get<unsigned short>(),
                   pair[1].get<unsigned short>());
      }
    } catch (const std::exception& e) {
      op->emitError("Failed to parse coupling-map: ") << e.what();
      op->emitError("Coupling map must be a string of pairs, but got: " +
                    couplingMapStr);
      return Architecture{};
    }
  }

  Architecture::Properties props{};
  return Architecture{numQubits, cm, props};
}

struct MQTQMAP final : impl::MQTQMAPBase<MQTQMAP> {
  using Base = impl::MQTQMAPBase<MQTQMAP>;

  // Default constructor
  MQTQMAP() = default;

  // Constructor that takes options
  MQTQMAP(MQTQMAPOptions options) : Base(std::move(options)) {}

  qc::QuantumComputation circuit;

  void runOnOperation() override {
    auto op = getOperation();
    auto* ctx = &getContext();

    qc::QuantumComputation circuit;

    // Get the parent module
    auto module = op->getParentOfType<mlir::ModuleOp>();
    if (!module) {
      op->emitError("Pass must be run on or within a module.");
      signalPassFailure();
      return;
    }
    // Prepare the architecture from the coupling map and module
    Architecture arch =
        extractArchitectureFromModule(couplingMap.getValue(), op, module);

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
      Configuration config;
      config.heuristic = Heuristic::GateCountMaxDistance;
      config.layering = Layering::IndividualGates;
      config.initialLayout = InitialLayout::Identity;
      config.preMappingOptimizations = false;
      config.postMappingOptimizations = false;
      config.lookaheadHeuristic = LookaheadHeuristic::None;
      config.debug = false;

      auto mapper = std::make_unique<HeuristicMapper>(circuit, arch);
      mapper->map(config);
      mapper->dumpResult(std::cout);

      circuit = mapper->moveMappedCircuit();

      llvm::outs() << "----------------------------------------\n\n";
      llvm::outs() << "-------------------QC-------------------\n";
      std::stringstream ss{};
      circuit.print(ss);
      const auto circuitString = ss.str();
      llvm::outs() << circuitString << "\n";
      llvm::outs() << "----------------------------------------\n\n";

    } catch (const std::exception& e) {
      llvm::errs() << "QMAP mapping failed: " << e.what() << "\n";
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
