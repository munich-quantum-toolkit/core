/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Compiler/TargetCompilation.h"

#include "mqt/Compiler/Target.h"
#include "mqt/Compiler/TargetEnvironment.h"
#include "mqt/Dialect/QCO/Transforms/Mapping/Mapping.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"
#include "mqt/Dialect/QTensor/Transforms/Passes.h"
#include "mqt/Support/Passes.h"

#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include <memory>
#include <utility>

namespace mlir {
namespace {

class PrepareTargetCompilationPass
    : public PassWrapper<PrepareTargetCompilationPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareTargetCompilationPass)

  explicit PrepareTargetCompilationPass(TargetEnvironment environment,
                                        bool allToAllOnly = false,
                                        MappingOptions mapping = {})
      : environment_(std::move(environment)), allToAllOnly_(allToAllOnly),
        mapping_(mapping) {}

protected:
  void runOnOperation() override {
    if (mapping_.trials == 0) {
      getOperation().emitError("mapping trials must be greater than zero");
      signalPassFailure();
      return;
    }
    if (mapping_.iterations == 0) {
      getOperation().emitError("mapping iterations must be greater than zero");
      signalPassFailure();
      return;
    }
    if (allToAllOnly_ && environment_.target().connectivityKind() !=
                             CompilerTarget::Connectivity::Kind::AllToAll) {
      getOperation().emitError(
          "target synthesis requires all-to-all connectivity; use target "
          "compilation for routing");
      signalPassFailure();
      return;
    }
    getAnalysis<TargetEnvironmentAnalysis>().initialize(environment_);
    auto result = getOperation().walk([](Operation* operation) {
      if (operation->getNumSuccessors() == 0) {
        return WalkResult::advance();
      }
      operation->emitError(
          "target compilation requires structured QCO/SCF input; normalize "
          "CFG branches before compilation");
      return WalkResult::interrupt();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }
    markAnalysesPreserved<TargetEnvironmentAnalysis>();
  }

private:
  TargetEnvironment environment_;
  bool allToAllOnly_;
  MappingOptions mapping_;
};

} /* namespace */

static void populatePostPlacementPipeline(OpPassManager& pm,
                                          const CompilerTarget& target) {
  /// Placement consumes allocations; native synthesis normalizes phases.
  pm.addPass(createCanonicalizerPass(
      GreedyRewriteConfig{}.setMaxIterations(GreedyRewriteConfig::kNoLimit)));
  /// Reuse unchanged classical reads before native synthesis splits their uses.
  pm.addPass(createCSEPass());
  pm.addPass(createRemoveDeadValuesPass());
  if (const auto basis = target.synthesisBasis();
      basis && basis->singleQubit != CompilerTarget::SingleQubitBasis::U) {
    pm.addPass(qco::createFuseSingleQubitUnitaryRuns(basis->singleQubit));
  }
  pm.addPass(qco::createTargetNativeSynthesis());
  pm.addPass(createCSEPass());
  pm.addPass(qco::createVerifyTargetConformance());
}

void populateTargetCompilationPipeline(OpPassManager& pm,
                                       const TargetEnvironment& environment,
                                       const MappingOptions& mapping) {
  pm.addPass(std::make_unique<PrepareTargetCompilationPass>(environment, false,
                                                            mapping));
  const auto& target = environment.target();
  pm.addPass(createInlinerPass());
  pm.addPass(createSymbolDCEPass());
  pm.addPass(createSCCPPass());
  populateQCOCleanupPipeline(pm);
  pm.addPass(qco::createUnrollLoopsForPayload());
  pm.addPass(createSCCPPass());
  /// Unrolling exposes static tensor slots and unreachable callees.
  pm.addPass(createCanonicalizerPass(
      GreedyRewriteConfig{}.setMaxIterations(GreedyRewriteConfig::kNoLimit)));
  pm.addPass(createCSEPass());
  pm.addPass(qtensor::createShrinkQTensorToFitPass());
  pm.addPass(createSymbolDCEPass());
  pm.addPass(qco::createLegalizeControlFlow());
  pm.addPass(qco::createDecomposeMultiControlled(target));
  pm.addPass(qco::createFuseTwoQubitGates(target));
  // Non-U targets fuse directly in their basis after placement, avoiding an
  // intermediate U representation and its symbolic phase correction.
  if (const auto basis = target.synthesisBasis();
      !basis || basis->singleQubit == CompilerTarget::SingleQubitBasis::U) {
    // Retain the U optimizer's treatment of isolated gates on U-based and
    // unrestricted targets.
    populateDefaultQCOOptimizationPipeline(pm);
  }
  switch (target.connectivityKind()) {
  case CompilerTarget::Connectivity::Kind::Explicit: {
    qco::MappingPassOptions mappingOptions;
    if (mapping.trials) {
      mappingOptions.ntrials = *mapping.trials;
    }
    mappingOptions.niterations = mapping.iterations;
    mappingOptions.nlookahead = mapping.lookahead;
    mappingOptions.searchMemoryLimit = mapping.searchMemoryLimit;
    pm.addPass(qco::createMappingPass(mappingOptions));
    break;
  }
  case CompilerTarget::Connectivity::Kind::AllToAll:
    pm.addPass(qco::createPlacementPass(target));
    break;
  }
  populatePostPlacementPipeline(pm, target);
}

void populateTargetSynthesisPipeline(OpPassManager& pm,
                                     const TargetEnvironment& environment,
                                     const MappingOptions& mapping) {
  pm.addPass(std::make_unique<PrepareTargetCompilationPass>(environment, true,
                                                            mapping));
  const auto& target = environment.target();
  pm.addPass(createInlinerPass());
  pm.addPass(createSymbolDCEPass());
  populateQCOCleanupPipeline(pm);
  pm.addPass(qco::createLegalizeControlFlow());
  pm.addPass(qco::createDecomposeMultiControlled(target));
  pm.addPass(qco::createFuseTwoQubitGates(target));
  pm.addPass(qco::createPlacementPass(target));
  populatePostPlacementPipeline(pm, target);
}

} // namespace mlir
