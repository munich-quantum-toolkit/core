/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Compiler/TargetCompilation.h"

#include "mlir/Compiler/Target.h"
#include "mlir/Compiler/TargetEnvironment.h"
#include "mlir/Dialect/QCO/Transforms/Mapping/Mapping.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Support/Passes.h"

#include <mlir/IR/Visitors.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/WalkResult.h>
#include <mlir/Transforms/Passes.h>

#include <memory>
#include <utility>

namespace mlir {
namespace {

class PrepareTargetCompilationPass
    : public PassWrapper<PrepareTargetCompilationPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareTargetCompilationPass)

  explicit PrepareTargetCompilationPass(TargetEnvironment environment)
      : environment_(std::move(environment)) {}

protected:
  void runOnOperation() override {
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
};

} /* namespace */

void populateTargetCompilationPipeline(OpPassManager& pm,
                                       const TargetEnvironment& environment) {
  pm.addPass(std::make_unique<PrepareTargetCompilationPass>(environment));
  const auto& target = environment.target();
  pm.addPass(createInlinerPass());
  pm.addPass(createSymbolDCEPass());
  pm.addPass(createSCCPPass());
  pm.addPass(qco::createUnrollUnsupportedPayloadLoops());
  pm.addPass(createSCCPPass());
  populateQCOCleanupPipeline(pm);
  pm.addPass(qco::createLegalizePayloadControlFlow());
  pm.addPass(qco::createDecomposeMultiControlled(target));
  populateDefaultQCOOptimizationPipeline(pm);
  /// ponytail: CX/CZ-cost fusion can increase square-root iSWAP counts;
  /// enable it for this basis when fusion uses the target entangler's cost.
  if (const auto basis = target.synthesisBasis();
      !basis || basis->entangler != CompilerTarget::GateKind::SQRTISWAP) {
    pm.addPass(qco::createFuseTwoQubitGates());
  }
  switch (target.connectivityKind()) {
  case CompilerTarget::Connectivity::Kind::Explicit:
    pm.addPass(qco::createMappingPass(qco::MappingPassOptions{}));
    break;
  case CompilerTarget::Connectivity::Kind::AllToAll:
    pm.addPass(qco::createPlacementPass(target));
    break;
  }
  populateQCOCleanupPipeline(pm);
  pm.addPass(qco::createTargetNativeSynthesis());
  pm.addPass(createCSEPass());
  pm.addPass(qco::createVerifyTargetConformance());
}

} // namespace mlir
