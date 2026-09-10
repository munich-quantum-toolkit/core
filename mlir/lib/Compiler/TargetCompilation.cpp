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
#include "mqt/Support/Passes.h"

#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/WalkResult.h"
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
  populateQCOCleanupPipeline(pm);
  pm.addPass(qco::createUnrollLoopsForPayload());
  pm.addPass(createSCCPPass());
  populateQCOCleanupPipeline(pm);
  pm.addPass(qco::createLegalizeControlFlow());
  pm.addPass(qco::createDecomposeMultiControlled(target));
  populateDefaultQCOOptimizationPipeline(pm);
  // Generic fusion must not introduce gates that the target cannot synthesize.
  // ponytail: its CX/CZ cost model can increase square-root iSWAP counts;
  // enable that basis when fusion uses the target entangler's cost.
  if (const auto basis = target.synthesisBasis();
      target.nativeOperationsKind() ==
          CompilerTarget::NativeOperations::Kind::Unrestricted ||
      (basis && basis->entangler &&
       basis->entangler != CompilerTarget::GateKind::SQRTISWAP)) {
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
