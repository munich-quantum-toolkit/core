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

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <memory>
#include <utility>

namespace mlir {
namespace {

class InitializeTargetEnvironmentPass
    : public PassWrapper<InitializeTargetEnvironmentPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InitializeTargetEnvironmentPass)

  explicit InitializeTargetEnvironmentPass(TargetEnvironment environment)
      : environment_(std::move(environment)) {}

protected:
  void runOnOperation() override {
    getAnalysis<TargetEnvironmentAnalysis>().initialize(environment_);
    markAnalysesPreserved<TargetEnvironmentAnalysis>();
  }

private:
  TargetEnvironment environment_;
};

} /* namespace */

void populateTargetCompilationPipeline(OpPassManager& pm,
                                       const TargetEnvironment& environment) {
  pm.addPass(std::make_unique<InitializeTargetEnvironmentPass>(environment));
  const auto& target = environment.target();
  pm.addPass(createInlinerPass());
  populateQCOCleanupPipeline(pm);
  pm.addPass(qco::createDecomposeMultiControlled(target));
  populateDefaultQCOOptimizationPipeline(pm);
  pm.addPass(qco::createFuseTwoQubitGates());
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
