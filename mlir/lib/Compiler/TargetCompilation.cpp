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
#include "mlir/Dialect/QCO/Transforms/Mapping/Mapping.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Support/Passes.h"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

namespace mlir {

void populateTargetCompilationPipeline(OpPassManager& pm,
                                       const CompilerTarget& target) {
  pm.addPass(createInlinerPass());
  populateQCOCleanupPipeline(pm);
  pm.addPass(qco::createDecomposeMultiControlled(target));
  populateDefaultQCOOptimizationPipeline(pm);
  pm.addPass(qco::createFuseTwoQubitGates());
  switch (target.connectivityKind()) {
  case CompilerTarget::Connectivity::Kind::Explicit:
    pm.addPass(qco::createMappingPass(target, qco::MappingPassOptions{}));
    break;
  case CompilerTarget::Connectivity::Kind::AllToAll:
    pm.addPass(qco::createPlacementPass(target));
    break;
  }
  populateQCOCleanupPipeline(pm);
  pm.addPass(qco::createTargetNativeSynthesis(target));
  pm.addPass(createCSEPass());
  pm.addPass(qco::createVerifyTargetConformance(target));
}

} // namespace mlir
