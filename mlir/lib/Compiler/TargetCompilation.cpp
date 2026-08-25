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

#include "mlir/Dialect/QCO/Transforms/Mapping/Mapping.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Support/Passes.h"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

namespace mlir {

void populateTargetCompilationPipeline(OpPassManager& pm,
                                       const CompilerTarget& target) {
  populateQCOCleanupPipeline(pm);
  populateDecomposeMultiControlledPipeline(pm, 3);
  populateDefaultQCOOptimizationPipeline(pm);
  pm.addPass(qco::createFuseTwoQubitGates());
  pm.addPass(qco::createMappingPass(target, qco::MappingPassOptions{}));
  populateQCOCleanupPipeline(pm);
  pm.addPass(qco::createTargetNativeSynthesis(target));
  pm.addPass(createCSEPass());
  pm.addPass(createRemoveDeadValuesPass());
  pm.addPass(qco::createVerifyTargetConformance(target));
}

} // namespace mlir
