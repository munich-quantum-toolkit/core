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

namespace mlir {

void populateTargetCompilationPipeline(OpPassManager& pm,
                                       const CompilerTarget& target) {
  populateDecomposeMultiControlledPipeline(pm, 2);
  populateDefaultQCOOptimizationPipeline(pm);
  pm.addPass(qco::createFuseTwoQubitGates());
  pm.addPass(qco::createMappingPass(target, qco::MappingPassOptions{}));
  pm.addPass(qco::createTargetNativeSynthesis(target));
  pm.addPass(qco::createVerifyTargetConformance(target));
  populateQCOCleanupPipeline(pm);
}

} // namespace mlir
