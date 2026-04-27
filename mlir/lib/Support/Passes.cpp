/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Support/Passes.h"

#include "mlir/Dialect/QC/Transforms/Passes.h"
#include "mlir/Dialect/QIR/Transforms/Passes.h"
#include "mlir/Dialect/QTensor/Transforms/Passes.h"

#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/Passes.h>

using namespace mlir;

static void addSimplificationPasses(PassManager& pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

static LogicalResult
runWithPassManager(ModuleOp module,
                   const llvm::function_ref<void(PassManager&)> populatePasses,
                   const llvm::StringRef errorMessage) {
  PassManager pm(module.getContext());
  populatePasses(pm);
  if (pm.run(module).failed()) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }
  return success();
}

void populateQCCleanupPipeline(PassManager& pm) {
  addSimplificationPasses(pm);
  pm.addPass(qc::createShrinkQubitRegistersPass());
  pm.addPass(createRemoveDeadValuesPass());
}

void populateQCOCleanupPipeline(PassManager& pm) {
  addSimplificationPasses(pm);
  pm.addPass(qtensor::createShrinkQTensorToFitPass());
  pm.addPass(createRemoveDeadValuesPass());
}

void populateQIRCleanupPipeline(PassManager& pm) {
  addSimplificationPasses(pm);
  pm.addPass(qir::createQIRCleanupPass());
  pm.addPass(createRemoveDeadValuesPass());
}

[[nodiscard]] LogicalResult runQCCleanupPipeline(ModuleOp module) {
  return runWithPassManager(module, populateQCCleanupPipeline,
                            "Failed to run QC cleanup pipeline.");
}

[[nodiscard]] LogicalResult runQCOCleanupPipeline(ModuleOp module) {
  return runWithPassManager(module, populateQCOCleanupPipeline,
                            "Failed to run QCO cleanup pipeline.");
}

[[nodiscard]] LogicalResult runQIRCleanupPipeline(ModuleOp module) {
  return runWithPassManager(module, populateQIRCleanupPipeline,
                            "Failed to run QIR cleanup pipeline.");
}
