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
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace mlir;

static void addSimplificationPasses(PassManager& passManager) {
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
}

static void
runWithPassManager(ModuleOp module,
                   const llvm::function_ref<void(PassManager&)> populatePasses,
                   const llvm::StringRef errorMessage) {
  PassManager pm(module.getContext());
  populatePasses(pm);
  if (pm.run(module).failed()) {
    llvm::errs() << errorMessage << "\n";
  }
}

void populateQCCleanupPipeline(PassManager& passManager) {
  addSimplificationPasses(passManager);
  passManager.addPass(qc::createShrinkQubitRegistersPass());
  passManager.addPass(createRemoveDeadValuesPass());
}

void populateQCOCleanupPipeline(PassManager& passManager) {
  addSimplificationPasses(passManager);
  passManager.addPass(qtensor::createShrinkQTensorToFitPass());
  passManager.addPass(createRemoveDeadValuesPass());
}

void populateQIRCleanupPipeline(PassManager& passManager) {
  addSimplificationPasses(passManager);
  passManager.addPass(qir::createQIRCleanupPass());
  passManager.addPass(createRemoveDeadValuesPass());
}

void runQCCleanupPipeline(ModuleOp module) {
  runWithPassManager(module, populateQCCleanupPipeline,
                     "Failed to run QC cleanup pipeline.");
}

void runQCOCleanupPipeline(ModuleOp module) {
  runWithPassManager(module, populateQCOCleanupPipeline,
                     "Failed to run QCO cleanup pipeline.");
}

void runQIRCleanupPipeline(ModuleOp module) {
  runWithPassManager(module, populateQIRCleanupPipeline,
                     "Failed to run QIR cleanup pipeline.");
}
