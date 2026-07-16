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
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QIR/Transforms/Passes.h"
#include "mlir/Dialect/QTensor/Transforms/Passes.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/Passes.h>

using namespace mlir;

static void addSimplificationPasses(OpPassManager& pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

LogicalResult
runWithPassManager(ModuleOp module,
                   const function_ref<void(OpPassManager&)> populatePasses,
                   const StringRef errorMessage) {
  PassManager pm(module.getContext());
  populatePasses(pm);
  if (pm.run(module).failed()) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }
  return success();
}

void registerMQTCompilerPasses() {
  static const auto REGISTERED = [] {
    qco::registerFuseSingleQubitUnitaryRuns();
    qco::registerFuseTwoQubitUnitaryRuns();
    qco::registerHadamardLifting();
    qco::registerMergeSingleQubitRotationGates();
    qco::registerQuantumLoopUnroll();
    PassPipelineRegistration<>("mqt-qco-default",
                               "Run the default MQT QCO optimization pipeline.",
                               populateDefaultQCOOptimizationPipeline);
    return true;
  }();
  static_cast<void>(REGISTERED);
}

void populateDefaultQCOOptimizationPipeline(OpPassManager& pm) {
  pm.addPass(qco::createMergeSingleQubitRotationGates());
}

LogicalResult runPassPipeline(ModuleOp module, const StringRef pipeline,
                              const bool enableTiming,
                              const bool enableStatistics) {
  registerMQTCompilerPasses();
  registerTransformsPasses();
  PassManager pm(module.getContext());
  if (enableTiming) {
    pm.enableTiming();
  }
  if (enableStatistics) {
    pm.enableStatistics();
  }
  if (failed(parsePassPipeline(pipeline, pm))) {
    return module.emitError()
           << "failed to parse pass pipeline '" << pipeline << "'";
  }
  return pm.run(module);
}

void populateQCCleanupPipeline(OpPassManager& pm) {
  addSimplificationPasses(pm);
  pm.addPass(qc::createShrinkQubitRegistersPass());
  pm.addPass(createRemoveDeadValuesPass());
}

void populateQCOCleanupPipeline(OpPassManager& pm) {
  addSimplificationPasses(pm);
  pm.addPass(qtensor::createShrinkQTensorToFitPass());
  pm.addPass(createRemoveDeadValuesPass());
}

void populateQIRCleanupPipeline(OpPassManager& pm, bool useAdaptive) {
  addSimplificationPasses(pm);
  pm.addPass(qir::createQIRCleanupPass());
  pm.addPass(createRemoveDeadValuesPass());
  pm.addPass(qir::createQIRSetAttributesAndMetadata({useAdaptive}));
}

void populateJeffCleanupPipeline(OpPassManager& pm) {
  addSimplificationPasses(pm);
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

[[nodiscard]] LogicalResult runQIRCleanupPipeline(ModuleOp module,
                                                  bool useAdaptive) {
  return runWithPassManager(
      module,
      [&](OpPassManager& pm) { populateQIRCleanupPipeline(pm, useAdaptive); },
      "Failed to run QIR cleanup pipeline.");
}

[[nodiscard]] LogicalResult runJeffCleanupPipeline(ModuleOp module) {
  return runWithPassManager(module, populateJeffCleanupPipeline,
                            "Failed to run Jeff cleanup pipeline.");
}
