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

/**
 * @brief Appends common simplification passes to a pass manager.
 *
 * Adds the canonicalizer and common-subexpression-elimination passes to the
 * provided PassManager.
 *
 * @param pm Pass manager to populate with simplification passes.
 */
static void addSimplificationPasses(PassManager& pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

/**
 * @brief Build and run a pass pipeline for a module using a caller-provided population callback.
 *
 * The function constructs a PassManager for the given module, invokes the provided
 * callback to populate it with passes, executes the pipeline, and emits the
 * provided error message to llvm::errs() if pipeline execution fails.
 *
 * @param module The MLIR module to run the pass pipeline on.
 * @param populatePasses Callback that receives the PassManager and adds passes to it.
 * @param errorMessage Message printed to llvm::errs() when pipeline execution fails.
 * @return LogicalResult `success()` if the pipeline ran without failure, `failure()` otherwise.
 */
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

/**
 * @brief Populate a PassManager with QC cleanup and simplification passes.
 *
 * Adds common simplification passes, a pass that shrinks qubit registers for
 * the QC dialect, and a dead-value removal pass to the given pass manager.
 *
 * @param pm Pass manager to populate.
 */
void populateQCCleanupPipeline(PassManager& pm) {
  addSimplificationPasses(pm);
  pm.addPass(qc::createShrinkQubitRegistersPass());
  pm.addPass(createRemoveDeadValuesPass());
}

/**
 * @brief Populates a PassManager with a cleanup pipeline for the QCO (qtensor) dialect.
 *
 * The pipeline performs general simplification (canonicalization and CSE),
 * a qtensor-specific shrink-to-fit pass, and a final dead-value removal pass.
 *
 * @param pm PassManager to populate with the QCO cleanup passes.
 */
void populateQCOCleanupPipeline(PassManager& pm) {
  addSimplificationPasses(pm);
  pm.addPass(qtensor::createShrinkQTensorToFitPass());
  pm.addPass(createRemoveDeadValuesPass());
}

/**
 * @brief Populate a PassManager with a QIR-specific cleanup pipeline.
 *
 * Adds canonicalization and CSE simplification passes, the QIR cleanup pass,
 * and a dead-value removal pass to the provided pass manager in that order.
 *
 * @param pm PassManager to populate with cleanup passes.
 */
void populateQIRCleanupPipeline(PassManager& pm) {
  addSimplificationPasses(pm);
  pm.addPass(qir::createQIRCleanupPass());
  pm.addPass(createRemoveDeadValuesPass());
}

/**
 * @brief Run the QC cleanup pass pipeline on a module.
 *
 * Populates a PassManager with QC-specific cleanup passes (including
 * canonicalization, CSE, QC shrink-qubit-registers, and dead-value removal)
 * and runs it on the provided module.
 *
 * @param module The module to run the cleanup pipeline on.
 * @returns LogicalResult `success()` if the pipeline completed successfully, `failure()` otherwise.
 *
 * On failure, an error message is emitted to `llvm::errs()`.
 */
[[nodiscard]] LogicalResult runQCCleanupPipeline(ModuleOp module) {
  return runWithPassManager(module, populateQCCleanupPipeline,
                            "Failed to run QC cleanup pipeline.");
}

/**
 * Runs the QCO (qtensor) cleanup pass pipeline on the given module.
 *
 * @returns `success()` if the pipeline completed successfully, `failure()` otherwise.
 */
[[nodiscard]] LogicalResult runQCOCleanupPipeline(ModuleOp module) {
  return runWithPassManager(module, populateQCOCleanupPipeline,
                            "Failed to run QCO cleanup pipeline.");
}

/**
 * Run the QIR cleanup pass pipeline on the given module.
 *
 * @returns `success()` if the pipeline completed successfully, `failure()` otherwise.
 */
[[nodiscard]] LogicalResult runQIRCleanupPipeline(ModuleOp module) {
  return runWithPassManager(module, populateQIRCleanupPipeline,
                            "Failed to run QIR cleanup pipeline.");
}
