/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Compiler/CompilerPipeline.h"

#include "mlir/Conversion/QCOToQC/QCOToQC.h"
#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Conversion/QCToQIR/QCToQIR.h"
#include "mlir/Support/Passes.h"
#include "mlir/Support/PrettyPrinting.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>

#include <string>

namespace mlir {

/**
 * @brief Print the module IR with an ASCII-art bordered header indicating the pipeline stage.
 *
 * Prints the module to stderr (`llvm::errs()`) with a header of the form
 * "Stage X/Y: <stageName>" and an ASCII-art box around the program output.
 *
 * @param module The MLIR module to print.
 * @param stageName Human-readable name of the compilation stage shown in the header.
 * @param stageNumber The current stage index (1-based) shown as X in "Stage X/Y".
 * @param totalStages The total number of stages shown as Y in "Stage X/Y".
 */
static void prettyPrintStage(ModuleOp module, const llvm::StringRef stageName,
                             const int stageNumber, const int totalStages) {
  llvm::errs() << "\n";

  // Build the stage header
  const std::string stageHeader = "Stage " + std::to_string(stageNumber) + "/" +
                                  std::to_string(totalStages) + ": " +
                                  stageName.str();
  printProgram(module, stageHeader, llvm::errs());
}

/**
 * @brief Apply pipeline configuration flags to a PassManager.
 *
 * Enables timing and pass statistics on the provided PassManager when those
 * options are enabled in the pipeline configuration.
 *
 * @param pm PassManager to configure.
 */
void QuantumCompilerPipeline::configurePassManager(PassManager& pm) const {
  // Enable timing statistics if requested
  if (config_.enableTiming) {
    pm.enableTiming();
  }

  // Enable pass statistics if requested
  if (config_.enableStatistics) {
    pm.enableStatistics();
  }
}

/**
 * @brief Execute the full compilation pipeline over the provided MLIR module.
 *
 * Runs a sequence of transformation and cleanup stages on `module`, optionally
 * recording intermediate IR snapshots into `record` and optionally printing
 * per-stage formatted IR and a final compilation summary to stderr.
 *
 * @param module The MLIR module to compile.
 * @param record If non-null and `config_.recordIntermediates` is enabled,
 *        intermediate IR is captured into the corresponding fields of this
 *        record after each stage; otherwise this parameter is ignored.
 *        Note: enabling `config_.printIRAfterAllStages` requires that
 *        `config_.recordIntermediates` is enabled and `record` is non-null.
 *
 * @returns `success()` if all pipeline stages complete successfully,
 *          `failure()` if any stage fails or if configuration preconditions
 *          (such as printing without recording) are violated.
 */
LogicalResult
QuantumCompilerPipeline::runPipeline(ModuleOp module,
                                     CompilationRecord* record) const {
  // Ensure printIRAfterAllStages implies recordIntermediates
  if (config_.printIRAfterAllStages &&
      (!config_.recordIntermediates || record == nullptr)) {
    llvm::errs() << "printIRAfterAllStages requires recordIntermediates to be "
                    "enabled and the record pointer to be non-null.\n";
    return failure();
  }

  auto runStage = [&](auto&& populatePasses) -> LogicalResult {
    PassManager pm(module.getContext());
    configurePassManager(pm);
    populatePasses(pm);
    return pm.run(module);
  };

  // Determine total number of stages for progress indication
  // 1. QC import
  // 2. QC cleanup
  // 3. QC-to-QCO conversion
  // 4. QCO cleanup
  // 5. Optimization passes
  // 6. QCO cleanup
  // 7. QCO-to-QC conversion
  // 8. QC cleanup
  // 9. QC-to-QIR conversion (optional)
  // 10. QIR cleanup (optional)
  auto totalStages = 8;
  if (config_.convertToQIR) {
    totalStages += 2;
  }
  auto currentStage = 0;

  // Stage 1: QC import
  if (record != nullptr && config_.recordIntermediates) {
    record->afterQCImport = captureIR(module);
    if (config_.printIRAfterAllStages) {
      prettyPrintStage(module, "QC Import", ++currentStage, totalStages);
    }
  }

  // Stage 2: QC cleanup
  if (failed(
          runStage([&](PassManager& pm) { populateQCCleanupPipeline(pm); }))) {
    return failure();
  }
  if (record != nullptr && config_.recordIntermediates) {
    record->afterInitialCanon = captureIR(module);
    if (config_.printIRAfterAllStages) {
      prettyPrintStage(module, "Initial QC Cleanup", ++currentStage,
                       totalStages);
    }
  }
  // Stage 3: QC-to-QCO conversion
  if (failed(runStage([&](PassManager& pm) { pm.addPass(createQCToQCO()); }))) {
    return failure();
  }
  if (record != nullptr && config_.recordIntermediates) {
    record->afterQCOConversion = captureIR(module);
    if (config_.printIRAfterAllStages) {
      prettyPrintStage(module, "QC → QCO Conversion", ++currentStage,
                       totalStages);
    }
  }
  // Stage 4: QCO cleanup
  if (failed(
          runStage([&](PassManager& pm) { populateQCOCleanupPipeline(pm); }))) {
    return failure();
  }
  if (record != nullptr && config_.recordIntermediates) {
    record->afterQCOCanon = captureIR(module);
    if (config_.printIRAfterAllStages) {
      prettyPrintStage(module, "Initial QCO Cleanup", ++currentStage,
                       totalStages);
    }
  }
  // Stage 5: Optimization passes
  // TODO: Add optimization passes
  if (failed(
          runStage([&](PassManager& pm) { populateQCOCleanupPipeline(pm); }))) {
    return failure();
  }
  if (record != nullptr && config_.recordIntermediates) {
    record->afterOptimization = captureIR(module);
    if (config_.printIRAfterAllStages) {
      prettyPrintStage(module, "Optimization Passes", ++currentStage,
                       totalStages);
    }
  }
  // Stage 6: QCO cleanup
  if (failed(
          runStage([&](PassManager& pm) { populateQCOCleanupPipeline(pm); }))) {
    return failure();
  }
  if (record != nullptr && config_.recordIntermediates) {
    record->afterOptimizationCanon = captureIR(module);
    if (config_.printIRAfterAllStages) {
      prettyPrintStage(module, "Final QCO Cleanup", ++currentStage,
                       totalStages);
    }
  }
  // Stage 7: QCO-to-QC conversion
  if (failed(runStage([&](PassManager& pm) { pm.addPass(createQCOToQC()); }))) {
    return failure();
  }
  if (record != nullptr && config_.recordIntermediates) {
    record->afterQCConversion = captureIR(module);
    if (config_.printIRAfterAllStages) {
      prettyPrintStage(module, "QCO → QC Conversion", ++currentStage,
                       totalStages);
    }
  }
  // Stage 8: QC cleanup
  if (failed(
          runStage([&](PassManager& pm) { populateQCCleanupPipeline(pm); }))) {
    return failure();
  }
  if (record != nullptr && config_.recordIntermediates) {
    record->afterQCCanon = captureIR(module);
    if (config_.printIRAfterAllStages) {
      prettyPrintStage(module, "Final QC Cleanup", ++currentStage, totalStages);
    }
  }
  // Stage 9: QC-to-QIR conversion (optional)
  if (config_.convertToQIR) {
    if (failed(
            runStage([&](PassManager& pm) { pm.addPass(createQCToQIR()); }))) {
      return failure();
    }
    if (record != nullptr && config_.recordIntermediates) {
      record->afterQIRConversion = captureIR(module);
      if (config_.printIRAfterAllStages) {
        prettyPrintStage(module, "QC → QIR Conversion", ++currentStage,
                         totalStages);
      }
    }
    // Stage 10: QIR cleanup (optional)
    if (failed(runStage(
            [&](PassManager& pm) { populateQIRCleanupPipeline(pm); }))) {
      return failure();
    }
    if (record != nullptr && config_.recordIntermediates) {
      record->afterQIRCanon = captureIR(module);
      if (config_.printIRAfterAllStages) {
        prettyPrintStage(module, "QIR Cleanup", ++currentStage, totalStages);
      }
    }
  }

  // Print compilation summary
  if (config_.printIRAfterAllStages) {
    llvm::errs() << "\n";
    printBoxTop();

    printBoxLine("✓ Compilation Complete");

    const std::string summaryLine =
        "Successfully processed " + std::to_string(currentStage) + " stages";
    printBoxLine(summaryLine, 1); // Indent by 1 space

    printBoxBottom();
    llvm::errs() << "\n";
    llvm::errs().flush();
  }
  return success();
}

std::string captureIR(ModuleOp module) {
  std::string result;
  llvm::raw_string_ostream os(result);
  module.print(os);
  return result;
}

} // namespace mlir
