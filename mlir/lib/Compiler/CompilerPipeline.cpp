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
#include "mlir/Passes/Passes.h"
#include "mlir/Support/PrettyPrinting.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/Passes.h>
#include <string>

namespace mlir {

/**
 * @brief Pretty print IR with ASCII art borders and stage identifier
 *
 * @param module The module to print
 * @param stageName Name of the compilation stage
 * @param stageNumber Current stage number
 * @param totalStages Total number of stages (for progress indication)
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

void QuantumCompilerPipeline::addCleanupPasses(PassManager& pm) {
  // Always run canonicalization and dead value removal
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createRemoveDeadValuesPass());
}

void QuantumCompilerPipeline::addOptimizationPasses(PassManager& pm) {
  // Always run all optimization passes for now
  pm.addPass(qco::createGateDecompositionPass());
}

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

  PassManager pm(module.getContext());

  // Configure PassManager with diagnostic options
  configurePassManager(pm);

  // Determine total number of stages for progress indication
  // 1. QC import
  // 2. QC canonicalization
  // 3. QC-to-QCO conversion
  // 4. QCO canonicalization
  // 5. Optimization passes
  // 6. QCO canonicalization
  // 7. QCO-to-QC conversion
  // 8. QC canonicalization
  // 9. QC-to-QIR conversion (optional)
  // 10. QIR canonicalization (optional)
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

  // Stage 2: QC canonicalization
  addCleanupPasses(pm);
  if (pm.run(module).failed()) {
    return failure();
  }
  if (record != nullptr && config_.recordIntermediates) {
    record->afterInitialCanon = captureIR(module);
    if (config_.printIRAfterAllStages) {
      prettyPrintStage(module, "Initial QC Canonicalization", ++currentStage,
                       totalStages);
    }
  }
  pm.clear();

  // Stage 3: QC-to-QCO conversion
  pm.addPass(createQCToQCO());
  if (failed(pm.run(module))) {
    return failure();
  }
  if (record != nullptr && config_.recordIntermediates) {
    record->afterQCOConversion = captureIR(module);
    if (config_.printIRAfterAllStages) {
      prettyPrintStage(module, "QC → QCO Conversion", ++currentStage,
                       totalStages);
    }
  }
  pm.clear();

  // Stage 4: QCO canonicalization
  addCleanupPasses(pm);
  if (failed(pm.run(module))) {
    return failure();
  }
  if (record != nullptr && config_.recordIntermediates) {
    record->afterQCOCanon = captureIR(module);
    if (config_.printIRAfterAllStages) {
      prettyPrintStage(module, "Initial QCO Canonicalization", ++currentStage,
                       totalStages);
    }
  }
  pm.clear();

  // Stage 5: Optimization passes
  addOptimizationPasses(pm);
  if (failed(pm.run(module))) {
    return failure();
  }
  if (record != nullptr && config_.recordIntermediates) {
    record->afterOptimization = captureIR(module);
    if (config_.printIRAfterAllStages) {
      prettyPrintStage(module, "Optimization Passes", ++currentStage,
                       totalStages);
    }
  }
  pm.clear();

  // Stage 6: QCO canonicalization
  addCleanupPasses(pm);
  if (failed(pm.run(module))) {
    return failure();
  }
  if (record != nullptr && config_.recordIntermediates) {
    record->afterOptimizationCanon = captureIR(module);
    if (config_.printIRAfterAllStages) {
      prettyPrintStage(module, "Final QCO Canonicalization", ++currentStage,
                       totalStages);
    }
  }
  pm.clear();

  // Stage 7: QCO-to-QC conversion
  pm.addPass(createQCOToQC());
  if (failed(pm.run(module))) {
    return failure();
  }
  if (record != nullptr && config_.recordIntermediates) {
    record->afterQCConversion = captureIR(module);
    if (config_.printIRAfterAllStages) {
      prettyPrintStage(module, "QCO → QC Conversion", ++currentStage,
                       totalStages);
    }
  }
  pm.clear();

  // Stage 8: QC canonicalization
  addCleanupPasses(pm);
  if (failed(pm.run(module))) {
    return failure();
  }
  if (record != nullptr && config_.recordIntermediates) {
    record->afterQCCanon = captureIR(module);
    if (config_.printIRAfterAllStages) {
      prettyPrintStage(module, "Final QC Canonicalization", ++currentStage,
                       totalStages);
    }
  }
  pm.clear();

  // Stage 9: QC-to-QIR conversion (optional)
  if (config_.convertToQIR) {
    pm.addPass(createQCToQIR());
    if (failed(pm.run(module))) {
      return failure();
    }
    if (record != nullptr && config_.recordIntermediates) {
      record->afterQIRConversion = captureIR(module);
      if (config_.printIRAfterAllStages) {
        prettyPrintStage(module, "QC → QIR Conversion", ++currentStage,
                         totalStages);
      }
    }
    pm.clear();

    // Stage 10: QIR canonicalization (optional)
    addCleanupPasses(pm);
    if (failed(pm.run(module))) {
      return failure();
    }
    if (record != nullptr && config_.recordIntermediates) {
      record->afterQIRCanon = captureIR(module);
      if (config_.printIRAfterAllStages) {
        prettyPrintStage(module, "QIR Canonicalization", ++currentStage,
                         totalStages);
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
