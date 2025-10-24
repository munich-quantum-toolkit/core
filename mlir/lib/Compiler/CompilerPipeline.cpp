/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Compiler/CompilerPipeline.h"

#include "mlir/Conversion/FluxToQuartz/FluxToQuartz.h"
#include "mlir/Conversion/QuartzToFlux/QuartzToFlux.h"
#include "mlir/Conversion/QuartzToQIR/QuartzToQIR.h"
#include "mlir/Support/PrettyPrinting.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/Passes.h>
#include <string>

namespace mlir {

namespace {

/**
 * @brief Pretty print IR with ASCII art borders and stage identifier
 *
 * @param module The module to print
 * @param stageName Name of the compilation stage
 * @param stageNumber Current stage number
 * @param totalStages Total number of stages (for progress indication)
 */
static void prettyPrintStage(ModuleOp module, const StringRef stageName,
                             const int stageNumber, const int totalStages) {
  llvm::errs() << "\n";
  printBoxTop();

  // Build the stage header
  const std::string stageHeader = "Stage " + std::to_string(stageNumber) + "/" +
                                  std::to_string(totalStages) + ": " +
                                  stageName.str();
  printBoxLine(stageHeader);

  printBoxMiddle();

  // Capture the IR to a string so we can wrap it in box lines
  std::string irString;
  llvm::raw_string_ostream irStream(irString);
  module.print(irStream);
  irStream.flush();

  // Print the IR with box lines and wrapping
  printBoxText(irString);

  printBoxBottom();
  llvm::errs().flush();
}

} // namespace

void QuantumCompilerPipeline::addCleanupPasses(PassManager& pm) {
  // Always run canonicalization and dead value removal
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createRemoveDeadValuesPass());
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
  PassManager pm(module.getContext());

  // Configure PassManager with diagnostic options
  configurePassManager(pm);

  // Determine total number of stages for progress indication
  auto totalStages =
      8; // Base stages: Initial + Flux + FluxCanon + Optimization +
         // OptimizationCanon + QuartzBack + QuartzCanon
  if (config_.convertToQIR) {
    totalStages += 2; // QIR + QIRCanon
  }
  auto currentStage = 0;

  // Record initial state if requested
  if (record != nullptr && config_.recordIntermediates) {
    record->afterQuartzImport = captureIR(module);
    if (config_.printIRAfterAllStages) {
      prettyPrintStage(module, "Quartz Import (Initial)", ++currentStage,
                       totalStages);
    }
  }

  // Stage 1: Initial canonicalization
  addCleanupPasses(pm);
  if (pm.run(module).failed()) {
    return failure();
  }
  if (record != nullptr && config_.recordIntermediates) {
    record->afterInitialCanon = captureIR(module);
    if (config_.printIRAfterAllStages) {
      prettyPrintStage(module, "Initial Canonicalization", ++currentStage,
                       totalStages);
    }
  }
  pm.clear();

  // Stage 2: Convert to Flux
  pm.addPass(createQuartzToFlux());
  if (failed(pm.run(module))) {
    return failure();
  }
  if (record != nullptr && config_.recordIntermediates) {
    record->afterFluxConversion = captureIR(module);
    if (config_.printIRAfterAllStages) {
      prettyPrintStage(module, "Quartz → Flux Conversion", ++currentStage,
                       totalStages);
    }
  }
  pm.clear();

  // Stage 3: Canonicalize Flux
  addCleanupPasses(pm);
  if (failed(pm.run(module))) {
    return failure();
  }
  if (record != nullptr && config_.recordIntermediates) {
    record->afterFluxCanon = captureIR(module);
    if (config_.printIRAfterAllStages) {
      prettyPrintStage(module, "Flux Canonicalization", ++currentStage,
                       totalStages);
    }
  }
  pm.clear();

  // Stage 4: Optimization passes
  // TODO: Add optimization passes
  addCleanupPasses(pm);
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

  // Stage 5: Canonicalize after optimization
  addCleanupPasses(pm);
  if (failed(pm.run(module))) {
    return failure();
  }
  if (record != nullptr && config_.recordIntermediates) {
    record->afterOptimizationCanon = captureIR(module);
    if (config_.printIRAfterAllStages) {
      prettyPrintStage(module, "Post-Optimization Canonicalization",
                       ++currentStage, totalStages);
    }
  }
  pm.clear();

  // Stage 6: Convert back to Quartz
  pm.addPass(createFluxToQuartz());
  if (failed(pm.run(module))) {
    return failure();
  }
  if (record != nullptr && config_.recordIntermediates) {
    record->afterQuartzConversion = captureIR(module);
    if (config_.printIRAfterAllStages) {
      prettyPrintStage(module, "Flux → Quartz Conversion", ++currentStage,
                       totalStages);
    }
  }
  pm.clear();

  // Stage 7: Canonicalize Quartz
  addCleanupPasses(pm);
  if (failed(pm.run(module))) {
    return failure();
  }
  if (record != nullptr && config_.recordIntermediates) {
    record->afterQuartzCanon = captureIR(module);
    if (config_.printIRAfterAllStages) {
      prettyPrintStage(module, "Final Quartz Canonicalization", ++currentStage,
                       totalStages);
    }
  }
  pm.clear();

  // Stage 8: Optional QIR conversion
  if (config_.convertToQIR) {
    pm.addPass(createQuartzToQIR());
    if (failed(pm.run(module))) {
      return failure();
    }
    if (record != nullptr && config_.recordIntermediates) {
      record->afterQIRConversion = captureIR(module);
      if (config_.printIRAfterAllStages) {
        prettyPrintStage(module, "Quartz → QIR Conversion", ++currentStage,
                         totalStages);
      }
    }
    pm.clear();

    // Final canonicalization
    addCleanupPasses(pm);
    if (failed(pm.run(module))) {
      return failure();
    }
    if (record != nullptr && config_.recordIntermediates) {
      record->afterQIRCanon = captureIR(module);
      if (config_.printIRAfterAllStages) {
        prettyPrintStage(module, "Final QIR Canonicalization", ++currentStage,
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
  os.flush();
  return result;
}

} // namespace mlir
