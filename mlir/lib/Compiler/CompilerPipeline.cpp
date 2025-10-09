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

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <string>

namespace mlir {

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

  // Enable IR printing options if requested
  if (config_.printIRAfterAll) {
    pm.enableIRPrinting(
        /*shouldPrintBeforePass=*/[](Pass*, Operation*) { return false; },
        /*shouldPrintAfterPass=*/[](Pass*, Operation*) { return true; },
        /*printModuleScope=*/true,
        /*printAfterOnlyOnChange=*/true,
        /*printAfterOnlyOnFailure=*/false);
  } else if (config_.printIRAfterFailure) {
    pm.enableIRPrinting(
        /*shouldPrintBeforePass=*/[](Pass*, Operation*) { return false; },
        /*shouldPrintAfterPass=*/[](Pass*, Operation*) { return false; },
        /*printModuleScope=*/true,
        /*printAfterOnlyOnChange=*/false,
        /*printAfterOnlyOnFailure=*/true);
  }
}

LogicalResult
QuantumCompilerPipeline::runPipeline(ModuleOp module,
                                     CompilationRecord* record) const {
  PassManager pm(module.getContext());

  // Configure PassManager with diagnostic options
  configurePassManager(pm);

  // Record initial state if requested
  if (record != nullptr && config_.recordIntermediates) {
    record->afterQuartzImport = captureIR(module);
  }

  // Stage 1: Initial canonicalization
  addCleanupPasses(pm);
  if (failed(pm.run(module))) {
    return failure();
  }
  if (record != nullptr && config_.recordIntermediates) {
    record->afterInitialCanon = captureIR(module);
  }
  pm.clear();

  // Stage 2: Convert to Flux
  pm.addPass(createQuartzToFlux());
  if (failed(pm.run(module))) {
    return failure();
  }
  if (record != nullptr && config_.recordIntermediates) {
    record->afterFluxConversion = captureIR(module);
  }
  pm.clear();

  // Stage 3: Canonicalize Flux
  addCleanupPasses(pm);
  if (failed(pm.run(module))) {
    return failure();
  }
  if (record != nullptr && config_.recordIntermediates) {
    record->afterFluxCanon = captureIR(module);
  }
  pm.clear();

  // Stage 4: Optimization passes (if enabled)
  if (config_.runOptimization) {
    // TODO: Add optimization passes
    addCleanupPasses(pm);
    if (failed(pm.run(module))) {
      return failure();
    }
    if (record != nullptr && config_.recordIntermediates) {
      record->afterOptimization = captureIR(module);
    }
    pm.clear();

    // Canonicalize after optimization
    addCleanupPasses(pm);
    if (failed(pm.run(module))) {
      return failure();
    }
    if (record != nullptr && config_.recordIntermediates) {
      record->afterOptimizationCanon = captureIR(module);
    }
    pm.clear();
  }

  // Stage 5: Convert back to Quartz
  pm.addPass(createFluxToQuartz());
  if (failed(pm.run(module))) {
    return failure();
  }
  if (record != nullptr && config_.recordIntermediates) {
    record->afterQuartzConversion = captureIR(module);
  }
  pm.clear();

  // Stage 6: Canonicalize Quartz
  addCleanupPasses(pm);
  if (failed(pm.run(module))) {
    return failure();
  }
  if (record != nullptr && config_.recordIntermediates) {
    record->afterQuartzCanon = captureIR(module);
  }
  pm.clear();

  // Stage 7: Optional QIR conversion
  if (config_.convertToQIR) {
    pm.addPass(createQuartzToQIR());
    if (failed(pm.run(module))) {
      return failure();
    }
    if (record != nullptr && config_.recordIntermediates) {
      record->afterQIRConversion = captureIR(module);
    }
    pm.clear();

    // Final canonicalization
    addCleanupPasses(pm);
    if (failed(pm.run(module))) {
      return failure();
    }
    if (record != nullptr && config_.recordIntermediates) {
      record->afterQIRCanon = captureIR(module);
    }
  }

  return success();
}

std::string captureIR(ModuleOp module) {
  module.dump();
  std::string result;
  llvm::raw_string_ostream os(result);
  module.print(os);
  os.flush();
  return result;
}

} // namespace mlir
