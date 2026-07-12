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

#include "mlir/Conversion/QCOToJeff/QCOToJeff.h"
#include "mlir/Conversion/QCOToQC/QCOToQC.h"
#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Conversion/QCToQIR/QIRAdaptive/QCToQIRAdaptive.h"
#include "mlir/Conversion/QCToQIR/QIRBase/QCToQIRBase.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Support/Passes.h"
#include "mlir/Support/PrettyPrinting.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>

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
static void prettyPrintStage(ModuleOp module, const StringRef stageName,
                             const int stageNumber, const int totalStages) {
  llvm::errs() << "\n";

  // Build the stage header
  const std::string stageHeader = "Stage " + std::to_string(stageNumber) + "/" +
                                  std::to_string(totalStages) + ": " +
                                  stageName.str();
  printProgram(module, stageHeader, llvm::errs());
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

namespace {
using PopulatePasses = void (*)(PassManager&, const QuantumCompilerConfig&);

/// A single compilation stage and its optional IR snapshot.
///
/// A stage with a null @c populatePasses function only records or prints the
/// initial import checkpoint. Function pointers avoid the heap allocations and
/// type erasure associated with `std::function` in this hot orchestration path.
struct Stage {
  StringRef name;
  PopulatePasses populatePasses;
  std::string CompilationRecord::* field;
};
} // namespace

static void
populateInitialQCCleanup(PassManager& pm,
                         [[maybe_unused]] const QuantumCompilerConfig& config) {
  populateQCCleanupPipeline(pm);
}

static void
populateQCToQCO(PassManager& pm,
                [[maybe_unused]] const QuantumCompilerConfig& config) {
  pm.addPass(createQCToQCO());
}

static void populateInitialQCOCleanup(
    PassManager& pm, [[maybe_unused]] const QuantumCompilerConfig& config) {
  populateQCOCleanupPipeline(pm);
}

static void populateOptimizations(PassManager& pm,
                                  const QuantumCompilerConfig& config) {
  if (!config.disableMergeSingleQubitRotationGates) {
    pm.addPass(qco::createMergeSingleQubitRotationGates());
  }
  if (config.enableHadamardLifting) {
    pm.addPass(qco::createHadamardLifting());
  }
}

static void
populateQCOToJeff(PassManager& pm,
                  [[maybe_unused]] const QuantumCompilerConfig& config) {
  pm.addPass(createQCOToJeff());
}

static void
populateJeffCleanup(PassManager& pm,
                    [[maybe_unused]] const QuantumCompilerConfig& config) {
  populateJeffCleanupPipeline(pm);
}

static void
populateQCOToQC(PassManager& pm,
                [[maybe_unused]] const QuantumCompilerConfig& config) {
  pm.addPass(createQCOToQC());
}

static void
populateFinalQCCleanup(PassManager& pm,
                       [[maybe_unused]] const QuantumCompilerConfig& config) {
  populateQCCleanupPipeline(pm);
}

static void populateQCToQIR(PassManager& pm,
                            const QuantumCompilerConfig& config) {
  if (config.convertToQIRAdaptive) {
    pm.addPass(createQCToQIRAdaptive());
    return;
  }
  pm.addPass(createQCToQIRBase());
}

static void populateQIRCleanup(PassManager& pm,
                               const QuantumCompilerConfig& config) {
  populateQIRCleanupPipeline(pm, config.convertToQIRAdaptive);
}

LogicalResult QuantumCompilerPipeline::run(ModuleOp module,
                                           const PipelineDialect from,
                                           const PipelineDialect to,
                                           CompilationRecord* record) const {
  if (config_.convertToQIRBase && config_.convertToQIRAdaptive) {
    llvm::errs()
        << "convertToQIRBase and convertToQIRAdaptive are mutually "
           "exclusive; only one QIR profile can be targeted at a time.\n";
    return failure();
  }

  if (from != PipelineDialect::QC && from != PipelineDialect::QCO) {
    llvm::errs() << "The compiler pipeline can only be entered at the QC or "
                    "QCO dialect.\n";
    return failure();
  }
  if (to != PipelineDialect::QC && to != PipelineDialect::QCO &&
      to != PipelineDialect::QIR && to != PipelineDialect::Jeff) {
    llvm::errs() << "The compiler pipeline received an unknown target "
                    "dialect.\n";
    return failure();
  }

  const auto hasQIRProfile =
      config_.convertToQIRBase || config_.convertToQIRAdaptive;
  if (to == PipelineDialect::QIR && !hasQIRProfile) {
    llvm::errs() << "Lowering to QIR requires selecting either the Base or "
                    "Adaptive QIR profile.\n";
    return failure();
  }
  if (to != PipelineDialect::QIR && hasQIRProfile) {
    llvm::errs() << "A QIR profile can only be selected when lowering to "
                    "QIR.\n";
    return failure();
  }

  // Assemble the stages required to lower from `from` to `to`. Every run passes
  // through the optimized-QCO checkpoint; the entry and exit legs are appended
  // conditionally.
  llvm::SmallVector<Stage, 10> stages;
  if (from == PipelineDialect::QC) {
    stages.push_back({.name = "QC Import",
                      .populatePasses = nullptr,
                      .field = &CompilationRecord::afterQCImport});
    stages.push_back({.name = "Initial QC Cleanup",
                      .populatePasses = &populateInitialQCCleanup,
                      .field = &CompilationRecord::afterInitialCanon});
    stages.push_back({.name = "QC → QCO Conversion",
                      .populatePasses = &populateQCToQCO,
                      .field = &CompilationRecord::afterQCOConversion});
  }
  stages.push_back({.name = "Initial QCO Cleanup",
                    .populatePasses = &populateInitialQCOCleanup,
                    .field = &CompilationRecord::afterQCOCanon});
  stages.push_back({.name = "Optimization Passes",
                    .populatePasses = &populateOptimizations,
                    .field = &CompilationRecord::afterOptimization});
  stages.push_back({.name = "Final QCO Cleanup",
                    .populatePasses = &populateInitialQCOCleanup,
                    .field = &CompilationRecord::afterOptimizationCanon});

  if (to == PipelineDialect::Jeff) {
    stages.push_back({.name = "QCO → Jeff Conversion",
                      .populatePasses = &populateQCOToJeff,
                      .field = &CompilationRecord::afterJeffConversion});
    stages.push_back({.name = "Jeff Cleanup",
                      .populatePasses = &populateJeffCleanup,
                      .field = &CompilationRecord::afterJeffCanon});
  } else if (to != PipelineDialect::QCO) {
    stages.push_back({.name = "QCO → QC Conversion",
                      .populatePasses = &populateQCOToQC,
                      .field = &CompilationRecord::afterQCConversion});
    stages.push_back({.name = "Final QC Cleanup",
                      .populatePasses = &populateFinalQCCleanup,
                      .field = &CompilationRecord::afterQCCanon});
    if (to == PipelineDialect::QIR) {
      stages.push_back({.name = "QC → QIR Conversion",
                        .populatePasses = &populateQCToQIR,
                        .field = &CompilationRecord::afterQIRConversion});
      stages.push_back({.name = "QIR Cleanup",
                        .populatePasses = &populateQIRCleanup,
                        .field = &CompilationRecord::afterQIRCanon});
    }
  }

  auto runStage = [&](const PopulatePasses populatePasses) {
    PassManager pm(module.getContext());
    configurePassManager(pm);
    populatePasses(pm, config_);
    return pm.run(module);
  };

  const auto totalStages = static_cast<int>(stages.size());
  auto currentStage = 0;
  for (const auto& stage : stages) {
    if (stage.populatePasses != nullptr &&
        failed(runStage(stage.populatePasses))) {
      return failure();
    }
    if (record != nullptr && config_.recordIntermediates) {
      record->*(stage.field) = captureIR(module);
    }
    if (config_.printIRAfterAllStages) {
      prettyPrintStage(module, stage.name, ++currentStage, totalStages);
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

LogicalResult
QuantumCompilerPipeline::runPipeline(ModuleOp module,
                                     CompilationRecord* record) const {
  const auto convertToQIR =
      config_.convertToQIRAdaptive || config_.convertToQIRBase;
  return run(module, PipelineDialect::QC,
             convertToQIR ? PipelineDialect::QIR : PipelineDialect::QC, record);
}

std::string captureIR(ModuleOp module) {
  std::string result;
  llvm::raw_string_ostream os(result);
  module.print(os);
  return result;
}

} // namespace mlir
