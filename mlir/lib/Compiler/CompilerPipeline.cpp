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
#include "mlir/Dialect/QCO/Transforms/Mapping/Architecture.h"
// #include "mlir/Dialect/QCO/Transforms/Mapping/Mapping.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Support/Passes.h"
#include "mlir/Support/PrettyPrinting.h"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>

#include <memory>
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

  const static qco::Architecture::CouplingSet COUPLING{
      {{0, 12},    {12, 0},    {0, 1},     {1, 0},     {1, 13},    {13, 1},
       {1, 2},     {2, 1},     {2, 14},    {14, 2},    {2, 3},     {3, 2},
       {3, 15},    {15, 3},    {3, 4},     {4, 3},     {4, 16},    {16, 4},
       {4, 5},     {5, 4},     {5, 17},    {17, 5},    {5, 6},     {6, 5},
       {6, 18},    {18, 6},    {6, 7},     {7, 6},     {7, 19},    {19, 7},
       {7, 8},     {8, 7},     {8, 20},    {20, 8},    {8, 9},     {9, 8},
       {9, 21},    {21, 9},    {9, 10},    {10, 9},    {10, 22},   {22, 10},
       {10, 11},   {11, 10},   {11, 23},   {23, 11},   {12, 24},   {24, 12},
       {12, 13},   {13, 12},   {13, 25},   {25, 13},   {13, 14},   {14, 13},
       {14, 26},   {26, 14},   {14, 15},   {15, 14},   {15, 27},   {27, 15},
       {15, 16},   {16, 15},   {16, 28},   {28, 16},   {16, 17},   {17, 16},
       {17, 29},   {29, 17},   {17, 18},   {18, 17},   {18, 30},   {30, 18},
       {18, 19},   {19, 18},   {19, 31},   {31, 19},   {19, 20},   {20, 19},
       {20, 32},   {32, 20},   {20, 21},   {21, 20},   {21, 33},   {33, 21},
       {21, 22},   {22, 21},   {22, 34},   {34, 22},   {22, 23},   {23, 22},
       {23, 35},   {35, 23},   {24, 36},   {36, 24},   {24, 25},   {25, 24},
       {25, 37},   {37, 25},   {25, 26},   {26, 25},   {26, 38},   {38, 26},
       {26, 27},   {27, 26},   {27, 39},   {39, 27},   {27, 28},   {28, 27},
       {28, 40},   {40, 28},   {28, 29},   {29, 28},   {29, 41},   {41, 29},
       {29, 30},   {30, 29},   {30, 42},   {42, 30},   {30, 31},   {31, 30},
       {31, 43},   {43, 31},   {31, 32},   {32, 31},   {32, 44},   {44, 32},
       {32, 33},   {33, 32},   {33, 45},   {45, 33},   {33, 34},   {34, 33},
       {34, 46},   {46, 34},   {34, 35},   {35, 34},   {35, 47},   {47, 35},
       {36, 48},   {48, 36},   {36, 37},   {37, 36},   {37, 49},   {49, 37},
       {37, 38},   {38, 37},   {38, 50},   {50, 38},   {38, 39},   {39, 38},
       {39, 51},   {51, 39},   {39, 40},   {40, 39},   {40, 52},   {52, 40},
       {40, 41},   {41, 40},   {41, 53},   {53, 41},   {41, 42},   {42, 41},
       {42, 54},   {54, 42},   {42, 43},   {43, 42},   {43, 55},   {55, 43},
       {43, 44},   {44, 43},   {44, 56},   {56, 44},   {44, 45},   {45, 44},
       {45, 57},   {57, 45},   {45, 46},   {46, 45},   {46, 58},   {58, 46},
       {46, 47},   {47, 46},   {47, 59},   {59, 47},   {48, 60},   {60, 48},
       {48, 49},   {49, 48},   {49, 61},   {61, 49},   {49, 50},   {50, 49},
       {50, 62},   {62, 50},   {50, 51},   {51, 50},   {51, 63},   {63, 51},
       {51, 52},   {52, 51},   {52, 64},   {64, 52},   {52, 53},   {53, 52},
       {53, 65},   {65, 53},   {53, 54},   {54, 53},   {54, 66},   {66, 54},
       {54, 55},   {55, 54},   {55, 67},   {67, 55},   {55, 56},   {56, 55},
       {56, 68},   {68, 56},   {56, 57},   {57, 56},   {57, 69},   {69, 57},
       {57, 58},   {58, 57},   {58, 70},   {70, 58},   {58, 59},   {59, 58},
       {59, 71},   {71, 59},   {60, 72},   {72, 60},   {60, 61},   {61, 60},
       {61, 73},   {73, 61},   {61, 62},   {62, 61},   {62, 74},   {74, 62},
       {62, 63},   {63, 62},   {63, 75},   {75, 63},   {63, 64},   {64, 63},
       {64, 76},   {76, 64},   {64, 65},   {65, 64},   {65, 77},   {77, 65},
       {65, 66},   {66, 65},   {66, 78},   {78, 66},   {66, 67},   {67, 66},
       {67, 79},   {79, 67},   {67, 68},   {68, 67},   {68, 80},   {80, 68},
       {68, 69},   {69, 68},   {69, 81},   {81, 69},   {69, 70},   {70, 69},
       {70, 82},   {82, 70},   {70, 71},   {71, 70},   {71, 83},   {83, 71},
       {72, 84},   {84, 72},   {72, 73},   {73, 72},   {73, 85},   {85, 73},
       {73, 74},   {74, 73},   {74, 86},   {86, 74},   {74, 75},   {75, 74},
       {75, 87},   {87, 75},   {75, 76},   {76, 75},   {76, 88},   {88, 76},
       {76, 77},   {77, 76},   {77, 89},   {89, 77},   {77, 78},   {78, 77},
       {78, 90},   {90, 78},   {78, 79},   {79, 78},   {79, 91},   {91, 79},
       {79, 80},   {80, 79},   {80, 92},   {92, 80},   {80, 81},   {81, 80},
       {81, 93},   {93, 81},   {81, 82},   {82, 81},   {82, 94},   {94, 82},
       {82, 83},   {83, 82},   {83, 95},   {95, 83},   {84, 96},   {96, 84},
       {84, 85},   {85, 84},   {85, 97},   {97, 85},   {85, 86},   {86, 85},
       {86, 98},   {98, 86},   {86, 87},   {87, 86},   {87, 99},   {99, 87},
       {87, 88},   {88, 87},   {88, 100},  {100, 88},  {88, 89},   {89, 88},
       {89, 101},  {101, 89},  {89, 90},   {90, 89},   {90, 102},  {102, 90},
       {90, 91},   {91, 90},   {91, 103},  {103, 91},  {91, 92},   {92, 91},
       {92, 104},  {104, 92},  {92, 93},   {93, 92},   {93, 105},  {105, 93},
       {93, 94},   {94, 93},   {94, 106},  {106, 94},  {94, 95},   {95, 94},
       {95, 107},  {107, 95},  {96, 108},  {108, 96},  {96, 97},   {97, 96},
       {97, 109},  {109, 97},  {97, 98},   {98, 97},   {98, 110},  {110, 98},
       {98, 99},   {99, 98},   {99, 111},  {111, 99},  {99, 100},  {100, 99},
       {100, 112}, {112, 100}, {100, 101}, {101, 100}, {101, 113}, {113, 101},
       {101, 102}, {102, 101}, {102, 114}, {114, 102}, {102, 103}, {103, 102},
       {103, 115}, {115, 103}, {103, 104}, {104, 103}, {104, 116}, {116, 104},
       {104, 105}, {105, 104}, {105, 117}, {117, 105}, {105, 106}, {106, 105},
       {106, 118}, {118, 106}, {106, 107}, {107, 106}, {107, 119}, {119, 107},
       {108, 109}, {109, 108}, {109, 110}, {110, 109}, {110, 111}, {111, 110},
       {111, 112}, {112, 111}, {112, 113}, {113, 112}, {113, 114}, {114, 113},
       {114, 115}, {115, 114}, {115, 116}, {116, 115}, {116, 117}, {117, 116},
       {117, 118}, {118, 117}, {118, 119}, {119, 118}}};
  auto arch = std::make_shared<qco::Architecture>("Nighthawk", 120, COUPLING);

  // const static qco::Architecture::CouplingSet COUPLING{
  //     {0, 3}, {3, 0}, {0, 1}, {1, 0}, {1, 4}, {4, 1}, {1, 2}, {2, 1},
  //     {2, 5}, {5, 2}, {3, 6}, {6, 3}, {3, 4}, {4, 3}, {4, 7}, {7, 4},
  //     {4, 5}, {5, 4}, {5, 8}, {8, 5}, {6, 7}, {7, 6}, {7, 8}, {8, 7}};
  // auto arch = std::make_shared<qco::Architecture>("RigettiNovera", 9, COUPLING);

  // Stage 5: Optimization passes
  // TODO: Add optimization passes
  if (failed(runStage([&](PassManager& pm) {
        // const qco::MappingPassOptions options{.nlookahead = 15,
        //                                       .lambda = 0.5,
        //                                       .niterations = 1,
        //                                       .ntrials = 16,
        //                                       .seed = 42};
        // pm.addPass(qco::createMappingPass(std::move(arch), options));
        pm.addPass(qco::createSwapAbsorb());
        populateQCOCleanupPipeline(pm);
      }))) {
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
