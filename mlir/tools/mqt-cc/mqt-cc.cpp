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
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Support/LogicalResult.h>
#include <string>
#include <utility>

using namespace llvm;
using namespace mlir;

namespace {

// Command-line options
const cl::opt<std::string> INPUT_FILENAME(cl::Positional,
                                          cl::desc("<input .mlir file>"),
                                          cl::init("-"));

const cl::opt<std::string> OUTPUT_FILENAME("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

const cl::opt<bool> CONVERT_TO_QIR("emit-qir",
                                   cl::desc("Convert to QIR at the end"),
                                   cl::init(false));

const cl::opt<bool> RECORD_INTERMEDIATES(
    "record-intermediates",
    cl::desc("Record intermediate IR after each compiler stage"),
    cl::init(false));

const cl::opt<bool> ENABLE_TIMING("mlir-timing",
                                  cl::desc("Enable pass timing statistics"),
                                  cl::init(false));

const cl::opt<bool> ENABLE_STATISTICS("mlir-statistics",
                                      cl::desc("Enable pass statistics"),
                                      cl::init(false));

const cl::opt<bool>
    PRINT_IR_AFTER_ALL_STAGES("mlir-print-ir-after-all-stages",
                              cl::desc("Print IR after each compiler stage"),
                              cl::init(false));

} // namespace

/**
 * @brief Load and parse a .mlir file
 */
static OwningOpRef<ModuleOp> loadMLIRFile(StringRef filename,
                                          MLIRContext* context) {
  // Set up the input file
  std::string errorMessage;
  auto file = openInputFile(filename, &errorMessage);
  if (!file) {
    errs() << errorMessage << "\n";
    return nullptr;
  }

  // Parse the input MLIR
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  return parseSourceFile<ModuleOp>(sourceMgr, context);
}

/**
 * @brief Write the module to the output file
 */
static mlir::LogicalResult writeOutput(ModuleOp module, StringRef filename) {
  std::string errorMessage;
  const auto output = openOutputFile(filename, &errorMessage);
  if (!output) {
    errs() << errorMessage << "\n";
    return mlir::failure();
  }

  module.print(output->os());
  output->keep();
  return mlir::success();
}

int main(int argc, char** argv) {
  const InitLLVM y(argc, argv);

  // Parse command-line options
  cl::ParseCommandLineOptions(argc, argv, "MQT Core Compiler Driver\n");

  // Set up MLIR context with all required dialects
  DialectRegistry registry;
  registry.insert<qc::QCDialect>();
  registry.insert<qco::QCODialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<cf::ControlFlowDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<LLVM::LLVMDialect>();
  registry.insert<memref::MemRefDialect>();

  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // Load the input .mlir file
  const auto module = loadMLIRFile(INPUT_FILENAME, &context);
  if (!module) {
    errs() << "Failed to load input file: " << INPUT_FILENAME << "\n";
    return 1;
  }

  // Configure the compiler pipeline
  QuantumCompilerConfig config;
  config.convertToQIR = CONVERT_TO_QIR;
  config.recordIntermediates = RECORD_INTERMEDIATES;
  config.enableTiming = ENABLE_TIMING;
  config.enableStatistics = ENABLE_STATISTICS;
  config.printIRAfterAllStages = PRINT_IR_AFTER_ALL_STAGES;

  // Run the compilation pipeline
  CompilationRecord record;
  if (const QuantumCompilerPipeline pipeline(config);
      pipeline
          .runPipeline(module.get(), RECORD_INTERMEDIATES ? &record : nullptr)
          .failed()) {
    errs() << "Compilation pipeline failed\n";
    return 1;
  }

  if (RECORD_INTERMEDIATES) {
    outs() << "=== Compilation Record ===\n";
    outs() << "After QC Import:\n" << record.afterQCImport << "\n";
    outs() << "After Initial QC Canonicalization:\n"
           << record.afterInitialCanon << "\n";
    outs() << "After QC-to-QCO Conversion:\n"
           << record.afterQCOConversion << "\n";
    outs() << "After Initial QCO Canonicalization:\n"
           << record.afterQCOCanon << "\n";
    outs() << "After Optimization:\n" << record.afterOptimization << "\n";
    outs() << "After Final QCO Canonicalization:\n"
           << record.afterOptimizationCanon << "\n";
    outs() << "After QCO-to-QC Conversion:\n"
           << record.afterQCConversion << "\n";
    outs() << "After Final QC Canonicalization:\n"
           << record.afterQCCanon << "\n";
    outs() << "After QC-to-QIR Conversion:\n"
           << record.afterQIRConversion << "\n";
    outs() << "After QIR Canonicalization:\n" << record.afterQIRCanon << "\n";
  }

  // Write the output
  if (writeOutput(module.get(), OUTPUT_FILENAME).failed()) {
    errs() << "Failed to write output file: " << OUTPUT_FILENAME << "\n";
    return 1;
  }

  return 0;
}
