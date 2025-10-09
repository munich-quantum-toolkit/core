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
#include "mlir/Dialect/Flux/IR/FluxDialect.h"
#include "mlir/Dialect/Quartz/IR/QuartzDialect.h"

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/ToolOutputFile.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/FileUtilities.h>

using namespace llvm;
using namespace mlir;

// Command-line options
static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input .mlir file>"),
                                          cl::init("-"));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::opt<bool> convertToQIR("emit-qir",
                                  cl::desc("Convert to QIR at the end"),
                                  cl::init(false));

static cl::opt<bool> enableTiming("mlir-timing",
                                  cl::desc("Enable pass timing statistics"),
                                  cl::init(false));

static cl::opt<bool> enableStatistics("mlir-statistics",
                                      cl::desc("Enable pass statistics"),
                                      cl::init(false));

static cl::opt<bool>
    printIRAfterAllStages("mlir-print-ir-after-all-stages",
                          cl::desc("Print IR after each compiler stage"),
                          cl::init(false));

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
static LogicalResult writeOutput(ModuleOp module, const StringRef filename) {
  std::string errorMessage;
  const auto output = openOutputFile(filename, &errorMessage);
  if (!output) {
    errs() << errorMessage << "\n";
    return failure();
  }

  module.print(output->os());
  output->keep();
  return success();
}

int main(int argc, char** argv) {
  InitLLVM y(argc, argv);

  // Parse command-line options
  cl::ParseCommandLineOptions(argc, argv, "MQT Core Compiler Driver\n");

  // Set up MLIR context with all required dialects
  DialectRegistry registry;
  registry.insert<quartz::QuartzDialect>();
  registry.insert<flux::FluxDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<cf::ControlFlowDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<LLVM::LLVMDialect>();

  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // Load the input .mlir file
  const auto module = loadMLIRFile(inputFilename, &context);
  if (!module) {
    errs() << "Failed to load input file: " << inputFilename << "\n";
    return 1;
  }

  // Configure the compiler pipeline
  QuantumCompilerConfig config;
  config.convertToQIR = convertToQIR;
  config.enableTiming = enableTiming;
  config.enableStatistics = enableStatistics;
  config.printIRAfterAllStages = printIRAfterAllStages;

  // Run the compilation pipeline
  if (const QuantumCompilerPipeline pipeline(config);
      failed(pipeline.runPipeline(module.get()))) {
    errs() << "Compilation pipeline failed\n";
    return 1;
  }

  // Write the output
  if (failed(writeOutput(module.get(), outputFilename))) {
    errs() << "Failed to write output file: " << outputFilename << "\n";
    return 1;
  }

  return 0;
}
