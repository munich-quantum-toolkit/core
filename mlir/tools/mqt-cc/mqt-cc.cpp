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
#include "mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"

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
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <string>
#include <utility>

using namespace mlir;

// Command-line options
static llvm::cl::opt<std::string>
    inputFilename(llvm::cl::Positional,
                  llvm::cl::desc("<input .mlir/.qasm file>"),
                  llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<bool>
    convertToQIR("emit-qir", llvm::cl::desc("Convert to QIR at the end"),
                 llvm::cl::init(false));

static llvm::cl::opt<bool> recordIntermediates(
    "record-intermediates",
    llvm::cl::desc("Record intermediate IR after each compiler stage"),
    llvm::cl::init(false));

static llvm::cl::opt<bool>
    enableTiming("mlir-timing", llvm::cl::desc("Enable pass timing statistics"),
                 llvm::cl::init(false));

static llvm::cl::opt<bool>
    enableStatistics("mlir-statistics",
                     llvm::cl::desc("Enable pass statistics"),
                     llvm::cl::init(false));

static llvm::cl::opt<bool>
    printIRAfterAllStages("mlir-print-ir-after-all-stages",
                          llvm::cl::desc("Print IR after each compiler stage"),
                          llvm::cl::init(false));

static llvm::cl::opt<bool> disableMergeSingleQubitRotationGates(
    "disable-merge-single-qubit-rotation-gates",
    llvm::cl::desc(
        "Disable quaternion-based single-qubit rotation gate merging"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> enableHadamardLifting(
    "hadamard-lifting",
    llvm::cl::desc("Apply Hadamard lifting during optimization"),
    llvm::cl::init(false));

/**
 * @brief Load and parse a .qasm file, dispatching to the chosen import path.
 */
static OwningOpRef<ModuleOp> loadQASMFile(StringRef filename,
                                          MLIRContext* context) {
  return qc::translateQASM3ToQC(context, filename.str());
}

/**
 * @brief Load and parse a .mlir file
 */
static OwningOpRef<ModuleOp> loadMLIRFile(llvm::StringRef filename,
                                          MLIRContext* context) {
  // Set up the input file
  std::string errorMessage;
  auto file = openInputFile(filename, &errorMessage);
  if (!file) {
    llvm::errs() << "Failed to load file '" << filename << "': '"
                 << errorMessage << "'\n";
    return nullptr;
  }

  // Parse the input MLIR
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  return parseSourceFile<ModuleOp>(sourceMgr, context);
}

/**
 * @brief Write the module to the output file
 */
static mlir::LogicalResult writeOutput(ModuleOp module,
                                       llvm::StringRef filename) {
  std::string errorMessage;
  const auto output = openOutputFile(filename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return mlir::failure();
  }

  module.print(output->os());
  output->keep();
  return mlir::success();
}

int main(int argc, char** argv) {
  const llvm::InitLLVM y(argc, argv);

  // Parse command-line options; exit on error and print to stderr
  llvm::cl::ParseCommandLineOptions(argc, argv, "MQT Core Compiler Driver\n");

  // Set up MLIR context with all required dialects
  DialectRegistry registry;
  registry
      .insert<arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect,
              LLVM::LLVMDialect, memref::MemRefDialect, qc::QCDialect,
              qco::QCODialect, qtensor::QTensorDialect, scf::SCFDialect>();

  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // Load the input .mlir file
  OwningOpRef<ModuleOp> module;
  if (inputFilename.getValue().ends_with(".qasm")) {
    module = loadQASMFile(inputFilename, &context);
  } else {
    module = loadMLIRFile(inputFilename, &context);
  }
  if (!module) {
    return 1;
  }

  // Configure the compiler pipeline
  QuantumCompilerConfig config;
  config.convertToQIR = convertToQIR;
  config.recordIntermediates = recordIntermediates;
  config.enableTiming = enableTiming;
  config.enableStatistics = enableStatistics;
  config.printIRAfterAllStages = printIRAfterAllStages;
  config.disableMergeSingleQubitRotationGates =
      disableMergeSingleQubitRotationGates;
  config.enableHadamardLifting = enableHadamardLifting;

  // Run the compilation pipeline
  CompilationRecord record;
  if (const QuantumCompilerPipeline pipeline(config);
      pipeline
          .runPipeline(module.get(), recordIntermediates ? &record : nullptr)
          .failed()) {
    llvm::errs() << "Compilation pipeline failed\n";
    return 1;
  }

  if (recordIntermediates) {
    llvm::outs() << "=== Compilation Record ===\n";
    llvm::outs() << "After QC Import:\n" << record.afterQCImport << "\n";
    llvm::outs() << "After Initial QC Canonicalization:\n"
                 << record.afterInitialCanon << "\n";
    llvm::outs() << "After QC-to-QCO Conversion:\n"
                 << record.afterQCOConversion << "\n";
    llvm::outs() << "After Initial QCO Canonicalization:\n"
                 << record.afterQCOCanon << "\n";
    llvm::outs() << "After Optimization:\n" << record.afterOptimization << "\n";
    llvm::outs() << "After Final QCO Canonicalization:\n"
                 << record.afterOptimizationCanon << "\n";
    llvm::outs() << "After QCO-to-QC Conversion:\n"
                 << record.afterQCConversion << "\n";
    llvm::outs() << "After Final QC Canonicalization:\n"
                 << record.afterQCCanon << "\n";
    llvm::outs() << "After QC-to-QIR Conversion:\n"
                 << record.afterQIRConversion << "\n";
    llvm::outs() << "After QIR Canonicalization:\n"
                 << record.afterQIRCanon << "\n";
  }

  // Write the output
  if (writeOutput(module.get(), outputFilename).failed()) {
    llvm::errs() << "Failed to write output file: " << outputFilename << "\n";
    return 1;
  }

  return 0;
}
