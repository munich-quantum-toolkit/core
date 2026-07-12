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
#include "mlir/Conversion/JeffToQCO/JeffToQCO.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"

#include <jeff/IR/JeffDialect.h>
#include <jeff/Translation/Deserialize.hpp>
#include <jeff/Translation/Serialize.hpp>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Bytecode/BytecodeWriter.h>
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
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

using namespace mlir;

// Command-line options
static llvm::cl::opt<std::string>
    inputFilename(llvm::cl::Positional,
                  llvm::cl::desc("<input .jeff/.mlir/.qasm file>"),
                  llvm::cl::init("-"));

static llvm::cl::opt<std::string> inputFormat(
    "input-format",
    llvm::cl::desc(
        "Input format: auto, jeff, mlir, qco, or qasm (default: auto)"),
    llvm::cl::value_desc("format"), llvm::cl::init("auto"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<std::string> outputFormat(
    "emit",
    llvm::cl::desc(
        "Output format: qc-import, mlir, qco, qir-base, qir-adaptive, or "
        "jeff"),
    llvm::cl::value_desc("format"), llvm::cl::init("mlir"));

static llvm::cl::opt<bool>
    convertToQIRBase("emit-qir-base",
                     llvm::cl::desc("Convert to QIR Base Profile at the end"),
                     llvm::cl::init(false));

static llvm::cl::opt<bool> convertToQIRAdaptive(
    "emit-qir-adaptive",
    llvm::cl::desc("Convert to QIR Adaptive Profile at the end"),
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

namespace {
enum class InputFormat : std::uint8_t { MLIR, QCO, QASM, Jeff };
enum class OutputFormat : std::uint8_t {
  QCImport,
  QC,
  QCO,
  QIRBase,
  QIRAdaptive,
  Jeff
};

struct ParsedProgram {
  OwningOpRef<ModuleOp> module;
  PipelineDialect dialect = PipelineDialect::QC;
};
} // namespace

[[nodiscard]] static std::optional<InputFormat>
parseInputFormat(const StringRef format, const StringRef filename) {
  if (format == "mlir" || (format == "auto" && filename.ends_with(".mlir"))) {
    return InputFormat::MLIR;
  }
  if (format == "qco") {
    return InputFormat::QCO;
  }
  if (format == "qasm" || (format == "auto" && filename.ends_with(".qasm"))) {
    return InputFormat::QASM;
  }
  if (format == "jeff" || (format == "auto" && filename.ends_with(".jeff"))) {
    return InputFormat::Jeff;
  }
  if (format == "auto" && filename == "-") {
    return InputFormat::MLIR;
  }
  return std::nullopt;
}

[[nodiscard]] static std::optional<OutputFormat>
parseOutputFormat(const StringRef format) {
  if (format == "qc-import") {
    return OutputFormat::QCImport;
  }
  if (format == "mlir" || format == "qc") {
    return OutputFormat::QC;
  }
  if (format == "qco") {
    return OutputFormat::QCO;
  }
  if (format == "qir-base") {
    return OutputFormat::QIRBase;
  }
  if (format == "qir-adaptive") {
    return OutputFormat::QIRAdaptive;
  }
  if (format == "jeff") {
    return OutputFormat::Jeff;
  }
  return std::nullopt;
}

/**
 * @brief Load and parse a `.qasm` file
 */
static OwningOpRef<ModuleOp> loadQASMFile(const StringRef filename,
                                          MLIRContext* const context) {
  std::string errorMessage;
  auto file = openInputFile(filename, &errorMessage);
  if (!file) {
    llvm::errs() << "Failed to load file '" << filename << "': '"
                 << errorMessage << "'\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  return qc::translateQASM3ToQC(sourceMgr, context);
}

/**
 * @brief Load and parse an `.mlir` file
 */
static OwningOpRef<ModuleOp> loadMLIRFile(const StringRef filename,
                                          MLIRContext* const context) {
  std::string errorMessage;
  auto file = openInputFile(filename, &errorMessage);
  if (!file) {
    llvm::errs() << "Failed to load file '" << filename << "': '"
                 << errorMessage << "'\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  return parseSourceFile<ModuleOp>(sourceMgr, context);
}

/**
 * @brief Load and lower a `.jeff` file to QCO.
 */
static ParsedProgram loadJeffFile(const StringRef filename,
                                  MLIRContext* const context) {
  if (filename == "-") {
    llvm::errs() << "Reading `jeff` from standard input is not supported.\n";
    return {};
  }

  std::string errorMessage;
  if (!openInputFile(filename, &errorMessage)) {
    llvm::errs() << "Failed to load file '" << filename << "': '"
                 << errorMessage << "'\n";
    return {};
  }

  auto module = deserializeFromFile(context, filename);
  if (!module) {
    llvm::errs() << "Failed to deserialize jeff file '" << filename << "'.\n";
    return {};
  }

  PassManager pm(context);
  pm.addPass(createJeffToQCO());
  if (pm.run(*module).failed()) {
    llvm::errs() << "Failed to convert jeff input to QCO.\n";
    return {};
  }
  return {.module = std::move(module), .dialect = PipelineDialect::QCO};
}

/**
 * @brief Write serialized `jeff` bytes to an output file or standard output.
 */
static LogicalResult writeJeffOutput(const ModuleOp module,
                                     const StringRef filename) {
  std::string errorMessage;
  const auto output = openOutputFile(filename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  const auto serialized = serialize(module);
  const auto bytes = serialized.asBytes();
  output->os().write(reinterpret_cast<const char*>(bytes.begin()),
                     bytes.size());
  output->os().flush();
  if (output->os().has_error()) {
    llvm::errs() << "I/O error while writing output file: " << filename << "\n";
    return failure();
  }

  output->keep();
  return success();
}

/**
 * @brief Print all compiler checkpoints that were recorded for this run.
 */
static void printRecordedStage(const StringRef title, const std::string& ir) {
  if (!ir.empty()) {
    llvm::outs() << "After " << title << ":\n" << ir << "\n";
  }
}

/**
 * @brief Write a module to an output file
 */
template <typename ModuleType>
static LogicalResult writeOutput(ModuleType mod, StringRef filename) {
  std::string errorMessage;
  const auto output = openOutputFile(filename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  if constexpr (std::is_same_v<ModuleType, ModuleOp>) {
    if (filename == "-") {
      mod.print(output->os());
    } else {
      writeBytecodeToFile(mod, output->os());
    }
  } else if constexpr (std::is_same_v<ModuleType, llvm::Module*>) {
    if (filename == "-") {
      mod->print(output->os(), nullptr);
    } else {
      llvm::WriteBitcodeToFile(*mod, output->os());
    }
  } else {
    llvm_unreachable("Unsupported module type");
  }

  output->os().flush();
  if (output->os().has_error()) {
    llvm::errs() << "I/O error while writing output file: " << filename << "\n";
    return failure();
  }

  output->keep();
  return success();
}

int main(int argc, char** argv) {
  const llvm::InitLLVM y(argc, argv);

  // Parse command-line options; exit on error and print to stderr
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MQT Compiler Collection Driver\n");

  const auto parsedInputFormat = parseInputFormat(inputFormat, inputFilename);
  if (!parsedInputFormat) {
    llvm::errs() << "Could not determine the input format for '"
                 << inputFilename << "'. Use --input-format.\n";
    return 1;
  }
  auto parsedOutputFormat = parseOutputFormat(outputFormat);
  if (!parsedOutputFormat) {
    llvm::errs() << "Unknown output format '" << outputFormat << "'.\n";
    return 1;
  }
  if (convertToQIRBase && convertToQIRAdaptive) {
    llvm::errs() << "--emit-qir-base and --emit-qir-adaptive are mutually "
                    "exclusive.\n";
    return 1;
  }
  if ((convertToQIRBase || convertToQIRAdaptive) &&
      outputFormat.getNumOccurrences() != 0U) {
    llvm::errs() << "--emit cannot be combined with --emit-qir-base or "
                    "--emit-qir-adaptive.\n";
    return 1;
  }
  if (convertToQIRBase) {
    parsedOutputFormat = OutputFormat::QIRBase;
  } else if (convertToQIRAdaptive) {
    parsedOutputFormat = OutputFormat::QIRAdaptive;
  }

  // Set up MLIR context with all required dialects.
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, cf::ControlFlowDialect,
                  func::FuncDialect, LLVM::LLVMDialect, memref::MemRefDialect,
                  qc::QCDialect, qco::QCODialect, qtensor::QTensorDialect,
                  scf::SCFDialect, jeff::JeffDialect>();
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);

  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  ParsedProgram program;
  switch (*parsedInputFormat) {
  case InputFormat::MLIR:
    program.module = loadMLIRFile(inputFilename, &context);
    break;
  case InputFormat::QCO:
    program.module = loadMLIRFile(inputFilename, &context);
    program.dialect = PipelineDialect::QCO;
    break;
  case InputFormat::QASM:
    program.module = loadQASMFile(inputFilename, &context);
    break;
  case InputFormat::Jeff:
    program = loadJeffFile(inputFilename, &context);
    break;
  }
  if (!program.module) {
    return 1;
  }

  // Configure the compiler pipeline
  QuantumCompilerConfig config;
  config.convertToQIRBase = *parsedOutputFormat == OutputFormat::QIRBase;
  config.convertToQIRAdaptive =
      *parsedOutputFormat == OutputFormat::QIRAdaptive;
  config.recordIntermediates = recordIntermediates;
  config.enableTiming = enableTiming;
  config.enableStatistics = enableStatistics;
  config.printIRAfterAllStages = printIRAfterAllStages;
  config.disableMergeSingleQubitRotationGates =
      disableMergeSingleQubitRotationGates;
  config.enableHadamardLifting = enableHadamardLifting;

  if (*parsedOutputFormat == OutputFormat::QCImport &&
      program.dialect != PipelineDialect::QC) {
    llvm::errs() << "--emit=qc-import requires QC frontend input.\n";
    return 1;
  }

  auto target = PipelineDialect::QC;
  switch (*parsedOutputFormat) {
  case OutputFormat::QCO:
    target = PipelineDialect::QCO;
    break;
  case OutputFormat::Jeff:
    target = PipelineDialect::Jeff;
    break;
  case OutputFormat::QIRBase:
  case OutputFormat::QIRAdaptive:
    target = PipelineDialect::QIR;
    break;
  case OutputFormat::QCImport:
  case OutputFormat::QC:
    break;
  }

  // Run the compilation pipeline unless the requested QC import checkpoint is
  // already represented by the frontend result.
  CompilationRecord record;
  if (*parsedOutputFormat != OutputFormat::QCImport) {
    if (const QuantumCompilerPipeline pipeline(config);
        pipeline
            .run(program.module.get(), program.dialect, target,
                 recordIntermediates ? &record : nullptr)
            .failed()) {
      llvm::errs() << "Compilation pipeline failed\n";
      return 1;
    }
  }

  if (recordIntermediates) {
    llvm::outs() << "=== Compilation Record ===\n";
    printRecordedStage("QC Import", record.afterQCImport);
    printRecordedStage("Initial QC Canonicalization", record.afterInitialCanon);
    printRecordedStage("QC-to-QCO Conversion", record.afterQCOConversion);
    printRecordedStage("Initial QCO Canonicalization", record.afterQCOCanon);
    printRecordedStage("Optimization", record.afterOptimization);
    printRecordedStage("Final QCO Canonicalization",
                       record.afterOptimizationCanon);
    printRecordedStage("QCO-to-QC Conversion", record.afterQCConversion);
    printRecordedStage("Final QC Canonicalization", record.afterQCCanon);
    printRecordedStage("QC-to-QIR Conversion", record.afterQIRConversion);
    printRecordedStage("QIR Canonicalization", record.afterQIRCanon);
    printRecordedStage("QCO-to-Jeff Conversion", record.afterJeffConversion);
    printRecordedStage("Jeff Cleanup", record.afterJeffCanon);
  }

  // Write the output.
  if (*parsedOutputFormat == OutputFormat::Jeff) {
    if (writeJeffOutput(*program.module, outputFilename).failed()) {
      return 1;
    }
  } else if (config.convertToQIRBase || config.convertToQIRAdaptive) {
    llvm::LLVMContext llvmContext;
    std::unique_ptr<llvm::Module> llvmMod =
        translateModuleToLLVMIR(*program.module, llvmContext);
    if (!llvmMod) {
      llvm::errs() << "Failed to translate MLIR module to LLVM IR\n";
      return 1;
    }
    if (writeOutput<llvm::Module*>(llvmMod.get(), outputFilename).failed()) {
      return 1;
    }
  } else if (writeOutput<ModuleOp>(program.module.get(), outputFilename)
                 .failed()) {
    return 1;
  }

  return 0;
}
