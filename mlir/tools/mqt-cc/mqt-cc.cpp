/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Conversion/JeffToQCO/JeffToQCO.h"
#include "mlir/Conversion/QCOToJeff/QCOToJeff.h"
#include "mlir/Conversion/QCOToQC/QCOToQC.h"
#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Conversion/QCToQIR/QIRAdaptive/QCToQIRAdaptive.h"
#include "mlir/Conversion/QCToQIR/QIRBase/QCToQIRBase.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Support/Passes.h"

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
#include <mlir/Pass/PassRegistry.h>
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
    llvm::cl::desc("Input format: auto, jeff, mlir, or qasm (default: auto)"),
    llvm::cl::value_desc("format"), llvm::cl::init("auto"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<std::string> outputFormat(
    "emit",
    llvm::cl::desc(
        "Output format: qc-import, mlir, qco, qco-optimized, qir-base, "
        "qir-adaptive, or jeff"),
    llvm::cl::value_desc("format"), llvm::cl::init("mlir"));

namespace {
enum class InputFormat : std::uint8_t { MLIR, QASM, Jeff };
enum class InputDialect : std::uint8_t { QC, QCO };
enum class OutputFormat : std::uint8_t {
  QCImport,
  QC,
  QCO,
  QCOOptimized,
  QIRBase,
  QIRAdaptive,
  Jeff
};

struct ParsedProgram {
  OwningOpRef<ModuleOp> mod;
  InputDialect dialect = InputDialect::QC;
};
} // namespace

/**
 * @brief Parse an input format or infer it from a filename.
 */
[[nodiscard]] static std::optional<InputFormat>
parseInputFormat(const StringRef format, const StringRef filename) {
  if (format == "mlir" || (format == "auto" && filename.ends_with(".mlir"))) {
    return InputFormat::MLIR;
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

/**
 * @brief Check whether a module contains an operation from a dialect.
 */
[[nodiscard]] static bool moduleUsesDialect(ModuleOp mod,
                                            const StringRef dialect) {
  auto found = false;
  mod->walk([&](Operation* operation) {
    found |= operation->getDialect()->getNamespace() == dialect;
  });
  return found;
}

/**
 * @brief Detect the input dialect of a module.
 *
 * @details Defaults to QC if no QCO operation is found.
 */
[[nodiscard]] static InputDialect detectInputDialect(ModuleOp mod) {
  if (moduleUsesDialect(mod, "qco")) {
    return InputDialect::QCO;
  }
  return InputDialect::QC;
}

/**
 * @brief Parse an output format.
 */
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
  if (format == "qco-optimized") {
    return OutputFormat::QCOOptimized;
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

static llvm::cl::opt<bool> enableDecomposeMultiControlled(
    "decompose-multi-controlled",
    llvm::cl::desc(
        "Decompose controlled X/Z gates with at least "
        "--decompose-multi-controlled-min-controls controls (default 2; HP24 "
        "for k>=4, then lower building blocks when min-controls allows)."),
    llvm::cl::init(false));

static llvm::cl::opt<unsigned> decomposeMultiControlledMinControls(
    "decompose-multi-controlled-min-controls",
    llvm::cl::desc(
        "Minimum control count for --decompose-multi-controlled: decompose "
        "controlled X/Z with at least this many controls (default 2; must be "
        "at least 2). Higher values leave smaller controlled gates and HP24 "
        "building blocks undecomposed."),
    llvm::cl::init(2));

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
 * @brief Load a `.jeff` file and convert the program to QCO.
 */
static ParsedProgram loadJeffFile(const StringRef filename,
                                  MLIRContext* const context) {
  if (filename == "-") {
    llvm::errs() << "Reading jeff from standard input is not supported.\n";
    return {};
  }

  std::string errorMessage;
  if (!openInputFile(filename, &errorMessage)) {
    llvm::errs() << "Failed to load file '" << filename << "': '"
                 << errorMessage << "'\n";
    return {};
  }

  auto mod = deserializeFromFile(context, filename);
  if (!mod) {
    llvm::errs() << "Failed to deserialize jeff file '" << filename << "'.\n";
    return {};
  }

  PassManager pm(context);
  pm.addPass(createJeffToQCO());
  if (pm.run(*mod).failed()) {
    llvm::errs() << "Failed to convert jeff input to QCO.\n";
    return {};
  }
  return {.mod = std::move(mod), .dialect = InputDialect::QCO};
}

/**
 * @brief Write serialized `jeff` bytes to an output file or standard output.
 */
static LogicalResult writeJeffOutput(ModuleOp mod, const StringRef filename) {
  std::string errorMessage;
  const auto output = openOutputFile(filename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  const auto serialized = serialize(mod);
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
 * @brief Write a module to an output file.
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

  registerMQTCompilerPasses();
  registerPassManagerCLOptions();
  PassPipelineCLParser passPipeline(
      "passes", "QCO optimization passes to run instead of the default");

  // Parse command-line options; exit on error and print to stderr
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MQT Compiler Collection Driver\n");

  const auto parsedInputFormat = parseInputFormat(inputFormat, inputFilename);
  if (!parsedInputFormat) {
    llvm::errs() << "Could not determine the input format for '"
                 << inputFilename << "'. Use --input-format.\n";
    return 1;
  }
  const auto parsedOutputFormat = parseOutputFormat(outputFormat);
  if (!parsedOutputFormat) {
    llvm::errs() << "Unknown output format '" << outputFormat << "'.\n";
    return 1;
  }

  // Set up MLIR context with all required dialects
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
    program.mod = loadMLIRFile(inputFilename, &context);
    program.dialect = detectInputDialect(*program.mod);
    break;
  case InputFormat::QASM:
    program.mod = loadQASMFile(inputFilename, &context);
    break;
  case InputFormat::Jeff:
    program = loadJeffFile(inputFilename, &context);
    break;
  }
  if (!program.mod) {
    return 1;
  }

  if (*parsedOutputFormat == OutputFormat::QCImport &&
      program.dialect != InputDialect::QC) {
    llvm::errs() << "--emit=qc-import requires QC frontend input.\n";
    return 1;
  }
  if (passPipeline.hasAnyOccurrences() &&
      (*parsedOutputFormat == OutputFormat::QCImport ||
       *parsedOutputFormat == OutputFormat::QCO)) {
    llvm::errs() << "--pass-pipeline requires an output that passes through "
                    "QCO optimization.\n";
    return 1;
  }
  if (enableDecomposeMultiControlled &&
      !isDecomposeMultiControlledConfigValid(
          decomposeMultiControlledMinControls.getValue())) {
    llvm::errs()
        << "decompose-multi-controlled-min-controls must be at least 2 when "
           "--decompose-multi-controlled is enabled.\n";
    return 1;
  }

  const auto runPasses =
      [&](const function_ref<LogicalResult(OpPassManager&)> populate) {
        PassManager pm(&context);
        if (failed(applyPassManagerCLOptions(pm))) {
          return failure();
        }
        if (failed(populate(pm))) {
          return failure();
        }
        return pm.run(*program.mod);
      };

  if (*parsedOutputFormat != OutputFormat::QCImport &&
      program.dialect == InputDialect::QC &&
      failed(runPasses([](OpPassManager& pm) {
        pm.addPass(createQCToQCO());
        return success();
      }))) {
    return 1;
  }

  if (*parsedOutputFormat != OutputFormat::QCImport &&
      *parsedOutputFormat != OutputFormat::QCO) {
    if (failed(runPasses([&](OpPassManager& pm) {
          populateQCOCleanupPipeline(pm);
          if (passPipeline.hasAnyOccurrences()) {
            if (failed(passPipeline.addToPipeline(pm, [](const Twine& message) {
                  llvm::errs() << message << "\n";
                  return failure();
                }))) {
              return failure();
            }
          } else {
            if (enableDecomposeMultiControlled) {
              populateDecomposeMultiControlledPipeline(
                  pm, decomposeMultiControlledMinControls.getValue());
            }
            populateDefaultQCOOptimizationPipeline(pm);
          }
          populateQCOCleanupPipeline(pm);
          return success();
        }))) {
      return 1;
    }
  }

  if (*parsedOutputFormat == OutputFormat::Jeff &&
      failed(runPasses([](OpPassManager& pm) {
        pm.addPass(createQCOToJeff());
        populateJeffCleanupPipeline(pm);
        return success();
      }))) {
    return 1;
  }

  if ((*parsedOutputFormat == OutputFormat::QC ||
       *parsedOutputFormat == OutputFormat::QIRBase ||
       *parsedOutputFormat == OutputFormat::QIRAdaptive) &&
      failed(runPasses([](OpPassManager& pm) {
        pm.addPass(createQCOToQC());
        populateQCCleanupPipeline(pm);
        return success();
      }))) {
    return 1;
  }

  if (*parsedOutputFormat == OutputFormat::QIRBase &&
      failed(runPasses([](OpPassManager& pm) {
        pm.addPass(createQCToQIRBase());
        populateQIRCleanupPipeline(pm, false);
        return success();
      }))) {
    return 1;
  }

  if (*parsedOutputFormat == OutputFormat::QIRAdaptive &&
      failed(runPasses([](OpPassManager& pm) {
        pm.addPass(createQCToQIRAdaptive());
        populateQIRCleanupPipeline(pm, true);
        return success();
      }))) {
    return 1;
  }

  // Write the output
  if (*parsedOutputFormat == OutputFormat::Jeff) {
    if (writeJeffOutput(*program.mod, outputFilename).failed()) {
      return 1;
    }
  } else if (*parsedOutputFormat == OutputFormat::QIRBase ||
             *parsedOutputFormat == OutputFormat::QIRAdaptive) {
    llvm::LLVMContext llvmContext;
    std::unique_ptr<llvm::Module> llvmMod =
        translateModuleToLLVMIR(*program.mod, llvmContext);
    if (!llvmMod) {
      llvm::errs() << "Failed to translate MLIR module to LLVM IR\n";
      return 1;
    }
    if (writeOutput<llvm::Module*>(llvmMod.get(), outputFilename).failed()) {
      return 1;
    }
  } else if (writeOutput<ModuleOp>(program.mod.get(), outputFilename)
                 .failed()) {
    return 1;
  }

  return 0;
}
