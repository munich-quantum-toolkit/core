/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Compiler/QDMIAdapter.h"
#include "mlir/Compiler/TargetCompilation.h"
#include "mlir/Conversion/JeffToQCO/JeffToQCO.h"
#include "mlir/Conversion/QCOToJeff/QCOToJeff.h"
#include "mlir/Conversion/QCOToQC/QCOToQC.h"
#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Conversion/QCToQIR/QIRAdaptive/QCToQIRAdaptive.h"
#include "mlir/Conversion/QCToQIR/QIRBase/QCToQIRBase.h"
#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/MQT/Transforms/Passes.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h"
#include "mlir/Dialect/QC/Translation/TranslateQCToOpenQASM3.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/QCOUtils.h"
#include "mlir/Dialect/QIR/Utils/QIRUtils.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Support/Passes.h"

#include <jeff/IR/JeffDialect.h>
#include <jeff/Translation/Deserialize.hpp>
#include <jeff/Translation/Serialize.hpp>
#include <llvm/ADT/Twine.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Bytecode/BytecodeWriter.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
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
#include <cstdlib>
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
    outputFilename("o",
                   llvm::cl::desc("Output filename (for QIR, - and .ll write "
                                  "textual LLVM IR; .bc and other names write "
                                  "LLVM bitcode)"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<std::string> outputFormat(
    "emit",
    llvm::cl::desc(
        "Output format: qc-import, mlir, qco, qco-optimized, qir-base, "
        "qir-adaptive, openqasm3, or jeff"),
    llvm::cl::value_desc("format"), llvm::cl::init("mlir"));

static llvm::cl::opt<bool>
    qdmiListDevices("qdmi-list-devices",
                    llvm::cl::desc("List configured QDMI device IDs and exit"),
                    llvm::cl::init(false));

static llvm::cl::opt<std::string> qdmiDevice(
    "qdmi-device",
    llvm::cl::desc("Compile for the QDMI device with this stable ID"),
    llvm::cl::value_desc("id"), llvm::cl::init(""));

static llvm::cl::opt<std::string> qdmiConfig(
    "qdmi-config",
    llvm::cl::desc("Use an explicit QDMI registry configuration file"),
    llvm::cl::value_desc("registry.json"), llvm::cl::init(""));

namespace {
enum class InputFormat : std::uint8_t { MLIR, QASM, Jeff };
enum class InputDialect : std::uint8_t { QC, QCO };
enum class OutputFormat : std::uint8_t {
  QCImport,
  QC,
  QCO,
  QCOOptimized,
  OpenQASM3,
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
  if (format == "openqasm3") {
    return OutputFormat::OpenQASM3;
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
        "Decompose controlled X/Z/phase/SWAP gates and qco.rccx that act on at "
        "least --decompose-multi-controlled-min-qubits qubits (default 3)."),
    llvm::cl::init(false));

static llvm::cl::opt<unsigned> decomposeMultiControlledMinQubits(
    "decompose-multi-controlled-min-qubits",
    llvm::cl::desc(
        "Minimum qubit count for --decompose-multi-controlled: decompose "
        "controlled X/Z/phase/SWAP gates and qco.rccx that act on at least "
        "this many qubits (default 3; must be at least 3). Higher values leave "
        "narrower gates undecomposed."),
    llvm::cl::init(3));

/**
 * @brief Report a violated QDMI command-line constraint.
 */
[[nodiscard]] static LogicalResult reportQDMIErrorIf(const bool condition,
                                                     const Twine& message) {
  if (!condition) {
    return success();
  }
  llvm::errs() << message << "\n";
  return failure();
}

/**
 * @brief Configure the QDMI registry before initializing its singleton.
 */
[[nodiscard]] static LogicalResult configureQDMIRegistry(const StringRef path) {
#ifdef _WIN32
  const auto status =
      _putenv_s("MQT_CORE_QDMI_CONFIG_FILE", path.str().c_str());
#else
  const auto status =
      setenv("MQT_CORE_QDMI_CONFIG_FILE", path.str().c_str(), 1);
#endif
  return reportQDMIErrorIf(
      status != 0,
      Twine("Failed to configure the QDMI registry from '") + path + "'.");
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
 * @brief Write serialized `jeff` bytes to an output file.
 */
static LogicalResult writeJeffOutput(ModuleOp mod, const StringRef filename) {
  if (failed(serializeToFile(mod, filename))) {
    llvm::errs() << "Failed to write jeff file '" << filename << "'.\n";
    return failure();
  }
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
    if (filename == "-" || llvm::sys::path::extension(filename) == ".ll") {
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

static int runCompiler(int argc, char** argv) {
  const llvm::InitLLVM y(argc, argv);

  registerMQTCompilerPasses();
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  PassPipelineCLParser passPipeline(
      "passes", "QCO optimization passes to run instead of the default");

  // Parse command-line options; exit on error and print to stderr
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MQT Compiler Collection Driver\n");

  if ((!qdmiConfig.empty() && configureQDMIRegistry(qdmiConfig).failed()) ||
      reportQDMIErrorIf(
          qdmiListDevices && !qdmiDevice.empty(),
          "--qdmi-list-devices cannot be combined with --qdmi-device.")
          .failed() ||
      reportQDMIErrorIf(
          !qdmiConfig.empty() && !qdmiListDevices && qdmiDevice.empty(),
          "--qdmi-config requires --qdmi-device or --qdmi-list-devices.")
          .failed()) {
    return 1;
  }
  if (qdmiListDevices) {
    auto deviceIds = registeredQDMIDeviceIds();
    if (!deviceIds) {
      llvm::errs() << "Failed to list configured QDMI devices: "
                   << llvm::toString(deviceIds.takeError()) << '\n';
      return 1;
    }
    for (const auto& id : *deviceIds) {
      llvm::outs() << id << "\n";
    }
    return 0;
  }

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

  std::optional<CompilerTarget> compilerTarget;
  if (!qdmiDevice.empty()) {
    if (reportQDMIErrorIf(
            *parsedOutputFormat == OutputFormat::QCImport ||
                *parsedOutputFormat == OutputFormat::QCO ||
                *parsedOutputFormat == OutputFormat::Jeff,
            "--qdmi-device requires qco-optimized, qc/mlir, qir-base, or "
            "qir-adaptive output.")
            .failed() ||
        reportQDMIErrorIf(passPipeline.hasAnyOccurrences(),
                          "--qdmi-device cannot be combined with --passes.")
            .failed() ||
        reportQDMIErrorIf(
            enableDecomposeMultiControlled,
            "--qdmi-device cannot be combined with "
            "--decompose-multi-controlled; target compilation already "
            "performs the required decomposition.")
            .failed()) {
      return 1;
    }
    auto target = compilerTargetFromDeviceId(qdmiDevice.getValue());
    if (!target) {
      llvm::errs() << "Failed to create compiler target from QDMI device '"
                   << qdmiDevice << "': " << llvm::toString(target.takeError())
                   << '\n';
      return 1;
    }
    compilerTarget.emplace(std::move(*target));
  }

  // Set up MLIR context with all required dialects
  DialectRegistry registry;
  registry
      .insert<arith::ArithDialect, cbit::CBitDialect, cf::ControlFlowDialect,
              func::FuncDialect, LLVM::LLVMDialect, math::MathDialect,
              memref::MemRefDialect, mlir::mqt::MQTDialect, qc::QCDialect,
              qco::QCODialect, qtensor::QTensorDialect, scf::SCFDialect,
              tensor::TensorDialect, jeff::JeffDialect>();
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);

  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  ParsedProgram program;
  switch (*parsedInputFormat) {
  case InputFormat::MLIR:
    program.mod = loadMLIRFile(inputFilename, &context);
    if (program.mod) {
      program.dialect = detectInputDialect(*program.mod);
    }
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
          decomposeMultiControlledMinQubits.getValue())) {
    llvm::errs()
        << "decompose-multi-controlled-min-qubits must be at least 3 when "
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
      failed(qco::verifyLinearity(*program.mod))) {
    return 1;
  }

  if (*parsedOutputFormat != OutputFormat::QCImport &&
      *parsedOutputFormat != OutputFormat::QCO) {
    if (failed(runPasses([&](OpPassManager& pm) {
          if (compilerTarget) {
            populateTargetCompilationPipeline(pm, *compilerTarget);
            return success();
          }
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
                  pm, decomposeMultiControlledMinQubits.getValue());
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
        pm.addPass(mqt::createUnrollModifiers());
        pm.addPass(createQCOToJeff());
        populateJeffCleanupPipeline(pm);
        return success();
      }))) {
    return 1;
  }

  if ((*parsedOutputFormat == OutputFormat::QC ||
       *parsedOutputFormat == OutputFormat::OpenQASM3 ||
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
        pm.addPass(mqt::createUnrollModifiers());
        pm.addPass(createQCToQIRBase());
        populateQIRCleanupPipeline(pm, false);
        return success();
      }))) {
    return 1;
  }

  if (*parsedOutputFormat == OutputFormat::QIRAdaptive &&
      failed(runPasses([](OpPassManager& pm) {
        pm.addPass(mqt::createUnrollModifiers());
        pm.addPass(createQCToQIRAdaptive());
        populateQIRCleanupPipeline(pm, true);
        return success();
      }))) {
    return 1;
  }

  // Write the output
  if (*parsedOutputFormat == OutputFormat::Jeff) {
    if (failed(writeJeffOutput(*program.mod, outputFilename))) {
      return 1;
    }
  } else if (*parsedOutputFormat == OutputFormat::OpenQASM3) {
    std::string errorMessage;
    auto output = openOutputFile(outputFilename, &errorMessage);
    if (!output) {
      llvm::errs() << "Failed to open output file '" << outputFilename
                   << "': " << errorMessage << "\n";
      return 1;
    }
    if (failed(qc::translateQCToOpenQASM3(*program.mod, output->os()))) {
      return 1;
    }
    output->keep();
  } else if (*parsedOutputFormat == OutputFormat::QIRBase ||
             *parsedOutputFormat == OutputFormat::QIRAdaptive) {
    llvm::LLVMContext llvmContext;
    std::unique_ptr<llvm::Module> llvmMod =
        translateModuleToLLVMIR(*program.mod, llvmContext);
    if (!llvmMod) {
      llvm::errs() << "Failed to translate MLIR module to LLVM IR\n";
      return 1;
    }
    qir::normalizeQIRModuleFlags(*llvmMod, *program.mod);
    if (writeOutput<llvm::Module*>(llvmMod.get(), outputFilename).failed()) {
      return 1;
    }
  } else if (writeOutput<ModuleOp>(program.mod.get(), outputFilename)
                 .failed()) {
    return 1;
  }

  return 0;
}

int main(int argc, char** argv) { return runCompiler(argc, argv); }
