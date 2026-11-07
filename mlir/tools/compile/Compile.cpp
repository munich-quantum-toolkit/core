/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Compiler/Programs.h"
#include "mqt/Compiler/QDMIAdapter.h"
#include "mqt/Compiler/Target.h"
#include "mqt/Compiler/TargetCompilation.h"
#include "mqt/Compiler/TargetEnvironment.h"
#include "mqt/Conversion/JeffToQCO/JeffToQCO.h"
#include "mqt/Conversion/QCOToJeff/QCOToJeff.h"
#include "mqt/Conversion/QCOToQC/QCOToQC.h"
#include "mqt/Conversion/QCToQCO/QCToQCO.h"
#include "mqt/Conversion/QCToQIR/QIRAdaptive/QCToQIRAdaptive.h"
#include "mqt/Conversion/QCToQIR/QIRBase/QCToQIRBase.h"
#include "mqt/Dialect/MQT/IR/MQTAttributes.h"
#include "mqt/Dialect/QCO/Utils/LinearityUtils.h"
#include "mqt/Support/Passes.h"
#include "mqt/Target/OpenQASM/TranslateOpenQASMToQC.h"
#include "qdmi/QDMI.hpp"

#include "NativeCommands.h"

#include "jeff/Translation/Deserialize.hpp"
#include "jeff/Translation/Serialize.hpp"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

using namespace mlir;

namespace {
enum class InputFormat : std::uint8_t { MLIR, OpenQASM, Jeff };
enum class InputDialect : std::uint8_t { QC, QCO };

struct ParsedProgram {
  OwningOpRef<ModuleOp> mod;
  InputDialect dialect = InputDialect::QC;
};
} // namespace

/// Parse an input format or infer it from a filename.
[[nodiscard]] static std::optional<InputFormat>
parseInputFormat(const StringRef format, const StringRef filename) {
  if (format == "mlir" || (format == "auto" && filename.ends_with(".mlir"))) {
    return InputFormat::MLIR;
  }
  if (format == "openqasm" ||
      (format == "auto" && filename.ends_with(".qasm"))) {
    return InputFormat::OpenQASM;
  }
  if (format == "jeff" || (format == "auto" && filename.ends_with(".jeff"))) {
    return InputFormat::Jeff;
  }
  if (format == "auto" && filename == "-") {
    return InputFormat::MLIR;
  }
  return std::nullopt;
}

/// Detect the input dialect of a module.
///
/// Defaults to QC if no QCO operation is found.
[[nodiscard]] static InputDialect detectInputDialect(ModuleOp mod) {
  const bool hasQCO =
      mod->walk([](Operation* operation) {
           return operation->getDialect()->getNamespace() == "qco"
                      ? WalkResult::interrupt()
                      : WalkResult::advance();
         })
          .wasInterrupted();
  return hasQCO ? InputDialect::QCO : InputDialect::QC;
}

/// Parse an output format.
[[nodiscard]] static std::optional<ProgramFormat>
parseOutputFormat(const StringRef format) {
  return llvm::StringSwitch<std::optional<ProgramFormat>>(format)
      .Case("qc-import", ProgramFormat::QCImport)
      .Case("qc", ProgramFormat::QC)
      .Case("qco", ProgramFormat::QCO)
      .Case("qco-optimized", ProgramFormat::QCOOptimized)
      .Case("openqasm2", ProgramFormat::OpenQASM2)
      .Case("openqasm3", ProgramFormat::OpenQASM3)
      .Case("qir-base", ProgramFormat::QIRBase)
      .Case("qir-adaptive", ProgramFormat::QIRAdaptive)
      .Case("jeff", ProgramFormat::Jeff)
      .Default(std::nullopt);
}

static llvm::cl::opt<bool> enableDecomposeMultiControlled(
    "decompose-multi-controlled",
    llvm::cl::desc(
        "Decompose controlled X/Y/Z/rotation/phase/SWAP gates and qco.rccx "
        "that act on at "
        "least --decompose-multi-controlled-min-qubits qubits (default 3)."),
    llvm::cl::init(false));

static llvm::cl::opt<unsigned> decomposeMultiControlledMinQubits(
    "decompose-multi-controlled-min-qubits",
    llvm::cl::desc(
        "Minimum qubit count for --decompose-multi-controlled: decompose "
        "controlled X/Y/Z/rotation/phase/SWAP gates and qco.rccx that act on "
        "at least "
        "this many qubits (default 3; must be at least 3). Higher values leave "
        "narrower gates undecomposed."),
    llvm::cl::init(3));

/// Report a violated QDMI command-line constraint.
[[nodiscard]] static LogicalResult reportQDMIErrorIf(const bool condition,
                                                     const Twine& message) {
  if (!condition) {
    return success();
  }
  llvm::errs() << message << "\n";
  return failure();
}

/// Configure the QDMI registry before initializing its singleton.
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

/// Load and parse a `.qasm` file
static OwningOpRef<ModuleOp> loadOpenQASMFile(const StringRef filename,
                                              MLIRContext* const context,
                                              llvm::SourceMgr& sourceMgr) {
  std::string errorMessage;
  auto file = openInputFile(filename, &errorMessage);
  if (!file) {
    llvm::errs() << "Failed to load file '" << filename << "': '"
                 << errorMessage << "'\n";
    return nullptr;
  }

  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  return qc::translateOpenQASMToQC(sourceMgr, context);
}

/// Load and parse an `.mlir` file
static OwningOpRef<ModuleOp> loadMLIRFile(const StringRef filename,
                                          llvm::SourceMgr& sourceMgr,
                                          const ParserConfig& parserConfig) {
  std::string errorMessage;
  auto file = openInputFile(filename, &errorMessage);
  if (!file) {
    llvm::errs() << "Failed to load file '" << filename << "': '"
                 << errorMessage << "'\n";
    return nullptr;
  }

  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  return parseSourceFile<ModuleOp>(sourceMgr, parserConfig);
}

/// Load a `.jeff` file and convert the program to QCO.
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
  if (failed(applyPassManagerCLOptions(pm))) {
    return {};
  }
  pm.addPass(createJeffToQCO());
  if (pm.run(*mod).failed()) {
    llvm::errs() << "Failed to convert jeff input to QCO.\n";
    return {};
  }
  return {.mod = std::move(mod), .dialect = InputDialect::QCO};
}

/// Write serialized `jeff` bytes to an output file.
static LogicalResult writeJeffOutput(ModuleOp mod, const StringRef filename) {
  if (failed(serializeToFile(mod, filename))) {
    llvm::errs() << "Failed to write jeff file '" << filename << "'.\n";
    return failure();
  }
  return success();
}

/// Write MLIR assembly to stdout or bytecode to a file.
static LogicalResult writeOutput(ModuleOp mod, StringRef filename) {
  std::string errorMessage;
  const auto output = openOutputFile(filename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }
  if (filename == "-") {
    mod.print(output->os());
  } else if (failed(writeBytecodeToFile(mod, output->os()))) {
    return failure();
  }
  output->os().flush();
  if (output->os().has_error()) {
    llvm::errs() << "I/O error while writing output file: " << filename << "\n";
    return failure();
  }
  output->keep();
  return success();
}

/// Write a completed payload without truncating the destination on export
/// failure.
static LogicalResult writePayload(StringRef payload, StringRef filename,
                                  bool binary = false) {
  if (binary && filename == "-") {
    if (auto error = llvm::sys::ChangeStdoutToBinary()) {
      llvm::errs() << error.message() << '\n';
      return failure();
    }
  }
  std::string errorMessage;
  auto output = openOutputFile(filename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << '\n';
    return failure();
  }
  output->os() << payload;
  output->os().flush();
  if (output->os().has_error()) {
    llvm::errs() << "I/O error while writing output file: " << filename << '\n';
    return failure();
  }
  output->keep();
  return success();
}

int mqtCoreRunCompiler(int argc, char** argv, MQTQPYWriter qpyWriter) {
  llvm::cl::list<std::string> manifests(
      "qdmi-manifest", llvm::cl::Hidden,
      llvm::cl::desc("Register a QDMI provider manifest"),
      llvm::cl::callback([](const std::string& path) {
        llvm::ExitOnError("mqt-core: ")(
            qdmi::builtin_driver::addManifest(path));
      }));

  // Command-line options
  llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input .jeff/.mlir/.qasm file>"),
      llvm::cl::init("-"));

  llvm::cl::opt<std::string> inputFormat(
      "input-format",
      llvm::cl::desc(
          "Input format: auto, jeff, mlir, or openqasm (default: auto)"),
      llvm::cl::value_desc("format"), llvm::cl::init("auto"));

  llvm::cl::opt<std::string> outputFilename(
      "o",
      llvm::cl::desc("Output filename (for untargeted QIR, - and .ll write "
                     "textual LLVM IR; .bc and other names write LLVM "
                     "bitcode)"),
      llvm::cl::value_desc("filename"), llvm::cl::init("-"));

  llvm::cl::opt<std::string> outputFormat(
      "emit",
      llvm::cl::desc("Output checkpoint (default: infer from -o, otherwise "
                     "qc): qc-import, qc, "
                     "qco, qco-optimized, qir-base, "
                     "qir-adaptive, openqasm2, openqasm3, or jeff "
                     "(Python launcher also supports qpy with Qiskit)"),
      llvm::cl::value_desc("format"), llvm::cl::init("qc"));

  llvm::cl::opt<std::string> passPipeline(
      "pass-pipeline",
      llvm::cl::desc("Module pipeline replacing the default QCO optimizations "
                     "(e.g. builtin.module(canonicalize,cse))"));
  llvm::cl::alias passesAlias("passes", llvm::cl::aliasopt(passPipeline),
                              llvm::cl::desc("Alias for --pass-pipeline"));
  llvm::cl::opt<bool> runIsolatedPipeline(
      "run-pipeline",
      llvm::cl::desc("Run only --pass-pipeline on MLIR input and write MLIR"));
  llvm::cl::opt<bool> runReproducer(
      "run-reproducer",
      llvm::cl::desc("Replay an MLIR file's recorded pipeline and verification "
                     "settings, then write MLIR"));

  llvm::cl::opt<bool> qdmiListDevices(
      "qdmi-list-devices",
      llvm::cl::desc("List configured QDMI device IDs and exit"),
      llvm::cl::init(false));

  llvm::cl::opt<std::string> qdmiDevice(
      "qdmi-device",
      llvm::cl::desc("Compile for the QDMI device with this stable ID"),
      llvm::cl::value_desc("id"), llvm::cl::init(""));

  llvm::cl::opt<std::string> payloadSpecification(
      "payload-spec",
      llvm::cl::desc("Payload as #mqt.payload_spec (default: selected from the "
                     "QDMI device)"),
      llvm::cl::value_desc("attribute"), llvm::cl::init(""));

  llvm::cl::opt<std::string> explicitTarget(
      "target", llvm::cl::desc("Explicit #mqt.compilation_target attribute"),
      llvm::cl::value_desc("attribute"), llvm::cl::init(""));

  llvm::cl::opt<uint64_t> compilationSeedOption(
      "seed", llvm::cl::desc("Override all compiler random seeds"));
  llvm::cl::opt<size_t> mappingTrials(
      "mapping-trials", llvm::cl::desc("Positive native mapping trial count "
                                       "(default: max(8, logical CPUs))"));
  llvm::cl::opt<size_t> mappingIterations(
      "mapping-iterations",
      llvm::cl::desc(
          "Forward/backward layout refinement rounds; zero skips refinement"),
      llvm::cl::init(MappingOptions{}.iterations));
  llvm::cl::opt<size_t> mappingLookahead(
      "mapping-lookahead",
      llvm::cl::desc("Additional two-qubit gates considered "
                     "during routing (zero disables lookahead)"),
      llvm::cl::init(MappingOptions{}.lookahead));
  llvm::cl::opt<size_t> mappingSearchMemoryLimit(
      "mapping-search-memory-limit",
      llvm::cl::desc("Estimated node and layout bytes per routing search, per "
                     "concurrent trial (zero disables node expansion)"),
      llvm::cl::init(MappingOptions{}.searchMemoryLimit));

  llvm::cl::opt<std::string> qdmiConfig(
      "qdmi-config",
      llvm::cl::desc("Use an explicit QDMI registry configuration file"),
      llvm::cl::value_desc("registry.json"), llvm::cl::init(""));

  registerMQTCompilerPasses();
  registerTargetCompilationPasses();
  // Driver-owned conversion stages must also be available for replay.
  registerQCToQCO();
  registerQCOToQC();
  registerJeffToQCO();
  registerQCOToJeff();
  registerQCToQIRBase();
  registerQCToQIRAdaptive();
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();

  // Parse command-line options; exit on error and print to stderr
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MQT Compiler Collection Driver\n");

  const bool emitQPY = outputFormat == "qpy" ||
                       (outputFormat.getNumOccurrences() == 0 &&
                        llvm::sys::path::extension(outputFilename) == ".qpy");
  const bool hasExplicitTarget = explicitTarget.getNumOccurrences() != 0;
  const bool hasTarget = !qdmiDevice.empty() || hasExplicitTarget;
  if ((mappingTrials.getNumOccurrences() != 0 ||
       mappingIterations.getNumOccurrences() != 0 ||
       mappingLookahead.getNumOccurrences() != 0 ||
       mappingSearchMemoryLimit.getNumOccurrences() != 0) &&
      !hasTarget) {
    llvm::errs() << "Mapping controls require --qdmi-device or --target.\n";
    return 1;
  }
  if (mappingTrials.getNumOccurrences() != 0 && mappingTrials == 0) {
    llvm::errs() << "--mapping-trials must be greater than zero.\n";
    return 1;
  }
  const CompilationOptions options{
      .seed = compilationSeedOption.getNumOccurrences() == 0
                  ? std::nullopt
                  : std::optional<uint64_t>{compilationSeedOption.getValue()},
      .mapping =
          {
              .trials = mappingTrials.getNumOccurrences() == 0
                            ? std::nullopt
                            : std::optional<size_t>{mappingTrials.getValue()},
              .iterations = mappingIterations,
              .lookahead = mappingLookahead,
              .searchMemoryLimit = mappingSearchMemoryLimit,
          },
  };

  const bool isolated = runIsolatedPipeline || runReproducer;
  if (isolated &&
      ((runIsolatedPipeline && runReproducer) ||
       (runIsolatedPipeline && passPipeline.getNumOccurrences() == 0) ||
       (runReproducer && passPipeline.getNumOccurrences() != 0) ||
       outputFormat.getNumOccurrences() != 0 || !qdmiDevice.empty() ||
       hasExplicitTarget || qdmiListDevices ||
       enableDecomposeMultiControlled)) {
    llvm::errs() << "Use either --run-pipeline with --pass-pipeline or "
                    "--run-reproducer, without --emit, target compilation, "
                    "or --decompose-multi-controlled.\n";
    return 1;
  }

  if ((!qdmiConfig.empty() && configureQDMIRegistry(qdmiConfig).failed()) ||
      reportQDMIErrorIf(!qdmiDevice.empty() && hasExplicitTarget,
                        "--qdmi-device and --target are mutually exclusive.")
          .failed() ||
      reportQDMIErrorIf(
          qdmiListDevices && hasTarget,
          "--qdmi-list-devices cannot be combined with target compilation.")
          .failed() ||
      reportQDMIErrorIf((hasExplicitTarget && payloadSpecification.empty()) ||
                            (!hasTarget && !payloadSpecification.empty()),
                        "--qdmi-device or --target and --payload-spec must be "
                        "provided together.")
          .failed() ||
      reportQDMIErrorIf(hasTarget && outputFormat.getNumOccurrences() != 0 &&
                            outputFormat != "qco-optimized" && !emitQPY,
                        "Only --emit=qco-optimized or qpy can be combined with "
                        "target compilation; "
                        "--payload-spec selects the executable output.")
          .failed() ||
      reportQDMIErrorIf(
          !qdmiConfig.empty() && !qdmiListDevices && qdmiDevice.empty(),
          "--qdmi-config requires --qdmi-device or --qdmi-list-devices.")
          .failed()) {
    return 1;
  }
  if (qdmiListDevices) {
    auto deviceIds = qdmi::builtin_driver::registeredDeviceIds();
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
  if (isolated && *parsedInputFormat != InputFormat::MLIR) {
    llvm::errs() << "Isolated pipeline execution requires MLIR input.\n";
    return 1;
  }
  if (emitQPY && qpyWriter == nullptr) {
    llvm::errs() << "QPY output requires Qiskit: install mqt-core[qiskit].\n";
    return 1;
  }
  const auto extension = llvm::sys::path::extension(outputFilename);
  const auto inferredOutput =
      llvm::StringSwitch<std::optional<ProgramFormat>>(extension)
          .Case(".jeff", ProgramFormat::Jeff)
          .Case(".qasm", ProgramFormat::OpenQASM3)
          .Cases({".bc", ".ll"}, ProgramFormat::QIRAdaptive)
          .Case(".json", ProgramFormat::IQMJSON)
          .Default(std::nullopt);
  auto parsedOutputFormat =
      emitQPY ? std::optional{ProgramFormat::QC}
      : outputFormat.getNumOccurrences() == 0 && inferredOutput
          ? inferredOutput
          : parseOutputFormat(outputFormat);
  if (!parsedOutputFormat) {
    llvm::errs() << "Unknown output format '" << outputFormat << "'.\n";
    return 1;
  }

  std::optional<CompilerTarget> compilerTarget;
  std::optional<PayloadSpecification> selectedPayload;
  if (hasTarget) {
    if (reportQDMIErrorIf(
            passPipeline.getNumOccurrences() != 0,
            "Target compilation cannot be combined with --pass-pipeline.")
            .failed() ||
        reportQDMIErrorIf(
            enableDecomposeMultiControlled,
            "Target compilation cannot be combined with "
            "--decompose-multi-controlled; target compilation already "
            "performs the required decomposition.")
            .failed()) {
      return 1;
    }
  }
  if (!qdmiDevice.empty()) {
    auto target = [&]() -> llvm::Expected<CompilerTarget> {
      auto device = qdmi::builtin_driver::openDevice(qdmiDevice.getValue());
      if (!device) {
        return device.takeError();
      }
      if (!payloadSpecification.empty()) {
        return compilerTargetFromDevice(*device);
      }
      std::optional<QDMI_Program_Format> format;
      if (outputFormat.getNumOccurrences() == 0 && inferredOutput) {
        switch (*inferredOutput) {
        case ProgramFormat::QIRAdaptive: {
          auto supported = device->getSupportedProgramFormats();
          if (!supported) {
            return supported.takeError();
          }
          const auto adaptive = extension == ".bc"
                                    ? QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE
                                    : QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING;
          const auto base = extension == ".bc"
                                ? QDMI_PROGRAM_FORMAT_QIRBASEMODULE
                                : QDMI_PROGRAM_FORMAT_QIRBASESTRING;
          format = llvm::is_contained(*supported, adaptive) ? adaptive : base;
          break;
        }
        case ProgramFormat::OpenQASM3:
          format = QDMI_PROGRAM_FORMAT_QASM3;
          break;
        case ProgramFormat::IQMJSON:
          format = QDMI_PROGRAM_FORMAT_IQMJSON;
          break;
        case ProgramFormat::Jeff:
          return llvm::createStringError(
              "jeff export cannot be combined with a QDMI device target");
        default:
          llvm_unreachable("Unexpected inferred output format");
        }
      }
      auto environment = outputFormat == "qco-optimized"
                             ? inspectTargetEnvironment(*device, format)
                             : targetEnvironmentFromDevice(*device, format);
      if (!environment) {
        return environment.takeError();
      }
      selectedPayload = environment->payloadSpecification();
      return environment->target();
    }();
    if (!target) {
      llvm::errs() << "Failed to create compiler target from QDMI device '"
                   << qdmiDevice << "': " << llvm::toString(target.takeError())
                   << '\n';
      return 1;
    }
    compilerTarget.emplace(std::move(*target));
  }

  llvm::SourceMgr sourceMgr;
  auto context = createCompilerContext();
  SourceMgrDiagnosticHandler diagnosticHandler(sourceMgr, context.get());
  PassReproducerOptions reproducerOptions;
  ParserConfig parserConfig(context.get(), /*verifyAfterParse=*/!runReproducer);
  if (runReproducer) {
    reproducerOptions.attachResourceParser(parserConfig);
  }

  if (hasExplicitTarget) {
    const auto targetText = StringRef(explicitTarget).trim();
    const auto attribute = targetText.empty()
                               ? Attribute{}
                               : parseAttribute(targetText, context.get());
    const auto targetAttr =
        dyn_cast_if_present<mlir::mqt::CompilationTargetAttr>(attribute);
    if (!targetAttr) {
      llvm::errs()
          << "--target must be a valid #mqt.compilation_target attribute.\n";
      return 1;
    }
    auto target = CompilerTarget::create(targetAttr);
    if (!target) {
      llvm::errs() << "Invalid --target: " << llvm::toString(target.takeError())
                   << '\n';
      return 1;
    }
    compilerTarget.emplace(std::move(*target));
  }

  if (!payloadSpecification.empty()) {
    const auto attribute = parseAttribute(payloadSpecification, context.get());
    const auto payloadAttr =
        dyn_cast_if_present<mlir::mqt::PayloadSpecAttr>(attribute);
    if (!payloadAttr) {
      llvm::errs()
          << "--payload-spec must be a valid #mqt.payload_spec attribute.\n";
      return 1;
    }
    auto payload = PayloadSpecification::create(payloadAttr);
    if (!payload) {
      llvm::errs() << "Invalid --payload-spec: "
                   << llvm::toString(payload.takeError()) << '\n';
      return 1;
    }
    selectedPayload.emplace(std::move(*payload));
  }

  std::optional<TargetEnvironment> targetEnvironment;
  if (compilerTarget) {
    if (outputFormat.getNumOccurrences() == 0 && !emitQPY) {
      auto compilerOutput = selectedPayload->compilerOutput();
      if (!compilerOutput) {
        llvm::errs() << llvm::toString(compilerOutput.takeError()) << '\n';
        return 1;
      }
      parsedOutputFormat = *compilerOutput;
    }
    targetEnvironment.emplace(std::move(*compilerTarget),
                              std::move(*selectedPayload));
  }

  if (*parsedOutputFormat == ProgramFormat::IQMJSON &&
      (!targetEnvironment ||
       targetEnvironment->target().nativeOperationsKind() ==
           CompilerTarget::NativeOperations::Kind::Unrestricted)) {
    llvm::errs() << "IQM JSON requires explicit native IQM operations.\n";
    return 1;
  }

  ParsedProgram program;
  switch (*parsedInputFormat) {
  case InputFormat::MLIR:
    program.mod = loadMLIRFile(inputFilename, sourceMgr, parserConfig);
    if (program.mod) {
      program.dialect = detectInputDialect(*program.mod);
    }
    break;
  case InputFormat::OpenQASM:
    program.mod = loadOpenQASMFile(inputFilename, context.get(), sourceMgr);
    break;
  case InputFormat::Jeff:
    program = loadJeffFile(inputFilename, context.get());
    break;
  }
  if (!program.mod) {
    return 1;
  }
  const auto parseCustomPipeline = [&](OpPassManager& pm) {
    auto [anchor, pipeline] = StringRef(passPipeline).trim().split('(');
    if (anchor.rtrim() != ModuleOp::getOperationName() ||
        !pipeline.consume_back(")")) {
      llvm::errs() << "--pass-pipeline must be anchored on builtin.module.\n";
      return failure();
    }
    return parsePassPipeline(pipeline, pm);
  };

  if (isolated) {
    PassManager pm(context.get());
    if (runReproducer) {
      if (failed(reproducerOptions.apply(pm))) {
        return 1;
      }
      if (pm.empty() || pm.getOpAnchorName() != ModuleOp::getOperationName()) {
        llvm::errs() << "--run-reproducer requires a recorded, non-empty "
                        "builtin.module pipeline.\n";
        return 1;
      }
    } else if (failed(parseCustomPipeline(pm)) ||
               failed(qco::verifyLinearity(*program.mod))) {
      return 1;
    }
    if (failed(runWithCompilationOptions(pm, *program.mod, options))) {
      return 1;
    }
    if (!runReproducer && failed(qco::verifyLinearity(*program.mod))) {
      return 1;
    }
    return failed(writeOutput(*program.mod, outputFilename)) ? 1 : 0;
  }

  if (*parsedOutputFormat == ProgramFormat::QCImport &&
      program.dialect != InputDialect::QC) {
    llvm::errs() << "--emit=qc-import requires QC frontend input.\n";
    return 1;
  }
  if (passPipeline.getNumOccurrences() != 0 &&
      (*parsedOutputFormat == ProgramFormat::QCImport ||
       *parsedOutputFormat == ProgramFormat::QCO)) {
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

  if (passPipeline.getNumOccurrences() != 0) {
    PassManager validation(context.get());
    if (failed(parseCustomPipeline(validation))) {
      return 1;
    }
  }

  if (enableDecomposeMultiControlled &&
      *parsedOutputFormat != ProgramFormat::QCImport &&
      *parsedOutputFormat != ProgramFormat::QCO &&
      passPipeline.getNumOccurrences() == 0) {
    PassManager pm(context.get());
    if (program.dialect == InputDialect::QC) {
      pm.addPass(createQCToQCO());
    }
    // Keep potential ancilla qubits until decomposition.
    populateQCOCleanupPipeline(pm, /*removeUnusedValues=*/false);
    populateDecomposeMultiControlledPipeline(
        pm, decomposeMultiControlledMinQubits.getValue());
    if (failed(runWithCompilationOptions(pm, *program.mod, options))) {
      return 1;
    }
    program.dialect = InputDialect::QCO;
  }
  auto input = [&]() -> FailureOr<CompilerInput> {
    if (program.dialect == InputDialect::QC) {
      auto qc = QCProgram::fromModule(context, std::move(program.mod));
      if (failed(qc)) {
        return failure();
      }
      return CompilerInput(std::move(*qc));
    }
    auto qco = QCOProgram::fromModule(context, std::move(program.mod));
    if (failed(qco)) {
      return failure();
    }
    return CompilerInput(std::move(*qco));
  }();
  if (failed(input)) {
    return 1;
  }
  auto compiled =
      targetEnvironment
          ? runDefaultPipeline(std::move(*input), *parsedOutputFormat,
                               *targetEnvironment, options)
          : runDefaultPipeline(std::move(*input), *parsedOutputFormat,
                               passPipeline.getNumOccurrences() != 0
                                   ? std::string_view(passPipeline.getValue())
                                   : "mqt-qco-default",
                               options);
  if (!compiled) {
    llvm::errs() << llvm::toString(compiled.takeError()) << '\n';
    return 1;
  }

  if (emitQPY) {
    auto payload = qpyWriter(std::move(std::get<QCProgram>(*compiled)),
                             targetEnvironment ? &*targetEnvironment : nullptr);
    if (!payload) {
      llvm::errs() << "QPY export failed: "
                   << llvm::toString(payload.takeError()) << '\n';
      return 1;
    }
    return failed(writePayload(*payload, outputFilename, true)) ? 1 : 0;
  }
  if (auto* jeff = std::get_if<JeffProgram>(&*compiled)) {
    return failed(writeJeffOutput(jeff->module(), outputFilename)) ? 1 : 0;
  }
  if (auto* qasm = std::get_if<OpenQASMProgram>(&*compiled)) {
    return failed(writePayload(qasm->source(), outputFilename)) ? 1 : 0;
  }
  if (auto* iqm = std::get_if<IQMJSONProgram>(&*compiled)) {
    return failed(writePayload(iqm->source(), outputFilename)) ? 1 : 0;
  }
  if (auto* qir = std::get_if<QIRProgram>(&*compiled)) {
    const auto encoding =
        targetEnvironment
            ? targetEnvironment->payloadSpecification().format().encoding
        : outputFilename == "-" ||
                llvm::sys::path::extension(outputFilename) == ".ll"
            ? PayloadEncoding::Text
            : PayloadEncoding::Binary;
    if (encoding == PayloadEncoding::Text) {
      auto source = std::move(*qir).intoLLVMIR();
      return failed(source) || failed(writePayload(*source, outputFilename))
                 ? 1
                 : 0;
    }
    std::string bytes;
    llvm::raw_string_ostream stream(bytes);
    return failed(std::move(*qir).intoBitcode(stream)) ||
                   failed(writePayload(bytes, outputFilename, true))
               ? 1
               : 0;
  }
  if (auto* qc = std::get_if<QCProgram>(&*compiled)) {
    return failed(writeOutput(qc->module(), outputFilename)) ? 1 : 0;
  }
  return failed(writeOutput(std::get<QCOProgram>(*compiled).module(),
                            outputFilename))
             ? 1
             : 0;
}
