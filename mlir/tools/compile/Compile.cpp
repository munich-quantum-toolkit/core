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
#include "mqt/Conversion/QCToQIR/QCToQIR.h"
#include "mqt/Dialect/MQT/IR/MQTAttributes.h"
#include "mqt/Dialect/QCO/Utils/LinearityUtils.h"
#include "mqt/Support/Passes.h"
#include "qdmi/QDMI.hpp"

#include "NativeCommands.h"

#include "jeff/Translation/Serialize.hpp"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Debug/CLOptionsSetup.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/Timing.h"
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

/// Parse an input format or infer it from a filename.
///
/// Standard input defaults to MLIR.
[[nodiscard]] static std::optional<InputFormat>
parseInputFormat(const StringRef format, const StringRef filename) {
  if (format == "auto") {
    return filename == "-" ? InputFormat::MLIR : inputFormatForPath(filename);
  }
  return llvm::StringSwitch<std::optional<InputFormat>>(format)
      .Case("mlir", InputFormat::MLIR)
      .Case("openqasm", InputFormat::OpenQASM)
      .Case("jeff", InputFormat::Jeff)
      .Default(std::nullopt);
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

/// An error whose diagnostics were already reported.
[[nodiscard]] static llvm::Error reportedError() {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), "");
}

/// Print every error that was not reported yet and return the exit code.
[[nodiscard]] static int exitCode(llvm::Error error) {
  int code = 0;
  llvm::handleAllErrors(std::move(error), [&](const llvm::ErrorInfoBase& info) {
    code = 1;
    if (const auto message = info.message(); !message.empty()) {
      llvm::errs() << message << '\n';
    }
  });
  return code;
}

/// Configure the QDMI registry before initializing its singleton.
[[nodiscard]] static llvm::Error configureQDMIRegistry(const StringRef path) {
#ifdef _WIN32
  const auto status =
      _putenv_s("MQT_CORE_QDMI_CONFIG_FILE", path.str().c_str());
#else
  const auto status =
      setenv("MQT_CORE_QDMI_CONFIG_FILE", path.str().c_str(), 1);
#endif
  if (status != 0) {
    return llvm::createStringError(
        Twine("Failed to configure the QDMI registry from '") + path + "'.");
  }
  return llvm::Error::success();
}

/// Add the main source buffer to the diagnostic source manager.
static llvm::Error loadSourceFile(const StringRef filename,
                                  llvm::SourceMgr& sourceMgr) {
  std::string errorMessage;
  auto file = openInputFile(filename, &errorMessage);
  if (!file) {
    return llvm::createStringError(Twine("Failed to load file '") + filename +
                                   "': '" + errorMessage + "'");
  }
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  return llvm::Error::success();
}

/// Write serialized `jeff` bytes to an output file.
static llvm::Error writeJeffOutput(ModuleOp mod, const StringRef filename) {
  if (failed(serializeToFile(mod, filename))) {
    return llvm::createStringError(Twine("Failed to write jeff file '") +
                                   filename + "'.");
  }
  return llvm::Error::success();
}

/// Flush and keep a written output file.
static llvm::Error keepOutput(llvm::ToolOutputFile& output,
                              const StringRef filename) {
  output.os().flush();
  if (output.os().has_error()) {
    return llvm::createStringError(
        Twine("I/O error while writing output file: ") + filename);
  }
  output.keep();
  return llvm::Error::success();
}

/// Write MLIR assembly to stdout or bytecode to a file.
static llvm::Error writeOutput(ModuleOp mod, StringRef filename) {
  std::string errorMessage;
  const auto output = openOutputFile(filename, &errorMessage);
  if (!output) {
    return llvm::createStringError(errorMessage);
  }
  if (filename == "-") {
    mod.print(output->os());
  } else if (failed(writeBytecodeToFile(mod, output->os()))) {
    return reportedError();
  }
  return keepOutput(*output, filename);
}

/// Write a completed payload without truncating the destination on export
/// failure.
static llvm::Error writePayload(StringRef payload, StringRef filename,
                                bool binary = false) {
  if (binary && filename == "-") {
    if (auto error = llvm::sys::ChangeStdoutToBinary()) {
      return llvm::createStringError(error.message());
    }
  }
  std::string errorMessage;
  auto output = openOutputFile(filename, &errorMessage);
  if (!output) {
    return llvm::createStringError(errorMessage);
  }
  output->os() << payload;
  return keepOutput(*output, filename);
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
  registerQCToQIR();
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  tracing::DebugConfig::registerCLOptions();

  // Parse command-line options; exit on error and print to stderr
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "MQT Compiler Collection Driver\n");
  return exitCode([&]() -> llvm::Error {
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
      return llvm::createStringError(
          "Mapping controls require --qdmi-device or --target.");
    }
    if (mappingTrials.getNumOccurrences() != 0 && mappingTrials == 0) {
      return llvm::createStringError(
          "--mapping-trials must be greater than zero.");
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
      return llvm::createStringError(
          "Use either --run-pipeline with --pass-pipeline or "
          "--run-reproducer, without --emit, target compilation, "
          "or --decompose-multi-controlled.");
    }

    if (!qdmiConfig.empty()) {
      if (auto error = configureQDMIRegistry(qdmiConfig)) {
        return error;
      }
    }
    if (!qdmiDevice.empty() && hasExplicitTarget) {
      return llvm::createStringError(
          "--qdmi-device and --target are mutually exclusive.");
    }
    if (qdmiListDevices && hasTarget) {
      return llvm::createStringError(
          "--qdmi-list-devices cannot be combined with target compilation.");
    }
    if ((hasExplicitTarget && payloadSpecification.empty()) ||
        (!hasTarget && !payloadSpecification.empty())) {
      return llvm::createStringError(
          "--qdmi-device or --target and --payload-spec must be provided "
          "together.");
    }
    if (hasTarget && outputFormat.getNumOccurrences() != 0 &&
        outputFormat != "qco-optimized" && !emitQPY) {
      return llvm::createStringError(
          "Only --emit=qco-optimized or qpy can be combined with target "
          "compilation; --payload-spec selects the executable output.");
    }
    if (!qdmiConfig.empty() && !qdmiListDevices && qdmiDevice.empty()) {
      return llvm::createStringError(
          "--qdmi-config requires --qdmi-device or --qdmi-list-devices.");
    }
    if (qdmiListDevices) {
      auto deviceIds = qdmi::builtin_driver::registeredDeviceIds();
      if (!deviceIds) {
        return llvm::createStringError(
            "Failed to list configured QDMI devices: " +
            llvm::toString(deviceIds.takeError()));
      }
      for (const auto& id : *deviceIds) {
        llvm::outs() << id << "\n";
      }
      return llvm::Error::success();
    }

    const auto parsedInputFormat = parseInputFormat(inputFormat, inputFilename);
    if (!parsedInputFormat) {
      return llvm::createStringError(
          "Could not determine the input format for '" + inputFilename +
          "'. Use --input-format.");
    }
    if (isolated && *parsedInputFormat != InputFormat::MLIR) {
      return llvm::createStringError(
          "Isolated pipeline execution requires MLIR input.");
    }
    if (emitQPY && qpyWriter == nullptr) {
      return llvm::createStringError(
          "QPY output requires Qiskit: install mqt-core[qiskit].");
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
      return llvm::createStringError("Unknown output format '" + outputFormat +
                                     "'.");
    }

    std::optional<CompilerTarget> compilerTarget;
    std::optional<PayloadSpecification> selectedPayload;
    if (hasTarget && passPipeline.getNumOccurrences() != 0) {
      return llvm::createStringError(
          "Target compilation cannot be combined with --pass-pipeline.");
    }
    if (hasTarget && enableDecomposeMultiControlled) {
      return llvm::createStringError(
          "Target compilation cannot be combined with "
          "--decompose-multi-controlled; target compilation already "
          "performs the required decomposition.");
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
        return llvm::createStringError(
            "Failed to create compiler target from QDMI device '" + qdmiDevice +
            "': " + llvm::toString(target.takeError()));
      }
      compilerTarget.emplace(std::move(*target));
    }

    llvm::SourceMgr sourceMgr;
    auto context = createCompilerContext();
    SourceMgrDiagnosticHandler diagnosticHandler(sourceMgr, context.get());
    // The handler installs nothing when all action-tracing options are
    // disabled. It reports setup failures, such as unwritable files, as
    // diagnostics.
    std::optional<tracing::InstallDebugHandler> debugHandler;
    bool debugSetupFailed = false;
    {
      ScopedDiagnosticHandler setupDiagnosticHandler(
          context.get(), [&](Diagnostic& diagnostic) {
            if (diagnostic.getSeverity() == DiagnosticSeverity::Error) {
              debugSetupFailed = true;
            }
            return failure();
          });
      debugHandler.emplace(*context,
                           tracing::DebugConfig::createFromCLOptions());
    }
    if (debugSetupFailed) {
      return reportedError();
    }
    PassReproducerOptions reproducerOptions;
    ParserConfig parserConfig(context.get(),
                              /*verifyAfterParse=*/!runReproducer);
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
        return llvm::createStringError(
            "--target must be a valid #mqt.compilation_target attribute.");
      }
      auto target = CompilerTarget::create(targetAttr);
      if (!target) {
        return llvm::createStringError("Invalid --target: " +
                                       llvm::toString(target.takeError()));
      }
      compilerTarget.emplace(std::move(*target));
    }

    if (!payloadSpecification.empty()) {
      const auto attribute =
          parseAttribute(payloadSpecification, context.get());
      const auto payloadAttr =
          dyn_cast_if_present<mlir::mqt::PayloadSpecAttr>(attribute);
      if (!payloadAttr) {
        return llvm::createStringError(
            "--payload-spec must be a valid #mqt.payload_spec attribute.");
      }
      auto payload = PayloadSpecification::create(payloadAttr);
      if (!payload) {
        return llvm::createStringError("Invalid --payload-spec: " +
                                       llvm::toString(payload.takeError()));
      }
      selectedPayload.emplace(std::move(*payload));
    }

    std::optional<TargetEnvironment> targetEnvironment;
    if (compilerTarget) {
      if (outputFormat.getNumOccurrences() == 0 && !emitQPY) {
        auto compilerOutput = selectedPayload->compilerOutput();
        if (!compilerOutput) {
          return compilerOutput.takeError();
        }
        parsedOutputFormat = *compilerOutput;
      }
      targetEnvironment.emplace(std::move(*compilerTarget),
                                std::move(*selectedPayload));
    }

    if (auto error = loadSourceFile(inputFilename, sourceMgr)) {
      return error;
    }
    const auto parseCustomPipeline = [&](OpPassManager& pm) -> llvm::Error {
      auto [anchor, pipeline] = StringRef(passPipeline).trim().split('(');
      if (anchor.rtrim() != ModuleOp::getOperationName() ||
          !pipeline.consume_back(")")) {
        return llvm::createStringError(
            "--pass-pipeline must be anchored on builtin.module.");
      }
      return failed(parsePassPipeline(pipeline, pm)) ? reportedError()
                                                     : llvm::Error::success();
    };

    if (isolated) {
      auto mod = parseSourceFile<ModuleOp>(sourceMgr, parserConfig);
      if (!mod) {
        return reportedError();
      }
      PassManager pm(context.get());
      if (runReproducer) {
        if (failed(reproducerOptions.apply(pm))) {
          return reportedError();
        }
        if (pm.empty() ||
            pm.getOpAnchorName() != ModuleOp::getOperationName()) {
          return llvm::createStringError(
              "--run-reproducer requires a recorded, non-empty "
              "builtin.module pipeline.");
        }
      } else {
        if (auto error = parseCustomPipeline(pm)) {
          return error;
        }
        if (failed(qco::verifyLinearity(*mod))) {
          return reportedError();
        }
      }
      if (failed(runWithCompilationOptions(pm, *mod, options)) ||
          (!runReproducer && failed(qco::verifyLinearity(*mod)))) {
        return reportedError();
      }
      return writeOutput(*mod, outputFilename);
    }

    if (passPipeline.getNumOccurrences() != 0) {
      PassManager validation(context.get());
      if (auto error = parseCustomPipeline(validation)) {
        return error;
      }
    }

    auto input = parseCompilerInput(sourceMgr, context, *parsedInputFormat);
    if (failed(input)) {
      return reportedError();
    }
    if (enableDecomposeMultiControlled &&
        *parsedOutputFormat != ProgramFormat::QCImport &&
        *parsedOutputFormat != ProgramFormat::QCO &&
        passPipeline.getNumOccurrences() == 0) {
      if (!std::holds_alternative<QCOProgram>(*input)) {
        auto qco = std::holds_alternative<QCProgram>(*input)
                       ? std::move(std::get<QCProgram>(*input)).intoQCO()
                       : std::move(std::get<JeffProgram>(*input)).intoQCO();
        if (failed(qco)) {
          return reportedError();
        }
        *input = std::move(*qco);
      }
      if (failed(std::get<QCOProgram>(*input).decomposeMultiControlled(
              decomposeMultiControlledMinQubits))) {
        return reportedError();
      }
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
      return compiled.takeError();
    }

    if (emitQPY) {
      auto payload = qpyWriter(std::move(std::get<QCProgram>(*compiled)));
      if (!payload) {
        return llvm::createStringError("QPY export failed: " +
                                       llvm::toString(payload.takeError()));
      }
      return writePayload(*payload, outputFilename, true);
    }
    if (auto* jeff = std::get_if<JeffProgram>(&*compiled)) {
      return writeJeffOutput(jeff->module(), outputFilename);
    }
    if (auto* qasm = std::get_if<OpenQASMProgram>(&*compiled)) {
      return writePayload(qasm->source(), outputFilename);
    }
    if (auto* iqm = std::get_if<IQMJSONProgram>(&*compiled)) {
      return writePayload(iqm->source(), outputFilename);
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
        if (failed(source)) {
          return reportedError();
        }
        return writePayload(*source, outputFilename);
      }
      std::string bytes;
      llvm::raw_string_ostream stream(bytes);
      if (failed(std::move(*qir).intoBitcode(stream))) {
        return reportedError();
      }
      return writePayload(bytes, outputFilename, true);
    }
    if (auto* qc = std::get_if<QCProgram>(&*compiled)) {
      return writeOutput(qc->module(), outputFilename);
    }
    return writeOutput(std::get<QCOProgram>(*compiled).module(),
                       outputFilename);
  }());
}
