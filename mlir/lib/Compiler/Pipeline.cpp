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
#include "mqt/Compiler/TargetCompilation.h"
#include "mqt/Compiler/TargetEnvironment.h"
#include "mqt/Conversion/JeffToQCO/JeffToQCO.h"
#include "mqt/Conversion/QCOToJeff/QCOToJeff.h"
#include "mqt/Conversion/QCOToQC/QCOToQC.h"
#include "mqt/Conversion/QCToQCO/QCToQCO.h"
#include "mqt/Conversion/QCToQIR/QCToQIR.h"
#include "mqt/Dialect/MQT/IR/MQTAttributes.h"
#include "mqt/Dialect/MQT/Transforms/GlobalPhaseNormalization.h"
#include "mqt/Dialect/MQT/Transforms/Passes.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"
#include "mqt/Dialect/QCO/Utils/LinearityUtils.h"
#include "mqt/Dialect/QTensor/Transforms/Passes.h"
#include "mqt/Support/Passes.h"
#include "mqt/Target/IQM/Export.h"
#include "mqt/Target/OpenQASM/TranslateQCToOpenQASM.h"
#include "mqt/Target/QIR/Export.h"

#include "ActionTrace.h"
#include "Diagnostics.h"
#include "Export.h"
#include "ProgramAccess.h"
#include "ResonatorRouting.h"
#include "TerminalReadout.h"

#include "jeff/Translation/Serialize.hpp"
#include "kj/array.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

using llvm::createStringError;
using llvm::Expected;

namespace mlir {

[[nodiscard]] static LogicalResult runQCOTransformPasses(
    ModuleOp mod, llvm::function_ref<void(OpPassManager&)> populatePasses,
    StringRef failureMessage, const CompilationOptions& options = {},
    bool preservesLayout = false) {
  if (failed(runWithPassManager(mod, populatePasses, failureMessage, options,
                                preservesLayout))) {
    return failure();
  }
  return qco::verifyLinearity(mod);
}

/// Lower quantum modifiers and convert QC to QIR.
static void populateQCToQIRPipeline(OpPassManager& pm, QIRProfile profile,
                                    const TargetEnvironment* environment) {
  pm.addPass(mqt::createLowerQuantumModifiers());
  if (profile == QIRProfile::Adaptive) {
    pm.addPass(mqt::createPromoteClassicalMemory());
  }
  const QCToQIROptions options{.profile = profile};
  pm.addPass(environment != nullptr ? createQCToQIR(options, *environment)
                                    : createQCToQIR(options));
}

/// Convert QCO to QC and expand virtual resonator couplings.
static void populateQCOToQCPipeline(OpPassManager& pm) {
  pm.addPass(createQCOToQC());
  pm.addPass(createLowerResonatorCZPass());
}

/// Expand unitary calls in modifiers and convert QCO to `jeff`.
static void populateQCOToJeffPipeline(OpPassManager& pm) {
  pm.addPass(mqt::createUnrollModifiers());
  pm.addPass(createQCOToJeff());
}

//===----------------------------------------------------------------------===//
// QCProgram
//===----------------------------------------------------------------------===//

LogicalResult QCProgram::cleanup() {
  return runWithPassManager(mod(), populateQCCleanupPipeline,
                            "failed to run the QC cleanup pipeline");
}

LogicalResult QCProgram::normalizeGlobalPhases() {
  return mqt::normalizeGlobalPhases(mod());
}

FailureOr<OpenQASMProgram>
QCProgram::intoOpenQASM3(const TargetEnvironment* environment) && {
  auto storage = std::move(*this).releaseStorage();
  if (failed(runWithPassManager(
          *storage.mod,
          [](OpPassManager& pm) {
            // Emission needs canonical control flow, not only promoted memory.
            pm.addPass(createCanonicalizerPass());
            pm.addPass(mqt::createPromoteClassicalMemory());
          },
          "failed to prepare QC for OpenQASM export"))) {
    return failure();
  }
  auto source = qc::translateQCToOpenQASM3(std::move(storage.mod), environment);
  if (failed(source)) {
    return failure();
  }
  return OpenQASMProgram(std::move(*source));
}

static FailureOr<IQMJSONProgram> exportIQMJSON(QIRProgram& qir,
                                               const CompilerTarget& target) {
  llvm::LLVMContext context;
  auto llvmModule = qir::translateQIRToLLVM(qir.module(), context);
  if (!llvmModule) {
    return failure();
  }
  auto source = qir::exportIQMJSON(*llvmModule, target);
  if (!source) {
    qir.module().emitError() << llvm::toString(source.takeError());
    return failure();
  }
  return IQMJSONProgram(std::move(*source));
}

FailureOr<IQMJSONProgram>
QCProgram::intoIQMJSON(const CompilerTarget& target) && {
  const TargetEnvironment environment(
      target, llvm::cantFail(PayloadSpecification::forProgramFormat(
                  ProgramFormat::IQMJSON)));
  auto exported =
      exportQCProgram(std::move(*this), ProgramFormat::IQMJSON, &environment);
  if (failed(exported)) {
    return failure();
  }
  return std::move(std::get<IQMJSONProgram>(*exported));
}

FailureOr<OpenQASMProgram>
QCProgram::intoOpenQASM2(const TargetEnvironment* environment) && {
  auto storage = std::move(*this).releaseStorage();
  auto source = qc::translateQCToOpenQASM2(std::move(storage.mod), environment);
  if (failed(source)) {
    return failure();
  }
  return OpenQASMProgram(std::move(*source));
}

FailureOr<QIRProgram>
QCProgram::intoQIR(QIRProfile profile,
                   const TargetEnvironment* environment) && {
  auto storage = std::move(*this).releaseStorage();
  if (failed(runWithPassManager(
          *storage.mod,
          [profile, environment](OpPassManager& pm) {
            populateQCToQIRPipeline(pm, profile, environment);
          },
          "failed to convert QC to QIR"))) {
    return failure();
  }
  return QIRProgram(std::move(storage), profile);
}

//===----------------------------------------------------------------------===//
// QCOProgram
//===----------------------------------------------------------------------===//

LogicalResult QCOProgram::cleanup() {
  return runQCOTransformPasses(
      mod(), [](OpPassManager& pm) { populateQCOCleanupPipeline(pm); },
      "failed to run the QCO cleanup pipeline");
}

LogicalResult QCOProgram::normalizeGlobalPhases() {
  if (failed(mqt::normalizeGlobalPhases(mod()))) {
    return failure();
  }
  return qco::verifyLinearity(mod());
}

LogicalResult QCOProgram::runPassPipeline(std::string_view pipeline,
                                          const CompilationOptions& options,
                                          const TargetEnvironment* environment,
                                          bool preservesLayout) {
  registerTargetCompilationPasses();
  if (failed(::runPassPipeline(mod(), pipeline, options, environment,
                               preservesLayout))) {
    return failure();
  }
  return qco::verifyLinearity(mod());
}

LogicalResult QCOProgram::mergeSingleQubitRotationGates() {
  return runQCOTransformPasses(
      mod(),
      [](OpPassManager& pm) {
        pm.addPass(qco::createMergeSingleQubitRotationGates());
      },
      "failed to merge single-qubit rotation gates");
}

LogicalResult QCOProgram::fuseSingleQubitUnitaryRuns(std::string_view basis) {
  qco::FuseSingleQubitUnitaryRunsOptions options;
  options.basis = basis;
  return runQCOTransformPasses(
      mod(),
      [&options](OpPassManager& pm) {
        pm.addPass(qco::createFuseSingleQubitUnitaryRuns(options));
      },
      "failed to fuse single-qubit unitary runs");
}

LogicalResult QCOProgram::unrollQuantumLoops(int64_t factor) {
  qco::QuantumLoopUnrollOptions options;
  options.unrollFactor = factor;
  return runQCOTransformPasses(
      mod(),
      [&options](OpPassManager& pm) {
        pm.addNestedPass<func::FuncOp>(qco::createQuantumLoopUnroll(options));
      },
      "failed to unroll quantum loops");
}

LogicalResult QCOProgram::liftHadamards() {
  return runQCOTransformPasses(
      mod(),
      [](OpPassManager& pm) { pm.addPass(qco::createHadamardLifting()); },
      "failed to lift Hadamard gates");
}

LogicalResult QCOProgram::reuseQubits() {
  return runQCOTransformPasses(
      mod(), [](OpPassManager& pm) { pm.addPass(qco::createReuseQubits()); },
      "failed to reuse qubits");
}

LogicalResult QCOProgram::runQubitReusePipeline() {
  return runQCOTransformPasses(mod(), populateQubitReusePipeline,
                               "failed to run the qubit reuse pipeline");
}

LogicalResult QCOProgram::decomposeMultiControlled(uint64_t minQubits) {
  return runQCOTransformPasses(
      mod(),
      [minQubits](OpPassManager& pm) {
        // Keep unused qubits available as ancillas.
        populateQCOCleanupPipeline(pm, /*removeUnusedValues=*/false);
        populateDecomposeMultiControlledPipeline(pm, minQubits);
      },
      "failed to decompose multi-controlled gates");
}

LogicalResult QCOProgram::compileForTarget(const TargetEnvironment& environment,
                                           const CompilationOptions& options) {
  attachTargetEnvironment(mod(), environment);
  return runQCOTransformPasses(
      mod(),
      [&](OpPassManager& pm) {
        pm.addPass(createCompileForTargetPass(options.mapping));
      },
      "failed to compile the QCO program for the target", options,
      /*preservesLayout=*/true);
}

LogicalResult
QCOProgram::synthesizeForTarget(const TargetEnvironment& environment,
                                const CompilationOptions& options) {
  return runQCOTransformPasses(
      mod(),
      [&environment, &options](OpPassManager& pm) {
        populateTargetSynthesisPipeline(pm, environment, options.mapping);
      },
      "failed to synthesize the QCO program for the target", options, true);
}

FailureOr<QCProgram> QCOProgram::intoQC() && {
  auto storage = std::move(*this).releaseStorage();
  if (failed(runWithPassManager(*storage.mod, populateQCOToQCPipeline,
                                "failed to convert QCO to QC", {},
                                /*preservesLayout=*/true))) {
    return failure();
  }
  return QCProgram(std::move(storage));
}

FailureOr<JeffProgram> QCOProgram::intoJeff() && {
  auto storage = std::move(*this).releaseStorage();
  if (failed(runWithPassManager(*storage.mod, populateQCOToJeffPipeline,
                                "failed to convert QCO to jeff"))) {
    return failure();
  }
  return JeffProgram(std::move(storage));
}

//===----------------------------------------------------------------------===//
// JeffProgram
//===----------------------------------------------------------------------===//

JeffProgram JeffProgram::copy() const { return JeffProgram(cloneStorage()); }

LogicalResult JeffProgram::cleanup() {
  return runWithPassManager(mod(), populateJeffCleanupPipeline,
                            "failed to run the jeff cleanup pipeline");
}

LogicalResult JeffProgram::toMessage(capnp::MessageBuilder& message) const {
  return serialize(mod(), message);
}

std::vector<std::byte> JeffProgram::toBytes() const {
  const auto serialized = serialize(mod());
  const auto bytes = serialized.asBytes();
  std::vector<std::byte> result(bytes.size());
  std::memcpy(result.data(), bytes.begin(), bytes.size());
  return result;
}

LogicalResult JeffProgram::write(const std::filesystem::path& path) const {
  if (failed(serializeToFile(mod(), path.string()))) {
    mod().emitError() << "failed to write jeff file '" << path.string() << "'";
    return failure();
  }
  return success();
}

FailureOr<QCOProgram> JeffProgram::intoQCO() && {
  auto storage = std::move(*this).releaseStorage();
  auto moduleOp = *storage.mod;
  if (failed(runWithPassManager(
          moduleOp, [](OpPassManager& pm) { pm.addPass(createJeffToQCO()); },
          "failed to convert jeff to QCO"))) {
    return failure();
  }
  if (failed(qco::verifyLinearity(moduleOp))) {
    return failure();
  }
  return QCOProgram(std::move(storage));
}

//===----------------------------------------------------------------------===//
// QIRProgram
//===----------------------------------------------------------------------===//

QIRProgram::QIRProgram(Storage storage, QIRProfile profile)
    : Program(std::move(storage)), profile_(profile) {}

QIRProgram QIRProgram::copy() const { return {cloneStorage(), profile_}; }

LogicalResult QIRProgram::cleanup() {
  return runWithPassManager(
      mod(),
      [this](OpPassManager& pm) { populateQIRCleanupPipeline(pm, profile_); },
      "failed to run the QIR cleanup pipeline");
}

QIRProfile QIRProgram::profile() const noexcept { return profile_; }

FailureOr<std::string> QIRProgram::intoLLVMIR() && {
  auto storage = std::move(*this).releaseStorage();
  std::string result;
  llvm::raw_string_ostream stream(result);
  if (failed(qir::translateQIRToLLVMIR(std::move(storage.mod), stream))) {
    return failure();
  }
  return result;
}

FailureOr<std::vector<std::byte>> QIRProgram::intoBitcode() && {
  SmallVector<char> buffer;
  llvm::raw_svector_ostream stream(buffer);
  if (failed(std::move(*this).intoBitcode(stream))) {
    return failure();
  }
  std::vector<std::byte> result(buffer.size());
  std::memcpy(result.data(), buffer.data(), buffer.size());
  return result;
}

LogicalResult QIRProgram::intoBitcode(llvm::raw_ostream& output) && {
  auto storage = std::move(*this).releaseStorage();
  return qir::translateQIRToBitcode(std::move(storage.mod), output);
}

LogicalResult QIRProgram::writeBitcode(const std::filesystem::path& path) && {
  auto storage = std::move(*this).releaseStorage();
  return qir::writeQIRBitcode(std::move(storage.mod), path.string());
}

//===----------------------------------------------------------------------===//
// Pipeline
//===----------------------------------------------------------------------===//

/// Return the QIR profile of a QIR-based output format.
static QIRProfile qirProfile(ProgramFormat output) {
  return output == ProgramFormat::QIRAdaptive ? QIRProfile::Adaptive
                                              : QIRProfile::Base;
}

/// Prepare compiled QC for the selected output format.
static void populateExportPipeline(OpPassManager& pm, ProgramFormat output,
                                   const TargetEnvironment* environment,
                                   MLIRContext& context) {
  switch (output) {
  case ProgramFormat::OpenQASM3:
    populateQCStructuralCleanupPipeline(pm);
    pm.addPass(mqt::createPromoteClassicalMemory());
    return;
  case ProgramFormat::OpenQASM2:
    pm.addPass(mqt::createSimplifyClassicalControl());
    pm.addPass(mqt::createPromoteClassicalMemory());
    populateOpenQASM2PreparationPipeline(
        pm, llvm::cantFail(PayloadSpecification::forProgramFormat(
                               ProgramFormat::OpenQASM2))
                .materialize(context));
    populateQCStructuralCleanupPipeline(pm);
    return;
  case ProgramFormat::QIRBase:
  case ProgramFormat::QIRAdaptive:
  case ProgramFormat::IQMJSON: {
    const auto profile = qirProfile(output);
    populateQIRPreparationPipeline(
        pm, llvm::cantFail(PayloadSpecification::forProgramFormat(
                               profile == QIRProfile::Adaptive
                                   ? ProgramFormat::QIRAdaptive
                                   : ProgramFormat::QIRBase))
                .materialize(context));
    if (output == ProgramFormat::IQMJSON) {
      pm.addPass(mqt::createDropGlobalPhases());
    }
    populateQCToQIRPipeline(pm, profile, environment);
    return;
  }
  default:
    return;
  }
}

/// Wrap or translate an exported module as the selected output program.
static FailureOr<CompilerProgram>
finishExport(ProgramAccess::Storage storage, ProgramFormat output,
             const TargetEnvironment* environment) {
  switch (output) {
  case ProgramFormat::QC:
    // Keep verified independent gates ahead of readout before native frontends
    // replace them with opaque calls. Preserve resources and routing metadata.
    scheduleTerminalReadout(*storage.mod);
    return CompilerProgram(QCProgram(std::move(storage)));
  case ProgramFormat::OpenQASM2:
  case ProgramFormat::OpenQASM3: {
    auto source =
        output == ProgramFormat::OpenQASM2
            ? qc::translateQCToOpenQASM2(std::move(storage.mod), environment)
            : qc::translateQCToOpenQASM3(std::move(storage.mod), environment);
    if (failed(source)) {
      return failure();
    }
    return CompilerProgram(OpenQASMProgram(std::move(*source)));
  }
  default:
    break;
  }
  QIRProgram qir(std::move(storage), qirProfile(output));
  if (output != ProgramFormat::IQMJSON) {
    return CompilerProgram(std::move(qir));
  }
  auto source = exportIQMJSON(qir, environment->target());
  if (failed(source)) {
    return failure();
  }
  return CompilerProgram(std::move(*source));
}

FailureOr<CompilerProgram> exportQCProgram(QCProgram qc, ProgramFormat output,
                                           const TargetEnvironment* environment,
                                           const CompilationOptions& options) {
  if (output != ProgramFormat::QC &&
      failed(runWithPassManager(
          qc.module(),
          [&](OpPassManager& pm) {
            populateExportPipeline(pm, output, environment,
                                   *qc.module().getContext());
          },
          "failed to export compiled QC", options))) {
    return failure();
  }
  return finishExport(ProgramAccess::release(std::move(qc)), output,
                      environment);
}

/// Populate the untargeted QCO optimization stage.
///
/// A custom pipeline replaces the default optimizations. Its output is
/// verified because it may not preserve QCO linearity.
static void populateQCOOptimizationPipeline(OpPassManager& pm,
                                            std::string_view qcoPipeline,
                                            bool inlineCalls) {
  if (inlineCalls) {
    pm.addPass(createInlinerPass({}, [](OpPassManager&) {}));
  }
  // Optimizations need promoted classical values, not compact storage.
  populateQCOCleanupPipeline(pm, /*removeUnusedValues=*/false);
  if (qcoPipeline == "mqt-qco-default") {
    // The default optimizations end canonicalized; only compact storage.
    populateDefaultQCOOptimizationPipeline(pm);
    pm.addPass(qtensor::createShrinkQTensorToFitPass());
    pm.addPass(createSymbolDCEPass());
    pm.addPass(createRemoveDeadValuesPass());
    return;
  }
  // The pipeline was validated before compilation started.
  llvm::cantFail(parseModulePassPipeline(qcoPipeline, pm));
  pm.addPass(qco::createVerifyQCOProgram());
  populateQCOCleanupPipeline(pm);
}

/// Compile a program in one pass manager run.
///
/// The pass manager verifies the result, and after each pass when assertions
/// are enabled. QCO results are also checked for linearity.
[[nodiscard]] static FailureOr<CompilerProgram>
runDefaultPipelineImpl(CompilerInput program, ProgramFormat output,
                       const TargetEnvironment* environment,
                       std::string_view qcoPipeline,
                       const CompilationOptions& options) {
  if (output == ProgramFormat::QCImport) {
    return CompilerProgram(std::move(std::get<QCProgram>(program)));
  }
  const bool fromQC = std::holds_alternative<QCProgram>(program);
  const bool fromJeff = std::holds_alternative<JeffProgram>(program);
  const bool qirOutput =
      output == ProgramFormat::QIRBase || output == ProgramFormat::QIRAdaptive;
  auto storage = std::visit(
      // Every program branch forwards the consumed value.
      // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
      []<typename T>(T&& value) -> ProgramAccess::Storage {
        if constexpr (std::is_base_of_v<Program, std::remove_cvref_t<T>>) {
          return ProgramAccess::release(std::forward<T>(value));
        } else {
          llvm_unreachable(
              "OpenQASM inputs are imported before running passes");
        }
      },
      std::move(program));
  auto moduleOp = *storage.mod;
  // Every pass reads the target from the module, so pipelines can be printed
  // and replayed, for example from crash reproducers.
  if (environment != nullptr) {
    attachTargetEnvironment(moduleOp, *environment);
  }

  const auto populate = [&](OpPassManager& pm) {
    if (fromQC) {
      pm.addPass(createQCToQCO());
    } else if (fromJeff) {
      // External `jeff` input need not be linear.
      pm.addPass(createJeffToQCO());
      pm.addPass(qco::createVerifyQCOProgram());
    }
    if (output == ProgramFormat::QCO) {
      return;
    }
    if (environment != nullptr) {
      pm.addPass(createCompileForTargetPass(options.mapping));
    } else {
      populateQCOOptimizationPipeline(pm, qcoPipeline,
                                      /*inlineCalls=*/qirOutput);
    }
    if (output == ProgramFormat::QCOOptimized) {
      return;
    }
    if (output == ProgramFormat::Jeff) {
      populateQCOToJeffPipeline(pm);
      populateJeffCleanupPipeline(pm);
      return;
    }
    populateQCOToQCPipeline(pm);
    // Export reads the target that target compilation keeps attached.
    populateExportPipeline(pm, output, nullptr, *moduleOp.getContext());
  };
  // Target compilation composes input layouts. The QIR and `jeff` conversions
  // drop layouts; untargeted optimization invalidates them.
  const bool preservesLayout =
      environment != nullptr || output == ProgramFormat::QCO;
  if (failed(runWithPassManager(moduleOp, populate,
                                "failed to compile the program", options,
                                preservesLayout))) {
    return failure();
  }
  switch (output) {
  case ProgramFormat::QCO:
  case ProgramFormat::QCOOptimized:
    if (failed(qco::verifyLinearity(moduleOp))) {
      return failure();
    }
    return CompilerProgram(ProgramAccess::makeQCO(std::move(storage)));
  case ProgramFormat::Jeff:
    return CompilerProgram(JeffProgram(std::move(storage)));
  default:
    return finishExport(std::move(storage), output, environment);
  }
}

static llvm::Error importOpenQASM(CompilerInput& program) {
  if (auto* source = std::get_if<OpenQASMProgram>(&program)) {
    auto qc = QCProgram::fromOpenQASMString(source->source());
    if (!qc) {
      return qc.takeError();
    }
    program = std::move(*qc);
  }
  return llvm::Error::success();
}

static std::shared_ptr<MLIRContext> contextOf(const CompilerInput& program) {
  return std::visit(
      [](const auto& value) -> std::shared_ptr<MLIRContext> {
        if constexpr (std::is_base_of_v<Program,
                                        std::remove_cvref_t<decltype(value)>>) {
          return value.context();
        } else {
          llvm_unreachable("OpenQASM inputs must be imported first");
        }
      },
      program);
}

llvm::Error
withActionTrace(CompilerInput& program, const CompilationOptions& options,
                function_ref<llvm::Error(const CompilationOptions&)> run) {
  if (!options.traceFile) {
    return run(options);
  }
  if (auto error = importOpenQASM(program)) {
    return error;
  }
  return ::withActionTrace(*contextOf(program), options, run);
}

static Expected<CompilerProgram>
runNativePipeline(CompilerInput program, ProgramFormat output,
                  const TargetEnvironment* environment,
                  std::string_view qcoPipeline,
                  const CompilationOptions& options) {
  if (output == ProgramFormat::IQMJSON &&
      (environment == nullptr ||
       !environment->target().hasDeviceConstraints() ||
       environment->target().nativeOperationsKind() ==
           CompilerTarget::NativeOperations::Kind::Unrestricted)) {
    return createStringError(std::errc::invalid_argument,
                             "IQM JSON requires a physical target with "
                             "explicit native IQM operations");
  }
  if (output == ProgramFormat::QCImport &&
      !std::holds_alternative<QCProgram>(program) &&
      !std::holds_alternative<OpenQASMProgram>(program)) {
    return createStringError(
        std::errc::invalid_argument,
        "QCImport output is only available for QC or OpenQASM input");
  }
  if ((output == ProgramFormat::QCImport || output == ProgramFormat::QCO) &&
      qcoPipeline != "mqt-qco-default") {
    return createStringError(
        std::errc::invalid_argument,
        "a custom QCO pass pipeline cannot be used with an output that stops "
        "before QCO optimization");
  }
  if (qcoPipeline != "mqt-qco-default") {
    registerTargetCompilationPasses();
    OpPassManager pipeline(ModuleOp::getOperationName());
    if (auto error = parseModulePassPipeline(qcoPipeline, pipeline)) {
      return std::move(error);
    }
  }
  return withActionTrace<CompilerProgram>(
      program, options,
      [&](const CompilationOptions& pipelineOptions)
          -> Expected<CompilerProgram> {
        if (auto error = importOpenQASM(program)) {
          return std::move(error);
        }
        const auto context = contextOf(program);
        CompilerDiagnostics diagnostics(context.get());
        auto result =
            runDefaultPipelineImpl(std::move(program), output, environment,
                                   qcoPipeline, pipelineOptions);
        if (failed(result)) {
          return diagnostics.takeError(
              std::make_error_code(std::errc::invalid_argument));
        }
        return std::move(*result);
      });
}

Expected<CompilerProgram>
runDefaultPipeline(CompilerInput program, ProgramFormat output,
                   std::string_view qcoPipeline,
                   const CompilationOptions& options) {
  return runNativePipeline(std::move(program), output, nullptr, qcoPipeline,
                           options);
}

Expected<CompilerProgram>
runDefaultPipeline(CompilerInput program, ProgramFormat output,
                   const TargetEnvironment& environment,
                   const CompilationOptions& options) {
  if (output == ProgramFormat::QCImport || output == ProgramFormat::QCO) {
    return createStringError(
        std::errc::invalid_argument,
        "QCImport and raw QCO output stop before target compilation");
  }
  return runNativePipeline(std::move(program), output, &environment,
                           "mqt-qco-default", options);
}

Expected<CompilerProgram>
runDefaultPipeline(CompilerInput program, const TargetEnvironment& environment,
                   const CompilationOptions& options) {
  auto output = environment.payloadSpecification().compilerOutput();
  if (!output) {
    return output.takeError();
  }
  return runDefaultPipeline(std::move(program), *output, environment, options);
}

} // namespace mlir
