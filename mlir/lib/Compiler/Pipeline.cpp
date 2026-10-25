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
#include "mqt/Conversion/QCToQIR/QIRAdaptive/QCToQIRAdaptive.h"
#include "mqt/Conversion/QCToQIR/QIRBase/QCToQIRBase.h"
#include "mqt/Dialect/MQT/IR/MQTAttributes.h"
#include "mqt/Dialect/MQT/Transforms/GlobalPhaseNormalization.h"
#include "mqt/Dialect/MQT/Transforms/Passes.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"
#include "mqt/Support/Passes.h"
#include "mqt/Target/IQM/Export.h"
#include "mqt/Target/OpenQASM/TranslateQCToOpenQASM3.h"
#include "mqt/Target/QIR/Capabilities.h"
#include "mqt/Target/QIR/Export.h"

#include "capnp/common.h"
#include "jeff/Translation/Deserialize.hpp"
#include "jeff/Translation/Serialize.hpp"
#include "kj/array.h"

#include "mlir/Debug/ExecutionContext.h"
#include "mlir/Debug/Observers/ActionProfiler.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace mlir {

[[nodiscard]] static LogicalResult runQCOTransformPasses(
    ModuleOp mod, llvm::function_ref<void(OpPassManager&)> populatePasses,
    StringRef failureMessage, const CompilationOptions& options = {},
    bool preservesLayout = false) {
  if (failed(qco::verifyLinearity(mod))) {
    return failure();
  }
  if (failed(runWithPassManager(mod, populatePasses, failureMessage, options,
                                preservesLayout))) {
    return failure();
  }
  return qco::verifyLinearity(mod);
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

FailureOr<OpenQASMProgram> QCProgram::intoOpenQASM3() && {
  auto storage = std::move(*this).releaseStorage();
  if (failed(runWithPassManager(*storage.mod, populateQCExportPipeline,
                                "failed to prepare QC for OpenQASM export"))) {
    return failure();
  }
  auto source = qc::translateQCToOpenQASM3(std::move(storage.mod));
  if (failed(source)) {
    return failure();
  }
  return OpenQASMProgram(std::move(*source));
}

FailureOr<QIRProgram> QCProgram::intoQIR(QIRProfile profile) && {
  auto storage = std::move(*this).releaseStorage();
  auto moduleOp = *storage.mod;
  const auto payload = llvm::cantFail(PayloadSpecification::forProgramFormat(
      profile == QIRProfile::Adaptive ? ProgramFormat::QIRAdaptive
                                      : ProgramFormat::QIRBase));
  if (failed(runWithPassManager(
          moduleOp,
          [&](OpPassManager& pm) {
            populateQIRPreparationPipeline(
                pm, payload.materialize(*moduleOp.getContext()));
            if (profile == QIRProfile::Adaptive) {
              pm.addPass(createQCToQIRAdaptive());
            } else {
              pm.addPass(createQCToQIRBase());
            }
            populateQIRCleanupPipeline(pm, profile == QIRProfile::Adaptive);
          },
          "failed to convert QC to QIR"))) {
    return failure();
  }
  if (moduleOp->hasAttr(mqt::TargetEnvAttr::name)) {
    const TargetEnvironmentAnalysis environment(moduleOp);
    if (!environment) {
      moduleOp.emitError(environment.error());
      return failure();
    }
    const auto& targetPayload =
        environment.environment().payloadSpecification();
    const auto& format = targetPayload.format();
    if (format.id == payload.format().id &&
        format.version == payload.format().version &&
        format.profile == payload.format().profile &&
        failed(qir::validateCapabilities(moduleOp,
                                         targetPayload.capabilities()))) {
      return failure();
    }
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
  if (failed(qco::verifyLinearity(mod())) ||
      failed(mqt::normalizeGlobalPhases(mod()))) {
    return failure();
  }
  return qco::verifyLinearity(mod());
}

LogicalResult QCOProgram::runPassPipeline(std::string_view pipeline,
                                          const CompilationOptions& options) {
  if (failed(qco::verifyLinearity(mod())) ||
      failed(::runPassPipeline(mod(), pipeline, options))) {
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
        populateDecomposeMultiControlledPipeline(pm, minQubits);
      },
      "failed to decompose multi-controlled gates");
}

LogicalResult QCOProgram::compileForTarget(const TargetEnvironment& environment,
                                           const CompilationOptions& options) {
  return runQCOTransformPasses(
      mod(),
      [&environment, &options](OpPassManager& pm) {
        populateTargetCompilationPipeline(pm, environment, options.mapping);
      },
      "failed to compile the QCO program for the target", options, true);
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
  auto moduleOp = *storage.mod;
  if (failed(qco::verifyLinearity(moduleOp)) ||
      failed(runWithPassManager(
          moduleOp, [](OpPassManager& pm) { pm.addPass(createQCOToQC()); },
          "failed to convert QCO to QC", {}, true))) {
    return failure();
  }
  return QCProgram(std::move(storage));
}

FailureOr<JeffProgram> QCOProgram::intoJeff() && {
  auto storage = std::move(*this).releaseStorage();
  auto moduleOp = *storage.mod;
  if (failed(qco::verifyLinearity(moduleOp)) ||
      failed(runWithPassManager(
          moduleOp,
          [](OpPassManager& pm) {
            pm.addPass(mqt::createUnrollModifiers());
            pm.addPass(createQCOToJeff());
          },
          "failed to convert QCO to jeff"))) {
    return failure();
  }
  return JeffProgram(std::move(storage));
}

//===----------------------------------------------------------------------===//
// JeffProgram
//===----------------------------------------------------------------------===//

FailureOr<JeffProgram> JeffProgram::fromMessage(::jeff::Module::Reader module) {
  auto context = createCompilerContext();
  auto mod = deserialize(context.get(), module);
  if (!mod) {
    return failure();
  }
  return JeffProgram({.context = std::move(context), .mod = std::move(mod)});
}

FailureOr<JeffProgram>
JeffProgram::fromBytes(std::span<const std::byte> bytes) {
  if (bytes.size() % sizeof(capnp::word) != 0U) {
    auto context = createCompilerContext();
    emitError(UnknownLoc::get(context.get()),
              "jeff data size must be a multiple of the Cap'n Proto word size");
    return failure();
  }

  auto words = kj::heapArray<capnp::word>(bytes.size() / sizeof(capnp::word));
  std::memcpy(words.begin(), bytes.data(), bytes.size());

  auto context = createCompilerContext();
  auto mod = deserialize(context.get(), words.asPtr());
  if (!mod) {
    emitError(UnknownLoc::get(context.get()),
              "failed to deserialize jeff bytes");
    return failure();
  }
  return JeffProgram({.context = std::move(context), .mod = std::move(mod)});
}

FailureOr<JeffProgram>
JeffProgram::fromFile(const std::filesystem::path& path) {
  auto context = createCompilerContext();
  auto mod = deserializeFromFile(context.get(), path.string());
  if (!mod) {
    emitError(UnknownLoc::get(context.get()))
        << "failed to deserialize jeff file '" << path.string() << "'";
    return failure();
  }
  return JeffProgram({.context = std::move(context), .mod = std::move(mod)});
}

JeffProgram JeffProgram::copy() const { return JeffProgram(cloneStorage()); }

LogicalResult JeffProgram::cleanup() {
  return runWithPassManager(mod(), populateJeffCleanupPipeline,
                            "failed to run the jeff cleanup pipeline");
}

void JeffProgram::toMessage(capnp::MessageBuilder& message) const {
  serialize(mod(), message);
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
      [this](OpPassManager& pm) {
        populateQIRCleanupPipeline(pm, profile_ == QIRProfile::Adaptive);
      },
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

[[nodiscard]] static FailureOr<CompilerProgram>
runDefaultPipelineImpl(CompilerInput program, ProgramFormat output,
                       const TargetEnvironment* environment,
                       std::string_view qcoPipeline,
                       const CompilationOptions& options) {
  if (output == ProgramFormat::IQMJSON &&
      (environment == nullptr ||
       environment->target().nativeOperationsKind() ==
           CompilerTarget::NativeOperations::Kind::Unrestricted)) {
    llvm::errs() << "IQM JSON requires a physical target with explicit native "
                    "IQM operations.\n";
    return failure();
  }
  std::optional<llvm::raw_fd_ostream> profileStream;
  std::optional<tracing::ActionProfiler> profiler;
  std::optional<tracing::ExecutionContext> executionContext;
  std::shared_ptr<MLIRContext> profileContext;
  const llvm::scope_exit clearHandler([&] {
    if (profileContext) {
      profileContext->registerActionHandler(nullptr);
    }
  });
  if (!options.profileActionsTo.empty()) {
    const bool hasActionHandler = std::visit(
        []<typename T>(const T& input) {
          if constexpr (std::is_same_v<T, OpenQASMProgram>) {
            return false;
          } else {
            return input.context()->hasActionHandler();
          }
        },
        program);
    if (hasActionHandler) {
      llvm::errs() << "action profiling requires a context without an existing "
                      "action handler\n";
      return failure();
    }
    std::error_code error;
    profileStream.emplace(options.profileActionsTo, error);
    if (error) {
      llvm::errs() << "failed to open action profile '"
                   << options.profileActionsTo << "': " << error.message()
                   << '\n';
      return failure();
    }
    profiler.emplace(*profileStream);
    executionContext.emplace().registerObserver(&*profiler);
  }
  const auto profileProgram = [&](const Program& input) {
    if (!profiler) {
      return;
    }
    profileContext = input.context();
    profileContext->registerActionHandler(
        [&](llvm::function_ref<void()> transform,
            const tracing::Action& action) {
          (*executionContext)(transform, action);
        });
  };
  if ((output == ProgramFormat::QCImport || output == ProgramFormat::QCO) &&
      qcoPipeline != "mqt-qco-default") {
    llvm::errs() << "a custom QCO pass pipeline cannot be used with an output "
                    "that stops before QCO optimization.\n";
    return failure();
  }
  if (output == ProgramFormat::QCImport) {
    if (std::holds_alternative<QCProgram>(program)) {
      return CompilerProgram(std::move(std::get<QCProgram>(program)));
    }
    if (std::holds_alternative<OpenQASMProgram>(program)) {
      auto qc = QCProgram::fromOpenQASMString(
          std::get<OpenQASMProgram>(program).source());
      if (succeeded(qc)) {
        return CompilerProgram(std::move(*qc));
      }
    }
    llvm::errs() << "QCImport output is only available for QC or OpenQASM "
                    "input.\n";
    return failure();
  }

  auto qco = std::visit(
      // Every consuming branch below explicitly forwards the value.
      // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
      [&]<typename T>(T&& value) -> FailureOr<QCOProgram> {
        using ProgramType = std::remove_cvref_t<T>;
        if constexpr (std::is_same_v<ProgramType, QCOProgram>) {
          profileProgram(value);
          if (failed(qco::verifyLinearity(value.module()))) {
            return failure();
          }
          return std::forward<T>(value);
        } else if constexpr (std::is_same_v<ProgramType, OpenQASMProgram>) {
          auto qc = QCProgram::fromOpenQASMString(value.source());
          if (failed(qc)) {
            return failure();
          }
          profileProgram(*qc);
          return std::move(*qc).intoQCO();
        } else {
          profileProgram(value);
          return std::forward<T>(value).intoQCO();
        }
      },
      std::move(program));
  if (failed(qco)) {
    return failure();
  }
  if (output == ProgramFormat::QCO) {
    return CompilerProgram(std::move(*qco));
  }

  const bool prepareForQIR =
      output == ProgramFormat::QIRBase || output == ProgramFormat::QIRAdaptive;
  const bool defaultPipeline = qcoPipeline == "mqt-qco-default";
  if (environment == nullptr && !defaultPipeline && prepareForQIR &&
      failed(runQCOTransformPasses(
          qco->module(),
          [](OpPassManager& pm) { pm.addPass(createInlinerPass()); },
          "failed to inline QCO calls", options))) {
    return failure();
  }

  if (environment != nullptr) {
    if (failed(qco->compileForTarget(*environment, options))) {
      return failure();
    }
  } else if (defaultPipeline) {
    if (failed(runQCOTransformPasses(
            qco->module(),
            [prepareForQIR](OpPassManager& pm) {
              if (prepareForQIR) {
                pm.addPass(createInlinerPass());
              }
              populateQCOCleanupPipeline(pm);
              populateDefaultQCOOptimizationPipeline(pm);
              populateQCOCleanupPipeline(pm);
            },
            "failed to optimize the QCO program", options))) {
      return failure();
    }
  } else {
    // Arbitrary passes require verification and layout invalidation between
    // stages, even if a later cleanup could repair their output.
    if (failed(qco->cleanup()) ||
        failed(qco->runPassPipeline(qcoPipeline, options)) ||
        failed(qco->cleanup())) {
      return failure();
    }
  }
  if (output == ProgramFormat::QCOOptimized) {
    return CompilerProgram(std::move(*qco));
  }

  if (output == ProgramFormat::Jeff) {
    auto jeff = std::move(*qco).intoJeff();
    if (failed(jeff) || failed(jeff->cleanup())) {
      return failure();
    }
    return CompilerProgram(std::move(*jeff));
  }

  auto qc = std::move(*qco).intoQC();
  if (failed(qc)) {
    return failure();
  }
  if (output == ProgramFormat::QC) {
    // Export cleanup can change resources and invalidate routing metadata.
    return CompilerProgram(std::move(*qc));
  }
  if (output == ProgramFormat::OpenQASM3) {
    auto source = std::move(*qc).intoOpenQASM3();
    if (failed(source)) {
      return failure();
    }
    return CompilerProgram(std::move(*source));
  }
  if (failed(qc->cleanup())) {
    return failure();
  }

  const auto profile = output == ProgramFormat::QIRAdaptive
                           ? QIRProfile::Adaptive
                           : QIRProfile::Base;
  auto qir = std::move(*qc).intoQIR(profile);
  if (failed(qir)) {
    return failure();
  }
  if (output == ProgramFormat::IQMJSON) {
    llvm::LLVMContext context;
    auto llvmModule = qir::translateQIRToLLVM(qir->module(), context);
    if (!llvmModule) {
      return failure();
    }
    auto source = qir::exportIQMJSON(*llvmModule, environment->target());
    if (!source) {
      qir->module().emitError() << llvm::toString(source.takeError());
      return failure();
    }
    return CompilerProgram(IQMJSONProgram(std::move(*source)));
  }
  return CompilerProgram(std::move(*qir));
}

FailureOr<CompilerProgram>
runDefaultPipeline(CompilerInput program, ProgramFormat output,
                   std::string_view qcoPipeline,
                   const CompilationOptions& options) {
  return runDefaultPipelineImpl(std::move(program), output, nullptr,
                                qcoPipeline, options);
}

FailureOr<CompilerProgram>
runDefaultPipeline(CompilerInput program, ProgramFormat output,
                   const TargetEnvironment& environment,
                   const CompilationOptions& options) {
  if (output == ProgramFormat::QCImport || output == ProgramFormat::QCO) {
    llvm::errs()
        << "QCImport and raw QCO output stop before target compilation.\n";
    return failure();
  }
  return runDefaultPipelineImpl(std::move(program), output, &environment,
                                "mqt-qco-default", options);
}

FailureOr<CompilerProgram>
runDefaultPipeline(CompilerInput program, const TargetEnvironment& environment,
                   const CompilationOptions& options) {
  auto output = environment.payloadSpecification().compilerOutput();
  if (!output) {
    llvm::errs() << llvm::toString(output.takeError()) << '\n';
    return failure();
  }
  return runDefaultPipeline(std::move(program), *output, environment, options);
}

} // namespace mlir
