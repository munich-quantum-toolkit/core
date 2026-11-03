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
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"
#include "mqt/Dialect/QCO/Utils/LinearityUtils.h"
#include "mqt/Support/Passes.h"
#include "mqt/Target/IQM/Export.h"
#include "mqt/Target/OpenQASM/TranslateQCToOpenQASM.h"
#include "mqt/Target/QIR/Export.h"

#include "Diagnostics.h"

#include "capnp/common.h"
#include "jeff/Translation/Deserialize.hpp"
#include "jeff/Translation/Serialize.hpp"
#include "kj/array.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLFunctionalExtras.h"
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
#include <span>
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
  if (failed(runWithPassManager(
          *storage.mod,
          [](OpPassManager& pm) {
            pm.addPass(mqt::createSimplifyClassicalControl());
            pm.addPass(mqt::createPromoteClassicalMemory());
          },
          "failed to prepare QC for OpenQASM export"))) {
    return failure();
  }
  auto source = qc::translateQCToOpenQASM3(std::move(storage.mod));
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
  auto source = std::move(*this);
  if (failed(source.normalizeGlobalPhases()) ||
      failed(runWithPassManager(
          source.module(),
          [](OpPassManager& pm) { pm.addPass(mqt::createDropGlobalPhases()); },
          "failed to drop global phases for IQM JSON"))) {
    return failure();
  }
  auto qir = std::move(source).intoQIR(QIRProfile::Base);
  if (failed(qir)) {
    return failure();
  }
  return exportIQMJSON(*qir, target);
}

FailureOr<OpenQASMProgram> QCProgram::intoOpenQASM2() && {
  auto storage = std::move(*this).releaseStorage();
  auto source = qc::translateQCToOpenQASM2(std::move(storage.mod));
  if (failed(source)) {
    return failure();
  }
  return OpenQASMProgram(std::move(*source));
}

FailureOr<QIRProgram> QCProgram::intoQIR(QIRProfile profile) && {
  auto storage = std::move(*this).releaseStorage();
  auto moduleOp = *storage.mod;
  if (failed(runWithPassManager(
          moduleOp,
          [profile](OpPassManager& pm) {
            pm.addPass(mqt::createLowerQuantumModifiers());
            if (profile == QIRProfile::Adaptive) {
              pm.addPass(mqt::createPromoteClassicalMemory());
              pm.addPass(createQCToQIRAdaptive());
            } else {
              pm.addPass(createQCToQIRBase());
            }
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

Expected<JeffProgram> JeffProgram::fromMessage(::jeff::Module::Reader module) {
  auto context = createCompilerContext();
  CompilerDiagnostics diagnostics(context.get());
  auto mod = deserialize(context.get(), module);
  if (!mod) {
    return diagnostics.takeError();
  }
  return JeffProgram({.context = context, .mod = std::move(mod)});
}

Expected<JeffProgram> JeffProgram::fromBytes(std::span<const std::byte> bytes) {
  if (bytes.size() % sizeof(capnp::word) != 0U) {
    return createStringError(
        llvm::inconvertibleErrorCode(),
        "jeff data size must be a multiple of the Cap'n Proto word size");
  }

  auto words = kj::heapArray<capnp::word>(bytes.size() / sizeof(capnp::word));
  std::memcpy(words.begin(), bytes.data(), bytes.size());

  auto context = createCompilerContext();
  CompilerDiagnostics diagnostics(context.get());
  auto mod = deserialize(context.get(), words.asPtr());
  if (!mod) {
    emitError(UnknownLoc::get(context.get()),
              "failed to deserialize jeff bytes");
    return diagnostics.takeError();
  }
  return JeffProgram({.context = context, .mod = std::move(mod)});
}

Expected<JeffProgram> JeffProgram::fromFile(const std::filesystem::path& path) {
  auto context = createCompilerContext();
  CompilerDiagnostics diagnostics(context.get());
  auto mod = deserializeFromFile(context.get(), path.string());
  if (!mod) {
    emitError(UnknownLoc::get(context.get()))
        << "failed to deserialize jeff file '" << path.string() << "'";
    return diagnostics.takeError();
  }
  return JeffProgram({.context = context, .mod = std::move(mod)});
}

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
  if (output == ProgramFormat::QCImport) {
    if (std::holds_alternative<QCProgram>(program)) {
      return CompilerProgram(std::move(std::get<QCProgram>(program)));
    }
    std::visit(
        [](const auto& value) {
          if constexpr (std::is_base_of_v<
                            Program, std::remove_cvref_t<decltype(value)>>) {
            value.module().emitError(
                "QCImport output is only available for QC or OpenQASM input");
          }
        },
        program);
    return failure();
  }

  auto qco = std::visit(
      // Every consuming branch below explicitly forwards the value.
      // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
      []<typename T>(T&& value) -> FailureOr<QCOProgram> {
        using ProgramType = std::remove_cvref_t<T>;
        if constexpr (std::is_same_v<ProgramType, QCOProgram>) {
          if (failed(qco::verifyLinearity(value.module()))) {
            return failure();
          }
          return std::forward<T>(value);
        } else if constexpr (std::is_same_v<ProgramType, OpenQASMProgram>) {
          llvm_unreachable(
              "OpenQASM inputs are imported before running passes");
        } else {
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
  if (output == ProgramFormat::OpenQASM3 ||
      output == ProgramFormat::OpenQASM2) {
    if (failed(runWithPassManager(qc->module(), populateQCExportPipeline,
                                  "failed to clean up compiled QC"))) {
      return failure();
    }
    if (output == ProgramFormat::OpenQASM2 &&
        failed(runWithPassManager(
            qc->module(),
            [&](OpPassManager& pm) {
              const auto payload =
                  llvm::cantFail(PayloadSpecification::forProgramFormat(
                      ProgramFormat::OpenQASM2));
              populateOpenQASM2PreparationPipeline(
                  pm, payload.materialize(*qc->module().getContext()));
            },
            "failed to lower compiled QC for OpenQASM 2", options))) {
      return failure();
    }
    auto source = output == ProgramFormat::OpenQASM2
                      ? std::move(*qc).intoOpenQASM2()
                      : std::move(*qc).intoOpenQASM3();
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
  if (failed(runWithPassManager(
          qc->module(),
          [&](OpPassManager& pm) {
            const auto payload =
                llvm::cantFail(PayloadSpecification::forProgramFormat(
                    profile == QIRProfile::Adaptive ? ProgramFormat::QIRAdaptive
                                                    : ProgramFormat::QIRBase));
            populateQIRPreparationPipeline(
                pm, payload.materialize(*qc->module().getContext()));
          },
          "failed to prepare compiled QC for QIR", options))) {
    return failure();
  }
  auto qir = std::move(*qc).intoQIR(profile);
  if (failed(qir) || failed(qir->cleanup())) {
    return failure();
  }
  if (output == ProgramFormat::IQMJSON) {
    auto source = exportIQMJSON(*qir, environment->target());
    if (failed(source)) {
      return failure();
    }
    return CompilerProgram(std::move(*source));
  }
  return CompilerProgram(std::move(*qir));
}

static Expected<CompilerProgram>
runNativePipeline(CompilerInput program, ProgramFormat output,
                  const TargetEnvironment* environment,
                  std::string_view qcoPipeline,
                  const CompilationOptions& options) {
  if (output == ProgramFormat::IQMJSON &&
      (environment == nullptr ||
       environment->target().nativeOperationsKind() ==
           CompilerTarget::NativeOperations::Kind::Unrestricted)) {
    return createStringError(std::errc::invalid_argument,
                             "IQM JSON requires a physical target with "
                             "explicit native IQM operations");
  }
  if ((output == ProgramFormat::QCImport || output == ProgramFormat::QCO) &&
      qcoPipeline != "mqt-qco-default") {
    return createStringError(
        std::errc::invalid_argument,
        "a custom QCO pass pipeline cannot be used with an output that stops "
        "before QCO optimization");
  }
  if (auto* source = std::get_if<OpenQASMProgram>(&program)) {
    auto qc = QCProgram::fromOpenQASMString(source->source());
    if (!qc) {
      return qc.takeError();
    }
    program = std::move(*qc);
  }
  const auto context = std::visit(
      [](const auto& value) -> std::shared_ptr<MLIRContext> {
        if constexpr (std::is_base_of_v<Program,
                                        std::remove_cvref_t<decltype(value)>>) {
          return value.context();
        } else {
          llvm_unreachable("OpenQASM inputs were imported above");
        }
      },
      program);
  CompilerDiagnostics diagnostics(context.get());
  auto result = runDefaultPipelineImpl(std::move(program), output, environment,
                                       qcoPipeline, options);
  if (failed(result)) {
    return diagnostics.takeError(
        std::make_error_code(std::errc::invalid_argument));
  }
  return std::move(*result);
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
