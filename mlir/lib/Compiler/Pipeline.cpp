/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Compiler/Programs.h"
#include "mlir/Compiler/TargetCompilation.h"
#include "mlir/Compiler/TargetEnvironment.h"
#include "mlir/Conversion/JeffToQCO/JeffToQCO.h"
#include "mlir/Conversion/QCOToJeff/QCOToJeff.h"
#include "mlir/Conversion/QCOToQC/QCOToQC.h"
#include "mlir/Conversion/QCToQIR/QIRAdaptive/QCToQIRAdaptive.h"
#include "mlir/Conversion/QCToQIR/QIRBase/QCToQIRBase.h"
#include "mlir/Dialect/MQT/Transforms/GlobalPhaseNormalization.h"
#include "mlir/Dialect/MQT/Transforms/Passes.h"
#include "mlir/Dialect/QC/Translation/TranslateQCToOpenQASM3.h"
#include "mlir/Dialect/QCO/QCOUtils.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QIR/Utils/QIRUtils.h"
#include "mlir/Support/Passes.h"

#include <capnp/common.h>
#include <jeff/Translation/Deserialize.hpp>
#include <jeff/Translation/Serialize.hpp>
#include <kj/array.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Target/LLVMIR/ModuleTranslation.h>

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

[[nodiscard]] static LogicalResult
runPasses(ModuleOp mod, llvm::function_ref<void(OpPassManager&)> populatePasses,
          StringRef failureMessage, bool enableTiming = false,
          bool enableStatistics = false) {
  PassManager pm(mod.getContext());
  if (enableTiming) {
    pm.enableTiming();
  }
  if (enableStatistics) {
    pm.enableStatistics();
  }
  populatePasses(pm);
  if (failed(pm.run(mod))) {
    return mod.emitError(failureMessage);
  }
  return success();
}

[[nodiscard]] static LogicalResult
runQCOTransformPasses(ModuleOp mod,
                      llvm::function_ref<void(OpPassManager&)> populatePasses,
                      StringRef failureMessage, bool enableTiming = false,
                      bool enableStatistics = false) {
  if (failed(qco::verifyLinearity(mod))) {
    return failure();
  }
  if (failed(runPasses(mod, populatePasses, failureMessage, enableTiming,
                       enableStatistics))) {
    return failure();
  }
  return qco::verifyLinearity(mod);
}

//===----------------------------------------------------------------------===//
// QCProgram
//===----------------------------------------------------------------------===//

bool QCProgram::cleanup() {
  return succeeded(runPasses(mod(), populateQCCleanupPipeline,
                             "failed to run the QC cleanup pipeline"));
}

bool QCProgram::normalizeGlobalPhases() {
  return succeeded(mqt::normalizeGlobalPhases(mod()));
}

std::optional<OpenQASMProgram> QCProgram::toOpenQASM3() const {
  auto cleaned = copy();
  if (failed(runPasses(cleaned.mod(), populateQCExportPipeline,
                       "failed to prepare QC for OpenQASM export"))) {
    return std::nullopt;
  }
  auto source = qc::translateQCToOpenQASM3(cleaned.mod());
  if (failed(source)) {
    return std::nullopt;
  }
  return OpenQASMProgram(std::move(*source));
}

std::optional<QIRProgram> QCProgram::intoQIR(QIRProfile profile) && {
  if (failed(runPasses(
          mod(),
          [profile](OpPassManager& pm) {
            pm.addPass(mqt::createUnrollModifiers());
            if (profile == QIRProfile::Adaptive) {
              pm.addPass(createQCToQIRAdaptive());
            } else {
              pm.addPass(createQCToQIRBase());
            }
          },
          "failed to convert QC to QIR"))) {
    return std::nullopt;
  }
  auto result = QIRProgram(std::move(*this).releaseStorage(), profile);
  if (!result.cleanup()) {
    return std::nullopt;
  }
  return result;
}

//===----------------------------------------------------------------------===//
// QCOProgram
//===----------------------------------------------------------------------===//

bool QCOProgram::cleanup() {
  return succeeded(
      runQCOTransformPasses(mod(), populateQCOCleanupPipeline,
                            "failed to run the QCO cleanup pipeline"));
}

bool QCOProgram::normalizeGlobalPhases() {
  if (!hasValidLinearity()) {
    return false;
  }
  return succeeded(mqt::normalizeGlobalPhases(mod())) && hasValidLinearity();
}

bool QCOProgram::runPassPipeline(std::string_view pipeline, bool enableTiming,
                                 bool enableStatistics) {
  if (!hasValidLinearity()) {
    return false;
  }
  return succeeded(::runPassPipeline(mod(), pipeline, enableTiming,
                                     enableStatistics)) &&
         hasValidLinearity();
}

bool QCOProgram::mergeSingleQubitRotationGates() {
  return succeeded(runQCOTransformPasses(
      mod(),
      [](OpPassManager& pm) {
        pm.addPass(qco::createMergeSingleQubitRotationGates());
      },
      "failed to merge single-qubit rotation gates"));
}

bool QCOProgram::fuseSingleQubitUnitaryRuns(std::string_view basis) {
  qco::FuseSingleQubitUnitaryRunsOptions options;
  options.basis = basis;
  return succeeded(runQCOTransformPasses(
      mod(),
      [&options](OpPassManager& pm) {
        pm.addPass(qco::createFuseSingleQubitUnitaryRuns(options));
      },
      "failed to fuse single-qubit unitary runs"));
}

bool QCOProgram::unrollQuantumLoops(int64_t factor) {
  qco::QuantumLoopUnrollOptions options;
  options.unrollFactor = factor;
  return succeeded(runQCOTransformPasses(
      mod(),
      [&options](OpPassManager& pm) {
        pm.addNestedPass<func::FuncOp>(qco::createQuantumLoopUnroll(options));
      },
      "failed to unroll quantum loops"));
}

bool QCOProgram::liftHadamards() {
  return succeeded(runQCOTransformPasses(
      mod(),
      [](OpPassManager& pm) { pm.addPass(qco::createHadamardLifting()); },
      "failed to lift Hadamard gates"));
}

bool QCOProgram::reuseQubits() {
  return succeeded(runQCOTransformPasses(
      mod(), [](OpPassManager& pm) { pm.addPass(qco::createReuseQubits()); },
      "failed to reuse qubits"));
}

bool QCOProgram::runQubitReusePipeline() {
  return succeeded(
      runQCOTransformPasses(mod(), populateQubitReusePipeline,
                            "failed to run the qubit reuse pipeline"));
}

bool QCOProgram::decomposeMultiControlled(uint64_t minQubits) {
  return succeeded(runQCOTransformPasses(
      mod(),
      [minQubits](OpPassManager& pm) {
        populateDecomposeMultiControlledPipeline(pm, minQubits);
      },
      "failed to decompose multi-controlled gates"));
}

bool QCOProgram::compileForTarget(const TargetEnvironment& environment,
                                  bool enableTiming, bool enableStatistics) {
  attachTargetEnvironment(mod(), environment);
  return succeeded(runQCOTransformPasses(
      mod(),
      [&environment](OpPassManager& pm) {
        populateTargetCompilationPipeline(pm, environment.target());
      },
      "failed to compile the QCO program for the target", enableTiming,
      enableStatistics));
}

std::optional<QCProgram> QCOProgram::intoQC() && {
  if (failed(runQCOTransformPasses(
          mod(), [](OpPassManager& pm) { pm.addPass(createQCOToQC()); },
          "failed to convert QCO to QC"))) {
    return std::nullopt;
  }
  return QCProgram(std::move(*this).releaseStorage());
}

std::optional<JeffProgram> QCOProgram::intoJeff() && {
  if (failed(runQCOTransformPasses(
          mod(),
          [](OpPassManager& pm) {
            pm.addPass(mqt::createUnrollModifiers());
            pm.addPass(createQCOToJeff());
          },
          "failed to convert QCO to jeff"))) {
    return std::nullopt;
  }
  return JeffProgram(std::move(*this).releaseStorage());
}

//===----------------------------------------------------------------------===//
// JeffProgram
//===----------------------------------------------------------------------===//

std::optional<JeffProgram>
JeffProgram::fromBytes(std::span<const std::byte> bytes) {
  if (bytes.size() % sizeof(capnp::word) != 0U) {
    auto context = createCompilerContext();
    emitError(UnknownLoc::get(context.get()),
              "jeff data size must be a multiple of the Cap'n Proto word size");
    return std::nullopt;
  }

  auto words = kj::heapArray<capnp::word>(bytes.size() / sizeof(capnp::word));
  std::memcpy(words.begin(), bytes.data(), bytes.size());

  auto context = createCompilerContext();
  auto mod = deserialize(context.get(), words.asPtr());
  if (!mod) {
    emitError(UnknownLoc::get(context.get()),
              "failed to deserialize jeff bytes");
    return std::nullopt;
  }
  return JeffProgram({.context = std::move(context), .mod = std::move(mod)});
}

std::optional<JeffProgram>
JeffProgram::fromFile(const std::filesystem::path& path) {
  auto context = createCompilerContext();
  auto mod = deserializeFromFile(context.get(), path.string());
  if (!mod) {
    emitError(UnknownLoc::get(context.get()))
        << "failed to deserialize jeff file '" << path.string() << "'";
    return std::nullopt;
  }
  return JeffProgram({.context = std::move(context), .mod = std::move(mod)});
}

JeffProgram JeffProgram::copy() const { return JeffProgram(cloneStorage()); }

bool JeffProgram::cleanup() {
  return succeeded(runPasses(mod(), populateJeffCleanupPipeline,
                             "failed to run the jeff cleanup pipeline"));
}

std::vector<std::byte> JeffProgram::toBytes() const {
  const auto serialized = serialize(mod());
  const auto bytes = serialized.asBytes();
  std::vector<std::byte> result(bytes.size());
  std::memcpy(result.data(), bytes.begin(), bytes.size());
  return result;
}

bool JeffProgram::write(const std::filesystem::path& path) const {
  if (failed(serializeToFile(mod(), path.string()))) {
    mod().emitError() << "failed to write jeff file '" << path.string() << "'";
    return false;
  }
  return true;
}

std::optional<QCOProgram> JeffProgram::intoQCO() && {
  if (failed(runPasses(
          mod(), [](OpPassManager& pm) { pm.addPass(createJeffToQCO()); },
          "failed to convert jeff to QCO"))) {
    return std::nullopt;
  }
  if (failed(qco::verifyLinearity(mod()))) {
    return std::nullopt;
  }
  return QCOProgram(std::move(*this).releaseStorage());
}

//===----------------------------------------------------------------------===//
// QIRProgram
//===----------------------------------------------------------------------===//

QIRProgram::QIRProgram(Storage storage, QIRProfile profile)
    : Program(std::move(storage)), profile_(profile) {}

QIRProgram QIRProgram::copy() const { return {cloneStorage(), profile_}; }

bool QIRProgram::cleanup() {
  return succeeded(runPasses(
      mod(),
      [this](OpPassManager& pm) {
        populateQIRCleanupPipeline(pm, profile_ == QIRProfile::Adaptive);
      },
      "failed to run the QIR cleanup pipeline"));
}

QIRProfile QIRProgram::profile() const noexcept { return profile_; }

[[nodiscard]] static std::unique_ptr<llvm::Module>
translateToLLVM(ModuleOp mod, llvm::LLVMContext& context) {
  auto llvmModule = translateModuleToLLVMIR(mod, context);
  if (!llvmModule) {
    mod.emitError("failed to translate QIR MLIR to LLVM IR");
    return nullptr;
  }
  qir::normalizeQIRModuleFlags(*llvmModule, mod);
  return llvmModule;
}

std::optional<std::string> QIRProgram::llvmIR() const {
  llvm::LLVMContext context;
  auto llvmModule = translateToLLVM(mod(), context);
  if (!llvmModule) {
    return std::nullopt;
  }
  std::string result;
  llvm::raw_string_ostream stream(result);
  llvmModule->print(stream, nullptr);
  return result;
}

std::optional<std::vector<std::byte>> QIRProgram::toBitcode() const {
  llvm::LLVMContext context;
  auto llvmModule = translateToLLVM(mod(), context);
  if (!llvmModule) {
    return std::nullopt;
  }

  SmallVector<char> storage;
  llvm::raw_svector_ostream stream(storage);
  llvm::WriteBitcodeToFile(*llvmModule, stream);
  std::vector<std::byte> result(storage.size());
  std::memcpy(result.data(), storage.data(), storage.size());
  return result;
}

bool QIRProgram::writeBitcode(const std::filesystem::path& path) const {
  llvm::LLVMContext context;
  auto llvmModule = translateToLLVM(mod(), context);
  if (!llvmModule) {
    return false;
  }

  std::error_code error;
  llvm::raw_fd_ostream stream(path.string(), error, llvm::sys::fs::OF_None);
  if (error) {
    mod().emitError() << "failed to open bitcode output file '" << path.string()
                      << "': " << error.message();
    return false;
  }
  llvm::WriteBitcodeToFile(*llvmModule, stream);
  stream.flush();
  if (stream.has_error()) {
    mod().emitError() << "failed to write bitcode file '" << path.string()
                      << "'";
    return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Pipeline
//===----------------------------------------------------------------------===//

[[nodiscard]] static std::optional<CompilerProgram>
runDefaultPipelineImpl(CompilerInput&& program, ProgramFormat output,
                       const TargetEnvironment* environment,
                       std::string_view qcoPipeline, bool enableTiming,
                       bool enableStatistics) {
  if ((output == ProgramFormat::QCImport || output == ProgramFormat::QCO) &&
      qcoPipeline != "mqt-qco-default") {
    llvm::errs() << "a custom QCO pass pipeline cannot be used with an output "
                    "that stops before QCO optimization.\n";
    return std::nullopt;
  }
  if (output == ProgramFormat::QCImport) {
    if (std::holds_alternative<QCProgram>(program)) {
      return CompilerProgram(std::move(std::get<QCProgram>(program)));
    }
    if (std::holds_alternative<OpenQASMProgram>(program)) {
      auto qc = QCProgram::fromQASMString(
          std::get<OpenQASMProgram>(program).source());
      if (qc) {
        return CompilerProgram(std::move(*qc));
      }
    }
    llvm::errs() << "QCImport output is only available for QC or OpenQASM "
                    "input.\n";
    return std::nullopt;
  }

  auto qco = std::visit(
      // Every consuming branch below explicitly forwards the value.
      // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
      []<typename T>(T&& value) -> std::optional<QCOProgram> {
        using ProgramType = std::remove_cvref_t<T>;
        if constexpr (std::is_same_v<ProgramType, QCOProgram>) {
          return std::forward<T>(value);
        } else if constexpr (std::is_same_v<ProgramType, OpenQASMProgram>) {
          auto qc = QCProgram::fromQASMString(value.source());
          if (!qc) {
            return std::nullopt;
          }
          return std::move(*qc).intoQCO();
        } else {
          return std::forward<T>(value).intoQCO();
        }
      },
      std::move(program));
  if (!qco || failed(qco::verifyLinearity(qco->module()))) {
    return std::nullopt;
  }
  if (output == ProgramFormat::QCO) {
    return CompilerProgram(std::move(*qco));
  }

  if (environment != nullptr) {
    if (!qco->compileForTarget(*environment, enableTiming, enableStatistics)) {
      return std::nullopt;
    }
  } else {
    if (!qco->cleanup() ||
        !qco->runPassPipeline(qcoPipeline, enableTiming, enableStatistics) ||
        !qco->cleanup()) {
      return std::nullopt;
    }
  }
  if (output == ProgramFormat::QCOOptimized) {
    return CompilerProgram(std::move(*qco));
  }

  if (output == ProgramFormat::Jeff) {
    auto jeff = std::move(*qco).intoJeff();
    if (!jeff || !jeff->cleanup()) {
      return std::nullopt;
    }
    return CompilerProgram(std::move(*jeff));
  }

  auto qc = std::move(*qco).intoQC();
  if (!qc || !qc->cleanup()) {
    return std::nullopt;
  }
  if (output == ProgramFormat::QC) {
    return CompilerProgram(std::move(*qc));
  }
  if (output == ProgramFormat::OpenQASM3) {
    return qc->toOpenQASM3();
  }

  const auto profile = output == ProgramFormat::QIRAdaptive
                           ? QIRProfile::Adaptive
                           : QIRProfile::Base;
  return std::move(*qc).intoQIR(profile);
}

std::optional<CompilerProgram> runDefaultPipeline(CompilerInput&& program,
                                                  ProgramFormat output,
                                                  std::string_view qcoPipeline,
                                                  bool enableTiming,
                                                  bool enableStatistics) {
  return runDefaultPipelineImpl(std::move(program), output, nullptr,
                                qcoPipeline, enableTiming, enableStatistics);
}

std::optional<CompilerProgram>
runDefaultPipeline(CompilerInput&& program,
                   const TargetEnvironment& environment, bool enableTiming,
                   bool enableStatistics) {
  auto output = environment.payloadSpecification().compilerOutput();
  if (!output) {
    llvm::errs() << llvm::toString(output.takeError()) << '\n';
    return std::nullopt;
  }
  return runDefaultPipelineImpl(std::move(program), *output, &environment,
                                "mqt-qco-default", enableTiming,
                                enableStatistics);
}

} // namespace mlir
