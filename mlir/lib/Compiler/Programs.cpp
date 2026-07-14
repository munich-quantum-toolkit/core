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

#include "ir/QuantumComputation.hpp"
#include "mlir/Conversion/JeffToQCO/JeffToQCO.h"
#include "mlir/Conversion/QCOToJeff/QCOToJeff.h"
#include "mlir/Conversion/QCOToQC/QCOToQC.h"
#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Conversion/QCToQIR/QIRAdaptive/QCToQIRAdaptive.h"
#include "mlir/Conversion/QCToQIR/QIRBase/QCToQIRBase.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h"
#include "mlir/Dialect/QC/Translation/TranslateQuantumComputationToQC.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/Transforms/Mapping/Mapping.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Support/Passes.h"

#include <capnp/common.h>
#include <jeff/IR/JeffDialect.h>
#include <jeff/Translation/Deserialize.hpp>
#include <jeff/Translation/Serialize.hpp>
#include <kj/array.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/ModuleTranslation.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
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

[[nodiscard]] static std::shared_ptr<MLIRContext> createCompilerContext() {
  DialectRegistry registry;
  registry.insert<qc::QCDialect, qco::QCODialect, qtensor::QTensorDialect,
                  arith::ArithDialect, cf::ControlFlowDialect,
                  func::FuncDialect, scf::SCFDialect, LLVM::LLVMDialect,
                  memref::MemRefDialect, jeff::JeffDialect>();
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);

  auto context = std::make_shared<MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

[[nodiscard]] static FailureOr<OwningOpRef<ModuleOp>>
parseMLIRString(MLIRContext* context, const StringRef source) {
  auto module = parseSourceString<ModuleOp>(source, context);
  if (!module) {
    return emitError(UnknownLoc::get(context),
                     "failed to parse MLIR source string");
  }
  return std::move(module);
}

[[nodiscard]] static LogicalResult
openSourceMgr(const std::filesystem::path& path, MLIRContext* context,
              llvm::SourceMgr& sourceMgr) {
  std::string errorMessage;
  auto file = openInputFile(path.string(), &errorMessage);
  if (!file) {
    return emitError(UnknownLoc::get(context))
           << "failed to load file '" << path.string() << "': " << errorMessage;
  }

  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  return success();
}

[[nodiscard]] static FailureOr<OwningOpRef<ModuleOp>>
parseMLIRFile(MLIRContext* context, const std::filesystem::path& path) {
  llvm::SourceMgr sourceMgr;
  if (failed(openSourceMgr(path, context, sourceMgr))) {
    return failure();
  }
  auto module = parseSourceFile<ModuleOp>(sourceMgr, context);
  if (!module) {
    return emitError(UnknownLoc::get(context))
           << "failed to parse MLIR file '" << path.string() << "'";
  }
  return std::move(module);
}

template <class PopulatePasses>
[[nodiscard]] static LogicalResult runPasses(ModuleOp module,
                                             PopulatePasses&& populatePasses,
                                             const StringRef failureMessage) {
  PassManager pm(module.getContext());
  std::forward<PopulatePasses>(populatePasses)(pm);
  if (failed(pm.run(module))) {
    return module.emitError(failureMessage);
  }
  return success();
}

Program::Program(Storage storage) : storage_(std::move(storage)) {}

bool Program::isValid() const noexcept {
  return static_cast<bool>(storage_.module);
}

ModuleOp Program::module() const {
  assert(storage_.module && "cannot use a consumed compiler program");
  return *storage_.module;
}

std::string Program::str() const {
  std::string result;
  llvm::raw_string_ostream stream(result);
  module().print(stream);
  return result;
}

Program::Storage Program::cloneStorage() const {
  const auto cloned = cast<ModuleOp>(module()->clone());
  return {.context = storage_.context, .module = OwningOpRef<ModuleOp>(cloned)};
}

Program::Storage Program::releaseStorage() && {
  assert(storage_.module && "compiler program was already consumed");
  return {.context = std::move(storage_.context),
          .module = std::move(storage_.module)};
}

std::optional<QCProgram>
QCProgram::fromMLIRString(const std::string_view source) {
  auto context = createCompilerContext();
  auto module = parseMLIRString(context.get(), source);
  if (failed(module)) {
    return std::nullopt;
  }
  return QCProgram({.context = context, .module = std::move(*module)});
}

std::optional<QCProgram>
QCProgram::fromMLIRFile(const std::filesystem::path& path) {
  auto context = createCompilerContext();
  auto module = parseMLIRFile(context.get(), path);
  if (failed(module)) {
    return std::nullopt;
  }
  return QCProgram({.context = context, .module = std::move(*module)});
}

std::optional<QCProgram>
QCProgram::fromQASMString(const std::string_view source) {
  auto context = createCompilerContext();
  auto module = qc::translateQASM3ToQC(source, context.get());
  if (!module) {
    emitError(UnknownLoc::get(context.get()),
              "failed to translate OpenQASM 3 source to QC");
    return std::nullopt;
  }
  return QCProgram(
      {.context = std::move(context), .module = std::move(module)});
}

std::optional<QCProgram>
QCProgram::fromQASMFile(const std::filesystem::path& path) {
  auto context = createCompilerContext();
  llvm::SourceMgr sourceMgr;
  if (failed(openSourceMgr(path, context.get(), sourceMgr))) {
    return std::nullopt;
  }
  auto module = qc::translateQASM3ToQC(sourceMgr, context.get());
  if (!module) {
    emitError(UnknownLoc::get(context.get()))
        << "failed to translate OpenQASM 3 file '" << path.string()
        << "' to QC";
    return std::nullopt;
  }
  return QCProgram(
      {.context = std::move(context), .module = std::move(module)});
}

std::optional<QCProgram>
QCProgram::fromQuantumComputation(const ::qc::QuantumComputation& computation) {
  auto context = createCompilerContext();
  auto module = translateQuantumComputationToQC(context.get(), computation);
  if (!module) {
    emitError(UnknownLoc::get(context.get()),
              "failed to translate QuantumComputation to QC");
    return std::nullopt;
  }
  return QCProgram(
      {.context = std::move(context), .module = std::move(module)});
}

QCProgram QCProgram::copy() const { return QCProgram(cloneStorage()); }

bool QCProgram::cleanup() {
  return succeeded(runPasses(module(), populateQCCleanupPipeline,
                             "failed to run the QC cleanup pipeline"));
}

std::optional<QCOProgram> QCProgram::intoQCO() && {
  if (failed(runPasses(
          module(), [](OpPassManager& pm) { pm.addPass(createQCToQCO()); },
          "failed to convert QC to QCO"))) {
    return std::nullopt;
  }
  return QCOProgram(std::move(*this).releaseStorage());
}

std::optional<QIRProgram> QCProgram::intoQIR(const QIRProfile profile) && {
  if (failed(runPasses(
          module(),
          [profile](OpPassManager& pm) {
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

std::optional<QCOProgram>
QCOProgram::fromMLIRString(const std::string_view source) {
  auto context = createCompilerContext();
  auto module = parseMLIRString(context.get(), source);
  if (failed(module)) {
    return std::nullopt;
  }
  return QCOProgram({.context = context, .module = std::move(*module)});
}

std::optional<QCOProgram>
QCOProgram::fromMLIRFile(const std::filesystem::path& path) {
  auto context = createCompilerContext();
  auto module = parseMLIRFile(context.get(), path);
  if (failed(module)) {
    return std::nullopt;
  }
  return QCOProgram({.context = context, .module = std::move(*module)});
}

QCOProgram QCOProgram::copy() const { return QCOProgram(cloneStorage()); }

bool QCOProgram::cleanup() {
  return succeeded(runPasses(module(), populateQCOCleanupPipeline,
                             "failed to run the QCO cleanup pipeline"));
}

bool QCOProgram::runPassPipeline(const std::string_view pipeline,
                                 const bool enableTiming,
                                 const bool enableStatistics) {
  return succeeded(
      ::runPassPipeline(module(), pipeline, enableTiming, enableStatistics));
}

bool QCOProgram::mergeSingleQubitRotationGates() {
  return succeeded(runPasses(
      module(),
      [](OpPassManager& pm) {
        pm.addPass(qco::createMergeSingleQubitRotationGates());
      },
      "failed to merge single-qubit rotation gates"));
}

bool QCOProgram::fuseSingleQubitUnitaryRuns(const std::string_view basis) {
  qco::FuseSingleQubitUnitaryRunsOptions options;
  options.basis = basis;
  return succeeded(runPasses(
      module(),
      [&options](OpPassManager& pm) {
        pm.addPass(qco::createFuseSingleQubitUnitaryRuns(options));
      },
      "failed to fuse single-qubit unitary runs"));
}

bool QCOProgram::unrollQuantumLoops(const int64_t factor) {
  qco::QuantumLoopUnrollOptions options;
  options.unrollFactor = factor;
  return succeeded(runPasses(
      module(),
      [&options](OpPassManager& pm) {
        pm.addNestedPass<func::FuncOp>(qco::createQuantumLoopUnroll(options));
      },
      "failed to unroll quantum loops"));
}

bool QCOProgram::liftHadamards() {
  return succeeded(runPasses(
      module(),
      [](OpPassManager& pm) { pm.addPass(qco::createHadamardLifting()); },
      "failed to lift Hadamard gates"));
}

bool QCOProgram::placeAndRoute(
    const std::span<const std::pair<std::size_t, std::size_t>> coupling,
    const std::size_t nlookahead, const float alpha, const float lambda,
    const std::size_t niterations, const std::size_t ntrials,
    const std::size_t seed) {
  llvm::DenseSet<std::pair<std::size_t, std::size_t>> couplingSet;
  couplingSet.insert(coupling.begin(), coupling.end());
  qco::MappingPassOptions options;
  options.nlookahead = nlookahead;
  options.alpha = alpha;
  options.lambda = lambda;
  options.niterations = niterations;
  options.ntrials = ntrials;
  options.seed = seed;
  return succeeded(runPasses(
      module(),
      [&couplingSet, &options](OpPassManager& pm) {
        pm.addPass(qco::createMappingPass(couplingSet, options));
      },
      "failed to place and route the QCO program"));
}

std::optional<QCProgram> QCOProgram::intoQC() && {
  if (failed(runPasses(
          module(), [](OpPassManager& pm) { pm.addPass(createQCOToQC()); },
          "failed to convert QCO to QC"))) {
    return std::nullopt;
  }
  return QCProgram(std::move(*this).releaseStorage());
}

std::optional<JeffProgram> QCOProgram::intoJeff() && {
  if (failed(runPasses(
          module(), [](OpPassManager& pm) { pm.addPass(createQCOToJeff()); },
          "failed to convert QCO to Jeff"))) {
    return std::nullopt;
  }
  return JeffProgram(std::move(*this).releaseStorage());
}

std::optional<JeffProgram>
JeffProgram::fromFile(const std::filesystem::path& path) {
  auto context = createCompilerContext();
  auto module = deserializeFromFile(context.get(), path.string());
  if (!module) {
    emitError(UnknownLoc::get(context.get()))
        << "failed to deserialize Jeff file '" << path.string() << "'";
    return std::nullopt;
  }
  return JeffProgram(
      {.context = std::move(context), .module = std::move(module)});
}

std::optional<JeffProgram>
JeffProgram::fromBytes(const std::span<const std::byte> bytes) {
  if (bytes.size() % sizeof(capnp::word) != 0U) {
    auto context = createCompilerContext();
    emitError(UnknownLoc::get(context.get()),
              "Jeff data size must be a multiple of the Cap'n Proto word size");
    return std::nullopt;
  }

  auto words = kj::heapArray<capnp::word>(bytes.size() / sizeof(capnp::word));
  std::memcpy(words.begin(), bytes.data(), bytes.size());

  auto context = createCompilerContext();
  auto module = deserialize(context.get(), words.asPtr());
  if (!module) {
    emitError(UnknownLoc::get(context.get()),
              "failed to deserialize Jeff bytes");
    return std::nullopt;
  }
  return JeffProgram(
      {.context = std::move(context), .module = std::move(module)});
}

JeffProgram JeffProgram::copy() const { return JeffProgram(cloneStorage()); }

bool JeffProgram::cleanup() {
  return succeeded(runPasses(module(), populateJeffCleanupPipeline,
                             "failed to run the Jeff cleanup pipeline"));
}

std::vector<std::byte> JeffProgram::toBytes() const {
  const auto serialized = serialize(module());
  const auto bytes = serialized.asBytes();
  std::vector<std::byte> result(bytes.size());
  std::memcpy(result.data(), bytes.begin(), bytes.size());
  return result;
}

bool JeffProgram::write(const std::filesystem::path& path) const {
  const auto bytes = toBytes();
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    module().emitError() << "failed to open output file '" << path.string()
                         << "'";
    return false;
  }
  output.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!output) {
    module().emitError() << "failed to write output file '" << path.string()
                         << "'";
    return false;
  }
  return true;
}

std::optional<QCOProgram> JeffProgram::intoQCO() && {
  if (failed(runPasses(
          module(), [](OpPassManager& pm) { pm.addPass(createJeffToQCO()); },
          "failed to convert Jeff to QCO"))) {
    return std::nullopt;
  }
  return QCOProgram(std::move(*this).releaseStorage());
}

QIRProgram::QIRProgram(Storage storage, const QIRProfile profile)
    : Program(std::move(storage)), profile_(profile) {}

QIRProgram QIRProgram::copy() const { return {cloneStorage(), profile_}; }

bool QIRProgram::cleanup() {
  return succeeded(runPasses(
      module(),
      [this](OpPassManager& pm) {
        populateQIRCleanupPipeline(pm, profile_ == QIRProfile::Adaptive);
      },
      "failed to run the QIR cleanup pipeline"));
}

QIRProfile QIRProgram::profile() const noexcept { return profile_; }

[[nodiscard]] static std::unique_ptr<llvm::Module>
translateToLLVM(ModuleOp module, llvm::LLVMContext& context) {
  auto llvmModule = translateModuleToLLVMIR(module, context);
  if (!llvmModule) {
    module.emitError("failed to translate QIR MLIR to LLVM IR");
  }
  return llvmModule;
}

std::optional<std::string> QIRProgram::llvmIR() const {
  llvm::LLVMContext context;
  auto llvmModule = translateToLLVM(module(), context);
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
  auto llvmModule = translateToLLVM(module(), context);
  if (!llvmModule) {
    return std::nullopt;
  }

  llvm::SmallVector<char> storage;
  llvm::raw_svector_ostream stream(storage);
  llvm::WriteBitcodeToFile(*llvmModule, stream);
  std::vector<std::byte> result(storage.size());
  std::memcpy(result.data(), storage.data(), storage.size());
  return result;
}

bool QIRProgram::writeBitcode(const std::filesystem::path& path) const {
  llvm::LLVMContext context;
  auto llvmModule = translateToLLVM(module(), context);
  if (!llvmModule) {
    return false;
  }

  std::error_code error;
  llvm::raw_fd_ostream stream(path.string(), error, llvm::sys::fs::OF_None);
  if (error) {
    module().emitError() << "failed to open bitcode output file '"
                         << path.string() << "': " << error.message();
    return false;
  }
  llvm::WriteBitcodeToFile(*llvmModule, stream);
  stream.flush();
  if (stream.has_error()) {
    module().emitError() << "failed to write bitcode output file '"
                         << path.string() << "'";
    return false;
  }
  return true;
}

std::optional<CompilerProgram>
runDefaultPipeline(CompilerInput&& program, const ProgramFormat output,
                   const std::string_view qcoPipeline, const bool enableTiming,
                   const bool enableStatistics) {
  if ((output == ProgramFormat::QCImport || output == ProgramFormat::QCO) &&
      qcoPipeline != "mqt-qco-default") {
    llvm::errs() << "a custom QCO pass pipeline cannot be used with an output "
                    "that stops before QCO optimization.\n";
    return std::nullopt;
  }
  if (output == ProgramFormat::QCImport) {
    if (!std::holds_alternative<QCProgram>(program)) {
      llvm::errs() << "QCImport output is only available for QC input.\n";
      return std::nullopt;
    }
    return CompilerProgram(std::move(std::get<QCProgram>(program)));
  }

  auto qco = std::visit(
      []<typename T>(T&& value) -> std::optional<QCOProgram> {
        using ProgramType = std::remove_cvref_t<T>;
        if constexpr (std::is_same_v<ProgramType, QCOProgram>) {
          return std::forward<T>(value);
        } else {
          return std::forward<T>(value).intoQCO();
        }
      },
      std::move(program));
  if (!qco) {
    return std::nullopt;
  }

  if (output == ProgramFormat::QCO) {
    return CompilerProgram(std::move(*qco));
  }

  if (!qco->cleanup() ||
      !qco->runPassPipeline(qcoPipeline, enableTiming, enableStatistics) ||
      !qco->cleanup()) {
    return std::nullopt;
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

  const auto profile = output == ProgramFormat::QIRAdaptive
                           ? QIRProfile::Adaptive
                           : QIRProfile::Base;
  auto qir = std::move(*qc).intoQIR(profile);
  if (!qir) {
    return std::nullopt;
  }
  return CompilerProgram(std::move(*qir));
}

} // namespace mlir
