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
#include "mlir/Compiler/CompilerPipeline.h"
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
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Support/Passes.h"

#include <capnp/common.h>
#include <jeff/IR/JeffDialect.h>
#include <jeff/Translation/Deserialize.hpp>
#include <jeff/Translation/Serialize.hpp>
#include <kj/array.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/SMLoc.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/ModuleTranslation.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <variant>

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

[[nodiscard]] static OwningOpRef<ModuleOp>
parseMLIRString(MLIRContext* context, const std::string& source) {
  auto module = parseSourceString<ModuleOp>(StringRef(source), context);
  if (!module) {
    throw std::runtime_error("Failed to parse MLIR source string.");
  }
  return module;
}

[[nodiscard]] static llvm::SourceMgr
openSourceMgr(const std::filesystem::path& path) {
  std::string errorMessage;
  auto file = openInputFile(path.string(), &errorMessage);
  if (!file) {
    throw std::runtime_error("Failed to load file '" + path.string() +
                             "': " + errorMessage);
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  return sourceMgr;
}

[[nodiscard]] static OwningOpRef<ModuleOp>
parseMLIRFile(MLIRContext* context, const std::filesystem::path& path) {
  const auto sourceMgr = openSourceMgr(path);
  auto module = parseSourceFile<ModuleOp>(sourceMgr, context);
  if (!module) {
    throw std::runtime_error("Failed to parse MLIR file '" + path.string() +
                             "'.");
  }
  return module;
}

template <class PopulatePasses>
static void runPasses(ModuleOp module, PopulatePasses&& populatePasses,
                      const StringRef failureMessage) {
  PassManager pm(module.getContext());
  std::forward<PopulatePasses>(populatePasses)(pm);
  if (failed(pm.run(module))) {
    throw std::runtime_error(failureMessage.str());
  }
}

Program::Program(Storage storage) : storage_(std::move(storage)) {}

bool Program::isValid() const noexcept {
  return static_cast<bool>(storage_.module);
}

ModuleOp Program::module() const {
  if (!storage_.module) {
    throw std::runtime_error("This program was consumed by a conversion.");
  }
  return *storage_.module;
}

std::string Program::str() const { return captureIR(module()); }

Program::Storage Program::cloneStorage() const {
  const auto cloned = cast<ModuleOp>(module()->clone());
  return {.context = storage_.context, .module = OwningOpRef<ModuleOp>(cloned)};
}

Program::Storage Program::releaseStorage() && {
  if (!storage_.module) {
    throw std::runtime_error("This program was already consumed.");
  }
  return {.context = std::move(storage_.context),
          .module = std::move(storage_.module)};
}

QCProgram QCProgram::fromMLIRString(const std::string& source) {
  auto context = createCompilerContext();
  return QCProgram(
      {.context = context, .module = parseMLIRString(context.get(), source)});
}

QCProgram QCProgram::fromMLIRFile(const std::filesystem::path& path) {
  auto context = createCompilerContext();
  return QCProgram(
      {.context = context, .module = parseMLIRFile(context.get(), path)});
}

QCProgram QCProgram::fromQASMString(const std::string& source) {
  auto context = createCompilerContext();
  auto module = qc::translateQASM3ToQC(source, context.get());
  if (!module) {
    throw std::runtime_error("Failed to translate OpenQASM 3 source to QC.");
  }
  return QCProgram(
      {.context = std::move(context), .module = std::move(module)});
}

QCProgram QCProgram::fromQASMFile(const std::filesystem::path& path) {
  auto context = createCompilerContext();
  auto sourceMgr = openSourceMgr(path);
  auto module = qc::translateQASM3ToQC(sourceMgr, context.get());
  if (!module) {
    throw std::runtime_error("Failed to translate OpenQASM 3 file '" +
                             path.string() + "' to QC.");
  }
  return QCProgram(
      {.context = std::move(context), .module = std::move(module)});
}

QCProgram
QCProgram::fromQuantumComputation(const ::qc::QuantumComputation& computation) {
  auto context = createCompilerContext();
  auto module = translateQuantumComputationToQC(context.get(), computation);
  if (!module) {
    throw std::runtime_error("Failed to translate QuantumComputation to QC.");
  }
  return QCProgram(
      {.context = std::move(context), .module = std::move(module)});
}

QCProgram QCProgram::copy() const { return QCProgram(cloneStorage()); }

void QCProgram::cleanup() {
  runPasses(module(), populateQCCleanupPipeline,
            "Failed to run the QC cleanup pipeline.");
}

QCOProgram QCProgram::intoQCO() && {
  runPasses(
      module(), [](PassManager& pm) { pm.addPass(createQCToQCO()); },
      "Failed to convert QC to QCO.");
  return QCOProgram(std::move(*this).releaseStorage());
}

QIRProgram QCProgram::intoQIR(const QIRProfile profile) && {
  runPasses(
      module(),
      [profile](PassManager& pm) {
        if (profile == QIRProfile::Adaptive) {
          pm.addPass(createQCToQIRAdaptive());
        } else {
          pm.addPass(createQCToQIRBase());
        }
      },
      "Failed to convert QC to QIR.");
  auto result = QIRProgram(std::move(*this).releaseStorage(), profile);
  result.cleanup();
  return result;
}

QCOProgram QCOProgram::fromMLIRString(const std::string& source) {
  auto context = createCompilerContext();
  return QCOProgram(
      {.context = context, .module = parseMLIRString(context.get(), source)});
}

QCOProgram QCOProgram::fromMLIRFile(const std::filesystem::path& path) {
  auto context = createCompilerContext();
  return QCOProgram(
      {.context = context, .module = parseMLIRFile(context.get(), path)});
}

QCOProgram QCOProgram::copy() const { return QCOProgram(cloneStorage()); }

void QCOProgram::cleanup() {
  runPasses(module(), populateQCOCleanupPipeline,
            "Failed to run the QCO cleanup pipeline.");
}

void QCOProgram::optimize(const bool mergeSingleQubitRotations,
                          const bool enableHadamardLifting) {
  runPasses(
      module(),
      [mergeSingleQubitRotations, enableHadamardLifting](PassManager& pm) {
        if (mergeSingleQubitRotations) {
          pm.addPass(qco::createMergeSingleQubitRotationGates());
        }
        if (enableHadamardLifting) {
          pm.addPass(qco::createHadamardLifting());
        }
      },
      "Failed to run QCO optimization passes.");
}

QCProgram QCOProgram::intoQC() && {
  runPasses(
      module(), [](PassManager& pm) { pm.addPass(createQCOToQC()); },
      "Failed to convert QCO to QC.");
  return QCProgram(std::move(*this).releaseStorage());
}

JeffProgram QCOProgram::intoJeff() && {
  runPasses(
      module(), [](PassManager& pm) { pm.addPass(createQCOToJeff()); },
      "Failed to convert QCO to Jeff.");
  return JeffProgram(std::move(*this).releaseStorage());
}

JeffProgram JeffProgram::fromFile(const std::filesystem::path& path) {
  auto context = createCompilerContext();
  auto module = deserializeFromFile(context.get(), path.string());
  if (!module) {
    throw std::runtime_error("Failed to deserialize Jeff file '" +
                             path.string() + "'.");
  }
  return JeffProgram(
      {.context = std::move(context), .module = std::move(module)});
}

JeffProgram JeffProgram::fromBytes(const std::string& bytes) {
  if (bytes.size() % sizeof(capnp::word) != 0U) {
    throw std::invalid_argument(
        "Jeff data size must be a multiple of the Cap'n Proto word size.");
  }

  auto words = kj::heapArray<capnp::word>(bytes.size() / sizeof(capnp::word));
  std::memcpy(words.begin(), bytes.data(), bytes.size());

  auto context = createCompilerContext();
  auto module = deserialize(context.get(), words.asPtr());
  if (!module) {
    throw std::runtime_error("Failed to deserialize Jeff bytes.");
  }
  return JeffProgram(
      {.context = std::move(context), .module = std::move(module)});
}

JeffProgram JeffProgram::copy() const { return JeffProgram(cloneStorage()); }

void JeffProgram::cleanup() {
  runPasses(module(), populateJeffCleanupPipeline,
            "Failed to run the Jeff cleanup pipeline.");
}

std::string JeffProgram::toBytes() const {
  const auto serialized = serialize(module());
  const auto bytes = serialized.asBytes();
  return {reinterpret_cast<const char*>(bytes.begin()), bytes.size()};
}

void JeffProgram::write(const std::filesystem::path& path) const {
  const auto bytes = toBytes();
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("Failed to open output file '" + path.string() +
                             "'.");
  }
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!output) {
    throw std::runtime_error("Failed to write output file '" + path.string() +
                             "'.");
  }
}

QCOProgram JeffProgram::intoQCO() && {
  runPasses(
      module(), [](PassManager& pm) { pm.addPass(createJeffToQCO()); },
      "Failed to convert Jeff to QCO.");
  return QCOProgram(std::move(*this).releaseStorage());
}

QIRProgram::QIRProgram(Storage storage, const QIRProfile profile)
    : Program(std::move(storage)), profile_(profile) {}

QIRProgram QIRProgram::copy() const { return {cloneStorage(), profile_}; }

void QIRProgram::cleanup() {
  runPasses(
      module(),
      [this](PassManager& pm) {
        populateQIRCleanupPipeline(pm, profile_ == QIRProfile::Adaptive);
      },
      "Failed to run the QIR cleanup pipeline.");
}

QIRProfile QIRProgram::profile() const noexcept { return profile_; }

std::string QIRProgram::llvmIR() const {
  llvm::LLVMContext context;
  auto llvmModule = translateModuleToLLVMIR(module(), context);
  if (!llvmModule) {
    throw std::runtime_error("Failed to translate QIR MLIR to LLVM IR.");
  }
  std::string result;
  llvm::raw_string_ostream stream(result);
  llvmModule->print(stream, nullptr);
  return result;
}

CompilerProgram runDefaultPipeline(CompilerProgram&& program,
                                   const ProgramFormat output,
                                   const QuantumCompilerConfig& config) {
  if (output == ProgramFormat::QCImport) {
    if (!std::holds_alternative<QCProgram>(program)) {
      throw std::invalid_argument(
          "QCImport output is only available for QC frontend input.");
    }
    return std::move(std::get<QCProgram>(program));
  }

  // Jeff is deserialized into QCO before entering the shared pipeline, so it
  // must skip the QC-to-QCO conversion just like native QCO input.
  const auto inputDialect = std::holds_alternative<QCOProgram>(program) ||
                                    std::holds_alternative<JeffProgram>(program)
                                ? PipelineDialect::QCO
                                : PipelineDialect::QC;
  Program::Storage storage = std::visit(
      []<typename T>(T&& value) -> Program::Storage {
        using ProgramType = std::remove_cvref_t<T>;
        if constexpr (std::is_same_v<ProgramType, JeffProgram>) {
          return std::move(std::forward<T>(value).intoQCO()).releaseStorage();
        } else if constexpr (std::is_same_v<ProgramType, QIRProgram>) {
          throw std::invalid_argument(
              "QIR programs cannot be used as compiler pipeline input.");
        } else {
          return std::forward<T>(value).releaseStorage();
        }
      },
      std::move(program));

  auto pipelineConfig = config;
  pipelineConfig.convertToQIRBase = output == ProgramFormat::QIRBase;
  pipelineConfig.convertToQIRAdaptive = output == ProgramFormat::QIRAdaptive;

  auto target = PipelineDialect::QC;
  switch (output) {
  case ProgramFormat::QCO:
    target = PipelineDialect::QCO;
    break;
  case ProgramFormat::Jeff:
    target = PipelineDialect::Jeff;
    break;
  case ProgramFormat::QIRBase:
  case ProgramFormat::QIRAdaptive:
    target = PipelineDialect::QIR;
    break;
  case ProgramFormat::QCImport:
  case ProgramFormat::QC:
    break;
  }
  if (const QuantumCompilerPipeline pipeline(pipelineConfig);
      failed(pipeline.run(*storage.module, inputDialect, target))) {
    throw std::runtime_error("Failed to run the default compiler pipeline.");
  }

  if (target == PipelineDialect::QCO) {
    return QCOProgram(std::move(storage));
  }
  if (target == PipelineDialect::Jeff) {
    return JeffProgram(std::move(storage));
  }
  if (target == PipelineDialect::QIR) {
    const auto profile = output == ProgramFormat::QIRAdaptive
                             ? QIRProfile::Adaptive
                             : QIRProfile::Base;
    return QIRProgram(std::move(storage), profile);
  }
  return QCProgram(std::move(storage));
}

} // namespace mlir
