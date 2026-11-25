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

#include "mqt/Conversion/QCToQCO/QCToQCO.h"
#include "mqt/Dialect/CBit/IR/CBitDialect.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QC/IR/QCDialect.h"
#include "mqt/Dialect/QC/IR/QCInterfaces.h"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/Utils/LinearityUtils.h"
#include "mqt/Dialect/QTensor/IR/QTensorDialect.h"
#include "mqt/Support/Passes.h"
#include "mqt/Target/OpenQASM/TranslateOpenQASMToQC.h"

#include "Diagnostics.h"

#include "capnp/common.h"
#include "jeff/IR/JeffDialect.h"
#include "jeff/Translation/Deserialize.hpp"
#include "kj/array.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

using llvm::Error;
using llvm::Expected;

namespace mlir {

static void ensureInlinerExtensions(MLIRContext* context) {
  DialectRegistry registry;
  func::registerInlinerExtension(registry);
  LLVM::registerInlinerInterface(registry);
  context->appendDialectRegistry(registry);
}

std::shared_ptr<MLIRContext> createCompilerContext() {
  DialectRegistry registry;
  registry.insert<cbit::CBitDialect, mqt::MQTDialect, qc::QCDialect,
                  qco::QCODialect, qtensor::QTensorDialect, arith::ArithDialect,
                  cf::ControlFlowDialect, func::FuncDialect, math::MathDialect,
                  scf::SCFDialect, LLVM::LLVMDialect, memref::MemRefDialect,
                  tensor::TensorDialect, jeff::JeffDialect,
                  bufferization::BufferizationDialect>();
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);

  // Retained programs share workers instead of keeping one pool each.
  static llvm::DefaultThreadPool threadPool;
  auto context =
      std::make_shared<MLIRContext>(registry, MLIRContext::Threading::DISABLED);
  context->setThreadPool(threadPool);
  ensureInlinerExtensions(context.get());
  context->loadAllAvailableDialects();
  return context;
}

// Typed program factories verify the parsed module.
[[nodiscard]] static ParserConfig parserConfig(MLIRContext* context) {
  return {context, /*verifyAfterParse=*/false};
}

[[nodiscard]] static FailureOr<OwningOpRef<ModuleOp>>
parseMLIRString(MLIRContext* context, const StringRef source) {
  auto mod = parseSourceString<ModuleOp>(source, parserConfig(context));
  if (!mod) {
    return emitError(UnknownLoc::get(context),
                     "failed to parse MLIR source string");
  }
  return std::move(mod);
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
  auto mod = parseSourceFile<ModuleOp>(sourceMgr, parserConfig(context));
  if (!mod) {
    return emitError(UnknownLoc::get(context))
           << "failed to parse MLIR file '" << path.string() << "'";
  }
  return std::move(mod);
}

bool containsProgramOperations(ModuleOp moduleOp,
                               const ProgramDialect dialect) {
  return moduleOp
      ->walk([dialect](Operation* operation) {
        const auto ns = operation->getName().getDialectNamespace();
        const bool matches =
            dialect == ProgramDialect::QC
                ? ns == qc::QCDialect::getDialectNamespace()
                : ns == qco::QCODialect::getDialectNamespace() ||
                      ns == qtensor::QTensorDialect::getDialectNamespace();
        return matches ? WalkResult::interrupt() : WalkResult::advance();
      })
      .wasInterrupted();
}

/// Check that @p moduleOp can become a program of @p dialect in @p context.
///
/// Verify the module and reject operations of the other dialect. QCO programs
/// must also be linear.
[[nodiscard]] static LogicalResult
verifyProgramModule(const std::shared_ptr<MLIRContext>& context,
                    ModuleOp moduleOp, const ProgramDialect dialect) {
  const StringRef name = dialect == ProgramDialect::QC ? "QC" : "QCO";
  if (!moduleOp) {
    if (context) {
      emitError(UnknownLoc::get(context.get()))
          << "cannot construct a " << name << " program from a null module";
    }
    return failure();
  }
  if (!context) {
    return moduleOp->emitError() << "cannot construct a " << name
                                 << " program without its owning context";
  }
  if (moduleOp->getContext() != context.get()) {
    return moduleOp->emitError() << "cannot construct a " << name
                                 << " program with a different MLIR context";
  }
  context->getOrLoadDialect<mqt::MQTDialect>();
  if (failed(verify(moduleOp)) ||
      (!mqt::getEntryPoint(moduleOp) &&
       failed(mqt::verifyQuantumAllocations(moduleOp)))) {
    return failure();
  }
  if (dialect == ProgramDialect::QC) {
    if (containsProgramOperations(moduleOp, ProgramDialect::QCO)) {
      return moduleOp->emitError(
          "QC programs must not contain QCO or QTensor operations");
    }
    return success();
  }
  if (containsProgramOperations(moduleOp, ProgramDialect::QC)) {
    return moduleOp->emitError("QCO programs must not contain QC operations");
  }
  return qco::verifyLinearity(moduleOp);
}

template <class ProgramType, class Parse>
[[nodiscard]] static Expected<ProgramType> parseTypedProgram(Parse&& parse) {
  auto context = createCompilerContext();
  CompilerDiagnostics diagnostics(context.get());
  auto mod = std::forward<Parse>(parse)(context.get());
  if (failed(mod)) {
    return diagnostics.takeError();
  }
  auto program = ProgramType::fromModule(context, std::move(*mod));
  if (failed(program)) {
    return diagnostics.takeError();
  }
  return std::move(*program);
}

//===----------------------------------------------------------------------===//
// Program
//===----------------------------------------------------------------------===//

Program::Program(Storage storage) : storage_(std::move(storage)) {
  if (storage_.context) {
    ensureInlinerExtensions(storage_.context.get());
  }
}

Program& Program::operator=(Program&& other) noexcept {
  if (this != &other) {
    // Destroy the old module while its context is still alive.
    storage_.mod = std::move(other.storage_.mod);
    storage_.context = std::move(other.storage_.context);
  }
  return *this;
}

bool Program::isValid() const noexcept {
  return static_cast<bool>(storage_.mod);
}

ModuleOp Program::mod() const {
  assert(storage_.mod && "cannot use a consumed compiler program");
  return *storage_.mod;
}

ModuleOp Program::module() const { return mod(); }

std::shared_ptr<MLIRContext> Program::context() const noexcept {
  return storage_.context;
}

std::string Program::str() const {
  std::string result;
  llvm::raw_string_ostream stream(result);
  mod().print(stream);
  return result;
}

Program::Storage Program::cloneStorage() const {
  auto cloned = cast<ModuleOp>(mod()->clone());
  return {.context = storage_.context, .mod = OwningOpRef<ModuleOp>(cloned)};
}

Program::Storage Program::releaseStorage() && {
  assert(storage_.mod && "compiler program was already consumed");
  return {
      .context = std::move(storage_.context),
      .mod = std::move(storage_.mod),
  };
}

static Error writeTextProgram(StringRef source,
                              const std::filesystem::path& path,
                              StringRef format) {
  std::error_code error;
  llvm::raw_fd_ostream stream(path.string(), error, llvm::sys::fs::OF_Text);
  if (error) {
    return llvm::createStringError(error, "failed to open " + format +
                                              " output file '" + path.string() +
                                              "': " + error.message());
  }
  stream << source;
  stream.flush();
  if (stream.has_error()) {
    auto writeError = stream.error();
    stream.clear_error();
    return llvm::createStringError(
        writeError, "failed to write " + format + " file '" + path.string() +
                        "': " + writeError.message());
  }
  return Error::success();
}

//===----------------------------------------------------------------------===//
// OpenQASMProgram
//===----------------------------------------------------------------------===//

const std::string& OpenQASMProgram::source() const noexcept { return source_; }

std::string OpenQASMProgram::intoSource() && { return std::move(source_); }

const std::string& OpenQASMProgram::str() const noexcept { return source_; }

Error OpenQASMProgram::write(const std::filesystem::path& path) const {
  return writeTextProgram(source_, path, "OpenQASM");
}

//===----------------------------------------------------------------------===//
// IQMJSONProgram
//===----------------------------------------------------------------------===//

Error IQMJSONProgram::write(const std::filesystem::path& path) const {
  return writeTextProgram(source_, path, "IQM JSON");
}

//===----------------------------------------------------------------------===//
// QCProgram
//===----------------------------------------------------------------------===//

Expected<QCProgram> QCProgram::fromMLIRString(const std::string_view source) {
  return parseTypedProgram<QCProgram>([source](MLIRContext* context) {
    return parseMLIRString(context, source);
  });
}

Expected<QCProgram> QCProgram::fromMLIRFile(const std::filesystem::path& path) {
  return parseTypedProgram<QCProgram>(
      [&path](MLIRContext* context) { return parseMLIRFile(context, path); });
}

Expected<QCProgram>
QCProgram::fromOpenQASMString(const std::string_view source) {
  auto context = createCompilerContext();
  CompilerDiagnostics diagnostics(context.get());
  auto mod = qc::translateOpenQASMToQC(source, context.get());
  if (!mod) {
    emitError(UnknownLoc::get(context.get()),
              "failed to translate OpenQASM source to QC");
    return diagnostics.takeError();
  }
  return QCProgram({.context = context, .mod = std::move(mod)});
}

Expected<QCProgram>
QCProgram::fromOpenQASMFile(const std::filesystem::path& path) {
  auto context = createCompilerContext();
  CompilerDiagnostics diagnostics(context.get());
  llvm::SourceMgr sourceMgr;
  if (failed(openSourceMgr(path, context.get(), sourceMgr))) {
    return diagnostics.takeError();
  }
  auto program = fromOpenQASM(sourceMgr, context);
  if (failed(program)) {
    emitError(UnknownLoc::get(context.get()))
        << "failed to translate OpenQASM file '" << path.string() << "' to QC";
    return diagnostics.takeError();
  }
  return std::move(*program);
}

FailureOr<QCProgram>
QCProgram::fromOpenQASM(llvm::SourceMgr& sourceMgr,
                        std::shared_ptr<MLIRContext> context) {
  if (!context) {
    return failure();
  }
  auto mod = qc::translateOpenQASMToQC(sourceMgr, context.get());
  if (!mod) {
    return failure();
  }
  return QCProgram({.context = std::move(context), .mod = std::move(mod)});
}

FailureOr<QCProgram> QCProgram::fromModule(std::shared_ptr<MLIRContext> context,
                                           OwningOpRef<ModuleOp> moduleOp) {
  if (failed(
          verifyProgramModule(context, moduleOp.get(), ProgramDialect::QC))) {
    return failure();
  }
  return QCProgram({.context = std::move(context), .mod = std::move(moduleOp)});
}

QCProgram QCProgram::copy() const { return QCProgram(cloneStorage()); }

FailureOr<QCOProgram> QCProgram::intoQCO() && {
  auto storage = std::move(*this).releaseStorage();
  auto moduleOp = *storage.mod;
  if (failed(runWithPassManager(
          moduleOp, [](OpPassManager& pm) { pm.addPass(createQCToQCO()); },
          "failed to convert QC to QCO", {}, /*preservesLayout=*/true))) {
    return failure();
  }
  if (failed(qco::verifyLinearity(moduleOp))) {
    return failure();
  }
  return QCOProgram(std::move(storage));
}

//===----------------------------------------------------------------------===//
// QCOProgram
//===----------------------------------------------------------------------===//

Expected<QCOProgram> QCOProgram::fromMLIRString(const std::string_view source) {
  return parseTypedProgram<QCOProgram>([source](MLIRContext* context) {
    return parseMLIRString(context, source);
  });
}

Expected<QCOProgram>
QCOProgram::fromMLIRFile(const std::filesystem::path& path) {
  return parseTypedProgram<QCOProgram>(
      [&path](MLIRContext* context) { return parseMLIRFile(context, path); });
}

FailureOr<QCOProgram>
QCOProgram::fromModule(std::shared_ptr<MLIRContext> context,
                       OwningOpRef<ModuleOp> moduleOp) {
  if (failed(
          verifyProgramModule(context, moduleOp.get(), ProgramDialect::QCO))) {
    return failure();
  }
  return QCOProgram(
      {.context = std::move(context), .mod = std::move(moduleOp)});
}

QCOProgram QCOProgram::copy() const { return QCOProgram(cloneStorage()); }

//===----------------------------------------------------------------------===//
// JeffProgram
//===----------------------------------------------------------------------===//

/// Deserialize a binary `jeff` module, reporting failures as diagnostics.
[[nodiscard]] static OwningOpRef<ModuleOp>
deserializeJeff(MLIRContext* context, std::span<const std::byte> bytes) {
  if (bytes.size() % sizeof(capnp::word) != 0U) {
    emitError(UnknownLoc::get(context),
              "jeff data size must be a multiple of the Cap'n Proto word size");
    return {};
  }
  // Cap'n Proto reads word-aligned data.
  auto words = kj::heapArray<capnp::word>(bytes.size() / sizeof(capnp::word));
  std::memcpy(words.begin(), bytes.data(), bytes.size());
  auto mod = deserialize(context, words.asPtr());
  if (!mod) {
    emitError(UnknownLoc::get(context), "failed to deserialize jeff bytes");
  }
  return mod;
}

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
  auto context = createCompilerContext();
  CompilerDiagnostics diagnostics(context.get());
  auto mod = deserializeJeff(context.get(), bytes);
  if (!mod) {
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

//===----------------------------------------------------------------------===//
// Compiler input
//===----------------------------------------------------------------------===//

std::optional<InputFormat> inputFormatForPath(const StringRef path) {
  return llvm::StringSwitch<std::optional<InputFormat>>(
             llvm::sys::path::extension(path))
      .Case(".mlir", InputFormat::MLIR)
      .Case(".qasm", InputFormat::OpenQASM)
      .Case(".jeff", InputFormat::Jeff)
      .Default(std::nullopt);
}

FailureOr<CompilerInput>
parseCompilerInput(llvm::SourceMgr& sourceMgr,
                   std::shared_ptr<MLIRContext> context,
                   const InputFormat format) {
  if (!context) {
    return failure();
  }
  switch (format) {
  case InputFormat::OpenQASM: {
    auto program = QCProgram::fromOpenQASM(sourceMgr, std::move(context));
    if (failed(program)) {
      return failure();
    }
    return CompilerInput(std::move(*program));
  }
  case InputFormat::Jeff: {
    const auto bytes =
        sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID())->getBuffer();
    auto mod = deserializeJeff(
        context.get(), std::as_bytes(std::span(bytes.data(), bytes.size())));
    if (!mod) {
      return failure();
    }
    return CompilerInput(
        JeffProgram({.context = std::move(context), .mod = std::move(mod)}));
  }
  case InputFormat::MLIR:
    break;
  }
  auto mod = parseSourceFile<ModuleOp>(sourceMgr, parserConfig(context.get()));
  if (!mod) {
    return failure();
  }
  if (containsProgramOperations(*mod, ProgramDialect::QCO)) {
    auto program = QCOProgram::fromModule(std::move(context), std::move(mod));
    if (failed(program)) {
      return failure();
    }
    return CompilerInput(std::move(*program));
  }
  auto program = QCProgram::fromModule(std::move(context), std::move(mod));
  if (failed(program)) {
    return failure();
  }
  return CompilerInput(std::move(*program));
}

Expected<CompilerInput> loadCompilerInput(const std::filesystem::path& path,
                                          std::optional<InputFormat> format) {
  auto context = createCompilerContext();
  CompilerDiagnostics diagnostics(context.get());
  if (!format) {
    format = inputFormatForPath(path.string());
  }
  if (!format) {
    emitError(UnknownLoc::get(context.get()))
        << "input file '" << path.string()
        << "' has an unsupported extension; use .mlir, .qasm, or .jeff";
    return diagnostics.takeError();
  }
  llvm::SourceMgr sourceMgr;
  if (failed(openSourceMgr(path, context.get(), sourceMgr))) {
    return diagnostics.takeError();
  }
  // Keep the context alive until the diagnostic handler is removed.
  auto input = parseCompilerInput(sourceMgr, context, *format);
  if (failed(input)) {
    return diagnostics.takeError();
  }
  return std::move(*input);
}

} // namespace mlir
