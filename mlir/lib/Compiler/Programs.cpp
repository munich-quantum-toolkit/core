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

#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCInterfaces.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/QCOUtils.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"

#include <jeff/IR/JeffDialect.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Verifier.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

#include <cassert>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace mlir {

std::shared_ptr<MLIRContext> createCompilerContext() {
  DialectRegistry registry;
  registry.insert<cbit::CBitDialect, mqt::MQTDialect, qc::QCDialect,
                  qco::QCODialect, qtensor::QTensorDialect, arith::ArithDialect,
                  cf::ControlFlowDialect, func::FuncDialect, math::MathDialect,
                  scf::SCFDialect, LLVM::LLVMDialect, memref::MemRefDialect,
                  tensor::TensorDialect, jeff::JeffDialect>();
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);

  auto context = std::make_shared<MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

[[nodiscard]] static FailureOr<OwningOpRef<ModuleOp>>
parseMLIRString(MLIRContext* context, const StringRef source) {
  auto mod = parseSourceString<ModuleOp>(source, context);
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
  auto mod = parseSourceFile<ModuleOp>(sourceMgr, context);
  if (!mod) {
    return emitError(UnknownLoc::get(context))
           << "failed to parse MLIR file '" << path.string() << "'";
  }
  return std::move(mod);
}

/**
 * @brief Check whether a module contains an operation from a dialect.
 */
[[nodiscard]] static bool moduleUsesDialect(ModuleOp mod,
                                            const StringRef dialect) {
  return mod
      ->walk([&](Operation* operation) {
        return operation->getDialect()->getNamespace() == dialect
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      })
      .wasInterrupted();
}

template <class ProgramType, class Parse>
[[nodiscard]] static std::optional<ProgramType>
parseTypedProgram(Parse&& parse) {
  auto context = createCompilerContext();
  auto mod = std::forward<Parse>(parse)(context.get());
  if (failed(mod)) {
    return std::nullopt;
  }
  return ProgramType::fromModule(std::move(context), std::move(*mod));
}

//===----------------------------------------------------------------------===//
// Program
//===----------------------------------------------------------------------===//

Program::Program(Storage storage) : storage_(std::move(storage)) {}

bool Program::isValid() const noexcept {
  return static_cast<bool>(storage_.mod);
}

ModuleOp Program::mod() const {
  assert(storage_.mod && "cannot use a consumed compiler program");
  return *storage_.mod;
}

ModuleOp Program::module() const { return mod(); }

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
  return {.context = std::move(storage_.context),
          .mod = std::move(storage_.mod)};
}

//===----------------------------------------------------------------------===//
// OpenQASMProgram
//===----------------------------------------------------------------------===//

const std::string& OpenQASMProgram::source() const noexcept { return source_; }

const std::string& OpenQASMProgram::str() const noexcept { return source_; }

bool OpenQASMProgram::write(const std::filesystem::path& path) const {
  std::error_code error;
  llvm::raw_fd_ostream stream(path.string(), error, llvm::sys::fs::OF_Text);
  if (error) {
    llvm::errs() << "failed to open OpenQASM output file '" << path.string()
                 << "': " << error.message() << '\n';
    return false;
  }
  stream << source_;
  stream.flush();
  if (stream.has_error()) {
    llvm::errs() << "failed to write OpenQASM file '" << path.string() << "'\n";
    return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// QCProgram
//===----------------------------------------------------------------------===//

std::optional<QCProgram>
QCProgram::fromMLIRString(const std::string_view source) {
  return parseTypedProgram<QCProgram>([source](MLIRContext* context) {
    return parseMLIRString(context, source);
  });
}

std::optional<QCProgram>
QCProgram::fromMLIRFile(const std::filesystem::path& path) {
  return parseTypedProgram<QCProgram>(
      [&path](MLIRContext* context) { return parseMLIRFile(context, path); });
}

std::optional<QCProgram>
QCProgram::fromQASMString(const std::string_view source) {
  auto context = createCompilerContext();
  auto mod = qc::translateQASM3ToQC(source, context.get());
  if (!mod) {
    emitError(UnknownLoc::get(context.get()),
              "failed to translate OpenQASM 3 source to QC");
    return std::nullopt;
  }
  return QCProgram({.context = std::move(context), .mod = std::move(mod)});
}

std::optional<QCProgram>
QCProgram::fromQASMFile(const std::filesystem::path& path) {
  auto context = createCompilerContext();
  llvm::SourceMgr sourceMgr;
  if (failed(openSourceMgr(path, context.get(), sourceMgr))) {
    return std::nullopt;
  }
  auto mod = qc::translateQASM3ToQC(sourceMgr, context.get());
  if (!mod) {
    emitError(UnknownLoc::get(context.get()))
        << "failed to translate OpenQASM 3 file '" << path.string()
        << "' to QC";
    return std::nullopt;
  }
  return QCProgram({.context = std::move(context), .mod = std::move(mod)});
}

std::optional<QCProgram>
QCProgram::fromModule(std::shared_ptr<MLIRContext> context,
                      OwningOpRef<ModuleOp> moduleOp) {
  Storage storage{.context = std::move(context), .mod = std::move(moduleOp)};
  if (!storage.mod) {
    if (storage.context) {
      emitError(UnknownLoc::get(storage.context.get()),
                "cannot construct a QC program from a null module");
    }
    return std::nullopt;
  }
  if (!storage.context) {
    storage.mod->emitError(
        "cannot construct a QC program without its owning context");
    return std::nullopt;
  }
  if (storage.mod->getContext() != storage.context.get()) {
    storage.mod->emitError(
        "cannot construct a QC program with a different MLIR context");
    return std::nullopt;
  }
  if (failed(verify(*storage.mod))) {
    return std::nullopt;
  }
  if (!moduleUsesDialect(*storage.mod, "qc")) {
    storage.mod->emitError("expected a module using the 'qc' dialect");
    return std::nullopt;
  }
  return QCProgram(std::move(storage));
}

QCProgram QCProgram::copy() const { return QCProgram(cloneStorage()); }

std::optional<QCOProgram> QCProgram::intoQCO() && {
  PassManager pm(mod().getContext());
  pm.addPass(createQCToQCO());
  if (failed(pm.run(mod()))) {
    mod().emitError("failed to convert QC to QCO");
    return std::nullopt;
  }
  if (failed(qco::verifyLinearity(mod()))) {
    return std::nullopt;
  }
  return QCOProgram(std::move(*this).releaseStorage());
}

static size_t
countGatesIf(ModuleOp moduleOp,
             const llvm::function_ref<bool(qc::UnitaryOpInterface)> predicate) {
  auto entryPoint = mqt::getEntryPoint(moduleOp);
  if (!entryPoint) {
    return 0;
  }
  size_t count = 0;
  entryPoint.walk<WalkOrder::PreOrder>([&](qc::UnitaryOpInterface op) {
    count += !isa<qc::BarrierOp>(op) && predicate(op);
    return isa<qc::CtrlOp, qc::InvOp, qc::PowOp>(op) ? WalkResult::skip()
                                                     : WalkResult::advance();
  });
  return count;
}

size_t QCProgram::numGates() const {
  return countGatesIf(mod(), [](qc::UnitaryOpInterface) { return true; });
}

size_t QCProgram::numSingleQubitGates() const {
  return countGatesIf(
      mod(), [](qc::UnitaryOpInterface op) { return op.isSingleQubit(); });
}

size_t QCProgram::numTwoQubitGates() const {
  return countGatesIf(
      mod(), [](qc::UnitaryOpInterface op) { return op.isTwoQubit(); });
}

//===----------------------------------------------------------------------===//
// QCOProgram
//===----------------------------------------------------------------------===//

std::optional<QCOProgram>
QCOProgram::fromMLIRString(const std::string_view source) {
  return parseTypedProgram<QCOProgram>([source](MLIRContext* context) {
    return parseMLIRString(context, source);
  });
}

std::optional<QCOProgram>
QCOProgram::fromMLIRFile(const std::filesystem::path& path) {
  return parseTypedProgram<QCOProgram>(
      [&path](MLIRContext* context) { return parseMLIRFile(context, path); });
}

std::optional<QCOProgram>
QCOProgram::fromModule(std::shared_ptr<MLIRContext> context,
                       OwningOpRef<ModuleOp> moduleOp) {
  Storage storage{.context = std::move(context), .mod = std::move(moduleOp)};
  if (!storage.mod) {
    if (storage.context) {
      emitError(UnknownLoc::get(storage.context.get()),
                "cannot construct a QCO program from a null module");
    }
    return std::nullopt;
  }
  if (!storage.context) {
    storage.mod->emitError(
        "cannot construct a QCO program without its owning context");
    return std::nullopt;
  }
  if (storage.mod->getContext() != storage.context.get()) {
    storage.mod->emitError(
        "cannot construct a QCO program with a different MLIR context");
    return std::nullopt;
  }
  if (failed(verify(*storage.mod))) {
    return std::nullopt;
  }
  if (!moduleUsesDialect(*storage.mod, "qco")) {
    storage.mod->emitError("expected a module using the 'qco' dialect");
    return std::nullopt;
  }
  if (failed(qco::verifyLinearity(*storage.mod))) {
    return std::nullopt;
  }
  return QCOProgram(std::move(storage));
}

QCOProgram QCOProgram::copy() const { return QCOProgram(cloneStorage()); }

bool QCOProgram::hasValidLinearity() const {
  return succeeded(qco::verifyLinearity(mod()));
}

} // namespace mlir
