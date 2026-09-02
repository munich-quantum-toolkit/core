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

#include "mlir/Compiler/JeffDeserializerError.h"
#include "mlir/Compiler/TargetCompilation.h"
#include "mlir/Conversion/JeffToQCO/JeffToQCO.h"
#include "mlir/Conversion/QCOToJeff/QCOToJeff.h"
#include "mlir/Conversion/QCOToQC/QCOToQC.h"
#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Conversion/QCToQIR/QIRAdaptive/QCToQIRAdaptive.h"
#include "mlir/Conversion/QCToQIR/QIRBase/QCToQIRBase.h"
#include "mlir/Dialect/CBit/IR/CBitDialect.h"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/MQT/Transforms/GlobalPhaseNormalization.h"
#include "mlir/Dialect/MQT/Transforms/Passes.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/IR/QCInterfaces.h"
#include "mlir/Dialect/QC/IR/QCOps.h"
#include "mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h"
#include "mlir/Dialect/QC/Translation/TranslateQCToOpenQASM3.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/QCOUtils.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QIR/Utils/QIRUtils.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Support/Passes.h"

#include <capnp/common.h>
#include <capnp/serialize.h>
#include <jeff.capnp.h>
#include <jeff/IR/JeffDialect.h>
#include <jeff/Translation/Deserialize.hpp>
#include <jeff/Translation/Serialize.hpp>
#include <kj/array.h>
#include <kj/exception.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Twine.h>
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
#include <mlir/Target/LLVMIR/ModuleTranslation.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
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

[[nodiscard]] static LogicalResult
runPasses(ModuleOp mod,
          const llvm::function_ref<void(OpPassManager&)> populatePasses,
          const StringRef failureMessage, const bool enableTiming = false,
          const bool enableStatistics = false) {
  if (failed(mqt::verifyProgramMetadata(mod))) {
    return failure();
  }
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
  return mqt::verifyProgramMetadata(mod);
}

[[nodiscard]] static LogicalResult runQCOTransformPasses(
    ModuleOp mod, const llvm::function_ref<void(OpPassManager&)> populatePasses,
    const StringRef failureMessage, const bool enableTiming = false,
    const bool enableStatistics = false) {
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
  if (failed(verify(*storage.mod)) ||
      failed(mqt::verifyProgramMetadata(*storage.mod))) {
    return std::nullopt;
  }
  if (!moduleUsesDialect(*storage.mod, "qc")) {
    storage.mod->emitError("expected a module using the 'qc' dialect");
    return std::nullopt;
  }
  return QCProgram(std::move(storage));
}

QCProgram QCProgram::copy() const { return QCProgram(cloneStorage()); }

bool QCProgram::cleanup() {
  return succeeded(runPasses(mod(), populateQCCleanupPipeline,
                             "failed to run the QC cleanup pipeline"));
}

bool QCProgram::normalizeGlobalPhases() {
  return succeeded(mqt::verifyProgramMetadata(mod())) &&
         succeeded(mqt::normalizeGlobalPhases(mod())) &&
         succeeded(mqt::verifyProgramMetadata(mod()));
}

std::optional<OpenQASMProgram> QCProgram::toOpenQASM3() const {
  auto cleaned = copy();
  if (!cleaned.cleanup()) {
    return std::nullopt;
  }
  auto source = qc::translateQCToOpenQASM3(cleaned.mod());
  if (failed(source)) {
    return std::nullopt;
  }
  return OpenQASMProgram(std::move(*source));
}

std::optional<QCOProgram> QCProgram::intoQCO() && {
  if (failed(runPasses(
          mod(), [](OpPassManager& pm) { pm.addPass(createQCToQCO()); },
          "failed to convert QC to QCO"))) {
    return std::nullopt;
  }
  if (failed(qco::verifyLinearity(mod()))) {
    return std::nullopt;
  }
  return QCOProgram(std::move(*this).releaseStorage());
}

std::optional<QIRProgram> QCProgram::intoQIR(const QIRProfile profile) && {
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

static size_t
countGatesIf(ModuleOp moduleOp,
             const llvm::function_ref<bool(qc::UnitaryOpInterface)> predicate) {
  auto entryPoint = mqt::getEntryPoint(moduleOp);
  if (!entryPoint) {
    return 0;
  }
  size_t count = 0;
  entryPoint.walk<WalkOrder::PreOrder>([&](qc::UnitaryOpInterface op) {
    count += static_cast<size_t>(!isa<qc::BarrierOp>(op) && predicate(op));
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
  if (failed(verify(*storage.mod)) ||
      failed(mqt::verifyProgramMetadata(*storage.mod))) {
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

bool QCOProgram::cleanup() {
  return succeeded(
      runQCOTransformPasses(mod(), populateQCOCleanupPipeline,
                            "failed to run the QCO cleanup pipeline"));
}

bool QCOProgram::normalizeGlobalPhases() {
  if (!hasValidLinearity() || failed(mqt::verifyProgramMetadata(mod()))) {
    return false;
  }
  return succeeded(mqt::normalizeGlobalPhases(mod())) &&
         succeeded(mqt::verifyProgramMetadata(mod())) && hasValidLinearity();
}

bool QCOProgram::runPassPipeline(const std::string_view pipeline,
                                 const bool enableTiming,
                                 const bool enableStatistics) {
  if (!hasValidLinearity() || failed(mqt::verifyProgramMetadata(mod()))) {
    return false;
  }
  if (failed(
          ::runPassPipeline(mod(), pipeline, enableTiming, enableStatistics))) {
    return false;
  }
  return succeeded(mqt::verifyProgramMetadata(mod())) && hasValidLinearity();
}

bool QCOProgram::mergeSingleQubitRotationGates() {
  return succeeded(runQCOTransformPasses(
      mod(),
      [](OpPassManager& pm) {
        pm.addPass(qco::createMergeSingleQubitRotationGates());
      },
      "failed to merge single-qubit rotation gates"));
}

bool QCOProgram::fuseSingleQubitUnitaryRuns(const std::string_view basis) {
  qco::FuseSingleQubitUnitaryRunsOptions options;
  options.basis = basis;
  return succeeded(runQCOTransformPasses(
      mod(),
      [&options](OpPassManager& pm) {
        pm.addPass(qco::createFuseSingleQubitUnitaryRuns(options));
      },
      "failed to fuse single-qubit unitary runs"));
}

bool QCOProgram::unrollQuantumLoops(const int64_t factor) {
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
  return succeeded(runQCOTransformPasses(
      mod(), [](OpPassManager& pm) { populateQubitReusePipeline(pm); },
      "failed to run the qubit reuse pipeline"));
}

bool QCOProgram::decomposeMultiControlled(const uint64_t minQubits) {
  return succeeded(runQCOTransformPasses(
      mod(),
      [minQubits](OpPassManager& pm) {
        populateDecomposeMultiControlledPipeline(pm, minQubits);
      },
      "failed to decompose multi-controlled gates"));
}

bool QCOProgram::compileForTarget(const CompilerTarget& target,
                                  const bool enableTiming,
                                  const bool enableStatistics) {
  return succeeded(runQCOTransformPasses(
      mod(),
      [&target](OpPassManager& pm) {
        populateTargetCompilationPipeline(pm, target);
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

namespace {

class JeffDeserializerInputVerifier {
public:
  JeffDeserializerInputVerifier(MLIRContext* context,
                                const ::jeff::Module::Reader module)
      : context(context), module(module), functions(module.getFunctions()),
        stringsSize(module.getStrings().size()) {}

  [[nodiscard]] LogicalResult verify() {
    if (module.getVersion() != 0 || module.getVersionMinor() != 3 ||
        module.getVersionPatch() != 0) {
      return reject("unsupported jeff version; expected 0.3.0");
    }
    if (stringsSize > MAX_CONTAINER_SIZE) {
      return reject("jeff module contains too many strings");
    }

    for (uint64_t functionIndex = 0; functionIndex < functions.size();
         ++functionIndex) {
      const auto function = functions[functionIndex];
      if (function.getName() >= stringsSize) {
        return reject("jeff function name does not refer to the string table");
      }
      if (function.isDeclaration()) {
        return reject("jeff function declarations are not supported");
      }
      const auto definition = function.getDefinition();
      if (!definition.hasBody()) {
        return reject("jeff function definition must contain a body");
      }
      if (!definition.getBody().hasOperations()) {
        return reject("jeff function body must contain an operations list");
      }
      const auto values = definition.getValues();
      if (values.size() > MAX_CONTAINER_SIZE) {
        return reject("jeff function contains too many values");
      }
      for (const auto value : values) {
        if (failed(verifyType(value.getType()))) {
          return failure();
        }
      }
      this->values = values;
      currentFunctionIndex = functionIndex;
      if (failed(verifyRegion(definition.getBody(), std::nullopt, std::nullopt,
                              0))) {
        return failure();
      }
    }
    return success();
  }

private:
  static constexpr uint64_t MAX_CONTAINER_SIZE = 1U << 20;
  static constexpr uint64_t MAX_REGION_DEPTH = 64;

  [[nodiscard]] LogicalResult reject(const Twine& message) const {
    return emitError(UnknownLoc::get(context)) << message;
  }

  [[nodiscard]] static bool isSupportedIntegerWidth(const uint8_t width) {
    return width == 1 || width == 8 || width == 16 || width == 32 ||
           width == 64;
  }

  [[nodiscard]] static bool
  isSupportedFloatPrecision(const ::jeff::FloatPrecision precision) {
    return precision == ::jeff::FloatPrecision::FLOAT32 ||
           precision == ::jeff::FloatPrecision::FLOAT64;
  }

  [[nodiscard]] LogicalResult
  verifyType(const ::jeff::Type::Reader type) const {
    switch (type.which()) {
    case ::jeff::Type::QUBIT:
    case ::jeff::Type::QUREG:
      return success();
    case ::jeff::Type::INT:
      if (isSupportedIntegerWidth(type.getInt())) {
        return success();
      }
      return reject("jeff integer type has an unsupported bit width");
    case ::jeff::Type::INT_ARRAY:
      if (isSupportedIntegerWidth(type.getIntArray().getBitwidth())) {
        return success();
      }
      return reject("jeff integer-array type has an unsupported bit width");
    case ::jeff::Type::FLOAT:
      if (isSupportedFloatPrecision(type.getFloat())) {
        return success();
      }
      return reject("jeff floating-point type has an unsupported precision");
    case ::jeff::Type::FLOAT_ARRAY:
      if (isSupportedFloatPrecision(type.getFloatArray().getPrecision())) {
        return success();
      }
      return reject(
          "jeff floating-point-array type has an unsupported precision");
    default:
      return reject("jeff value has an unknown type");
    }
  }

  [[nodiscard]] LogicalResult verifyArity(const ::jeff::Op::Reader operation,
                                          const uint64_t inputs,
                                          const uint64_t outputs) const {
    if (operation.getInputs().size() == inputs &&
        operation.getOutputs().size() == outputs) {
      return success();
    }
    return reject(Twine("jeff instruction expects ") + Twine(inputs) +
                  " inputs and " + Twine(outputs) + " outputs");
  }

  [[nodiscard]] LogicalResult
  verifyValueIndices(const ::jeff::Op::Reader operation) const {
    for (const auto input : operation.getInputs()) {
      if (input >= values.size()) {
        return reject("jeff instruction input is outside the value table");
      }
    }
    for (const auto output : operation.getOutputs()) {
      if (output >= values.size()) {
        return reject("jeff instruction output is outside the value table");
      }
    }
    return success();
  }

  [[nodiscard]] LogicalResult
  verifyRegion(const ::jeff::Region::Reader region,
               const std::optional<uint64_t> expectedSources,
               const std::optional<uint64_t> expectedTargets,
               const uint64_t depth) {
    if (depth > MAX_REGION_DEPTH) {
      return reject("jeff structured control flow exceeds the nesting limit");
    }
    if (expectedSources && region.getSources().size() != *expectedSources) {
      return reject("jeff region source count does not match its operation");
    }
    if (expectedTargets && region.getTargets().size() != *expectedTargets) {
      return reject("jeff region target count does not match its operation");
    }
    for (const auto source : region.getSources()) {
      if (source >= values.size()) {
        return reject("jeff region source is outside the value table");
      }
    }
    for (const auto target : region.getTargets()) {
      if (target >= values.size()) {
        return reject("jeff region target is outside the value table");
      }
    }
    if (!region.hasOperations()) {
      return reject("jeff region must contain an operations list");
    }
    const auto operations = region.getOperations();
    if (operations.size() > MAX_CONTAINER_SIZE - totalOperations) {
      return reject("jeff module contains too many operations");
    }
    totalOperations += operations.size();
    for (const auto operation : operations) {
      if (failed(verifyOperation(operation, depth))) {
        return failure();
      }
    }
    return success();
  }

  [[nodiscard]] LogicalResult
  verifyWellKnownGate(const ::jeff::Op::Reader operation,
                      const ::jeff::QubitGate::Reader gate) const {
    const uint64_t controls = gate.getControlQubits();
    uint64_t targets = 0;
    uint64_t parameters = 0;
    switch (gate.getWellKnown()) {
    case ::jeff::WellKnownGate::X:
    case ::jeff::WellKnownGate::Y:
    case ::jeff::WellKnownGate::Z:
    case ::jeff::WellKnownGate::S:
    case ::jeff::WellKnownGate::T:
    case ::jeff::WellKnownGate::H:
    case ::jeff::WellKnownGate::I:
      targets = 1;
      break;
    case ::jeff::WellKnownGate::R1:
    case ::jeff::WellKnownGate::RX:
    case ::jeff::WellKnownGate::RY:
    case ::jeff::WellKnownGate::RZ:
      targets = 1;
      parameters = 1;
      break;
    case ::jeff::WellKnownGate::U:
      targets = 1;
      parameters = 3;
      break;
    case ::jeff::WellKnownGate::SWAP:
      targets = 2;
      break;
    case ::jeff::WellKnownGate::GPHASE:
      parameters = 1;
      break;
    default:
      return reject("jeff instruction names an unknown well-known gate");
    }
    return verifyArity(operation, targets + controls + parameters,
                       targets + controls);
  }

  [[nodiscard]] LogicalResult
  verifyGate(const ::jeff::Op::Reader operation,
             const ::jeff::QubitGate::Reader gate) const {
    const uint64_t controls = gate.getControlQubits();
    switch (gate.which()) {
    case ::jeff::QubitGate::WELL_KNOWN:
      return verifyWellKnownGate(operation, gate);
    case ::jeff::QubitGate::CUSTOM: {
      const auto custom = gate.getCustom();
      if (custom.getName() >= stringsSize) {
        return reject("jeff custom-gate name is outside the string table");
      }
      const uint64_t targets = custom.getNumQubits();
      const uint64_t parameters = custom.getNumParams();
      return verifyArity(operation, targets + controls + parameters,
                         targets + controls);
    }
    case ::jeff::QubitGate::PPR: {
      const uint64_t targets = gate.getPpr().getPauliString().size();
      return verifyArity(operation, targets + controls + 1, targets + controls);
    }
    default:
      return reject("jeff instruction contains an unknown gate kind");
    }
  }

  [[nodiscard]] LogicalResult
  verifyQubitOperation(const ::jeff::Op::Reader operation) const {
    const auto instruction = operation.getInstruction().getQubit();
    switch (instruction.which()) {
    case ::jeff::QubitOp::ALLOC:
      return verifyArity(operation, 0, 1);
    case ::jeff::QubitOp::FREE:
    case ::jeff::QubitOp::FREE_ZERO:
      return verifyArity(operation, 1, 0);
    case ::jeff::QubitOp::MEASURE:
    case ::jeff::QubitOp::RESET:
      return verifyArity(operation, 1, 1);
    case ::jeff::QubitOp::MEASURE_ND:
      return verifyArity(operation, 1, 2);
    case ::jeff::QubitOp::GATE:
      return verifyGate(operation, instruction.getGate());
    default:
      return reject("jeff instruction contains an unknown qubit operation");
    }
  }

  [[nodiscard]] LogicalResult
  verifyQuregOperation(const ::jeff::Op::Reader operation) const {
    switch (operation.getInstruction().getQureg().which()) {
    case ::jeff::QuregOp::ALLOC:
      return verifyArity(operation, 1, 1);
    case ::jeff::QuregOp::FREE:
    case ::jeff::QuregOp::FREE_ZERO:
      return verifyArity(operation, 1, 0);
    case ::jeff::QuregOp::EXTRACT_INDEX:
      return verifyArity(operation, 2, 2);
    case ::jeff::QuregOp::INSERT_INDEX:
    case ::jeff::QuregOp::INSERT_SLICE:
      return verifyArity(operation, 3, 1);
    case ::jeff::QuregOp::EXTRACT_SLICE:
      return verifyArity(operation, 3, 2);
    case ::jeff::QuregOp::LENGTH:
      return verifyArity(operation, 1, 2);
    case ::jeff::QuregOp::SPLIT:
      return verifyArity(operation, 2, 2);
    case ::jeff::QuregOp::JOIN:
      return verifyArity(operation, 2, 1);
    case ::jeff::QuregOp::CREATE:
      return verifyArity(operation, operation.getInputs().size(), 1);
    default:
      return reject("jeff instruction contains an unknown qureg operation");
    }
  }

  [[nodiscard]] LogicalResult
  verifyIntegerOperation(const ::jeff::Op::Reader operation) const {
    switch (operation.getInstruction().getInt().which()) {
    case ::jeff::IntOp::CONST1:
    case ::jeff::IntOp::CONST8:
    case ::jeff::IntOp::CONST16:
    case ::jeff::IntOp::CONST32:
    case ::jeff::IntOp::CONST64:
      return verifyArity(operation, 0, 1);
    case ::jeff::IntOp::NOT:
    case ::jeff::IntOp::ABS:
      return verifyArity(operation, 1, 1);
    case ::jeff::IntOp::ADD:
    case ::jeff::IntOp::SUB:
    case ::jeff::IntOp::MUL:
    case ::jeff::IntOp::DIV_S:
    case ::jeff::IntOp::DIV_U:
    case ::jeff::IntOp::POW:
    case ::jeff::IntOp::AND:
    case ::jeff::IntOp::OR:
    case ::jeff::IntOp::XOR:
    case ::jeff::IntOp::MIN_S:
    case ::jeff::IntOp::MIN_U:
    case ::jeff::IntOp::MAX_S:
    case ::jeff::IntOp::MAX_U:
    case ::jeff::IntOp::REM_S:
    case ::jeff::IntOp::REM_U:
    case ::jeff::IntOp::SHL:
    case ::jeff::IntOp::SHR:
    case ::jeff::IntOp::EQ:
    case ::jeff::IntOp::LT_S:
    case ::jeff::IntOp::LTE_S:
    case ::jeff::IntOp::LT_U:
    case ::jeff::IntOp::LTE_U:
      return verifyArity(operation, 2, 1);
    default:
      return reject("jeff instruction contains an unknown integer operation");
    }
  }

  [[nodiscard]] LogicalResult
  verifyIntegerArrayOperation(const ::jeff::Op::Reader operation) const {
    const auto instruction = operation.getInstruction().getIntArray();
    switch (instruction.which()) {
    case ::jeff::IntArrayOp::CONST1:
    case ::jeff::IntArrayOp::CONST8:
    case ::jeff::IntArrayOp::CONST16:
    case ::jeff::IntArrayOp::CONST32:
    case ::jeff::IntArrayOp::CONST64:
      return verifyArity(operation, 0, 1);
    case ::jeff::IntArrayOp::ZERO:
      if (!isSupportedIntegerWidth(instruction.getZero())) {
        return reject("jeff integer-array zero has an unsupported bit width");
      }
      return verifyArity(operation, 1, 1);
    case ::jeff::IntArrayOp::GET_INDEX:
      if (failed(verifyArity(operation, 2, 1))) {
        return failure();
      }
      if (values[operation.getInputs()[0]].getType().which() !=
          ::jeff::Type::INT_ARRAY) {
        return reject("jeff integer-array get requires an integer-array input");
      }
      return success();
    case ::jeff::IntArrayOp::SET_INDEX:
      return verifyArity(operation, 3, 1);
    case ::jeff::IntArrayOp::LENGTH:
      return verifyArity(operation, 1, 1);
    case ::jeff::IntArrayOp::CREATE:
      if (operation.getInputs().size() == 0) {
        return reject("jeff integer-array create requires at least one input");
      }
      return verifyArity(operation, operation.getInputs().size(), 1);
    default:
      return reject(
          "jeff instruction contains an unknown integer-array operation");
    }
  }

  [[nodiscard]] LogicalResult
  verifyFloatOperation(const ::jeff::Op::Reader operation) const {
    switch (operation.getInstruction().getFloat().which()) {
    case ::jeff::FloatOp::CONST32:
    case ::jeff::FloatOp::CONST64:
      return verifyArity(operation, 0, 1);
    case ::jeff::FloatOp::SQRT:
    case ::jeff::FloatOp::ABS:
    case ::jeff::FloatOp::CEIL:
    case ::jeff::FloatOp::FLOOR:
    case ::jeff::FloatOp::EXP:
    case ::jeff::FloatOp::LOG:
    case ::jeff::FloatOp::SIN:
    case ::jeff::FloatOp::COS:
    case ::jeff::FloatOp::TAN:
    case ::jeff::FloatOp::ASIN:
    case ::jeff::FloatOp::ACOS:
    case ::jeff::FloatOp::ATAN:
    case ::jeff::FloatOp::SINH:
    case ::jeff::FloatOp::COSH:
    case ::jeff::FloatOp::TANH:
    case ::jeff::FloatOp::ASINH:
    case ::jeff::FloatOp::ACOSH:
    case ::jeff::FloatOp::ATANH:
    case ::jeff::FloatOp::IS_NAN:
    case ::jeff::FloatOp::IS_INF:
      return verifyArity(operation, 1, 1);
    case ::jeff::FloatOp::ADD:
    case ::jeff::FloatOp::SUB:
    case ::jeff::FloatOp::MUL:
    case ::jeff::FloatOp::POW:
    case ::jeff::FloatOp::ATAN2:
    case ::jeff::FloatOp::MAX:
    case ::jeff::FloatOp::MIN:
    case ::jeff::FloatOp::EQ:
    case ::jeff::FloatOp::LT:
    case ::jeff::FloatOp::LTE:
      return verifyArity(operation, 2, 1);
    default:
      return reject(
          "jeff instruction contains an unknown floating-point operation");
    }
  }

  [[nodiscard]] LogicalResult
  verifyFloatArrayOperation(const ::jeff::Op::Reader operation) const {
    const auto instruction = operation.getInstruction().getFloatArray();
    switch (instruction.which()) {
    case ::jeff::FloatArrayOp::CONST32:
    case ::jeff::FloatArrayOp::CONST64:
      return verifyArity(operation, 0, 1);
    case ::jeff::FloatArrayOp::ZERO:
      if (!isSupportedFloatPrecision(instruction.getZero())) {
        return reject(
            "jeff floating-point-array zero has an unsupported precision");
      }
      return verifyArity(operation, 1, 1);
    case ::jeff::FloatArrayOp::GET_INDEX:
      if (failed(verifyArity(operation, 2, 1))) {
        return failure();
      }
      if (values[operation.getInputs()[0]].getType().which() !=
          ::jeff::Type::FLOAT_ARRAY) {
        return reject("jeff floating-point-array get requires a "
                      "floating-point-array input");
      }
      return success();
    case ::jeff::FloatArrayOp::SET_INDEX:
      return verifyArity(operation, 3, 1);
    case ::jeff::FloatArrayOp::LENGTH:
      return verifyArity(operation, 1, 1);
    case ::jeff::FloatArrayOp::CREATE:
      if (operation.getInputs().size() == 0) {
        return reject(
            "jeff floating-point-array create requires at least one input");
      }
      return verifyArity(operation, operation.getInputs().size(), 1);
    default:
      return reject("jeff instruction contains an unknown "
                    "floating-point-array operation");
    }
  }

  [[nodiscard]] LogicalResult
  verifyStructuredControlFlow(const ::jeff::Op::Reader operation,
                              const uint64_t depth) {
    const auto instruction = operation.getInstruction().getScf();
    const uint64_t inputs = operation.getInputs().size();
    const uint64_t outputs = operation.getOutputs().size();
    switch (instruction.which()) {
    case ::jeff::ScfOp::SWITCH: {
      if (inputs == 0 || outputs != inputs - 1) {
        return reject("jeff switch shape is unsupported by the deserializer");
      }
      const auto switchInstruction = instruction.getSwitch();
      for (const auto branch : switchInstruction.getBranches()) {
        if (failed(verifyRegion(branch, inputs - 1, outputs, depth + 1))) {
          return failure();
        }
      }
      if (switchInstruction.hasDefault() &&
          failed(verifyRegion(switchInstruction.getDefault(), inputs - 1,
                              outputs, depth + 1))) {
        return failure();
      }
      return success();
    }
    case ::jeff::ScfOp::FOR:
      if (inputs < 3 || outputs != inputs - 3) {
        return reject("jeff for-loop shape is unsupported by the deserializer");
      }
      return verifyRegion(instruction.getFor(), outputs + 1, outputs,
                          depth + 1);
    case ::jeff::ScfOp::WHILE:
      if (outputs != inputs) {
        return reject(
            "jeff while-loop shape is unsupported by the deserializer");
      }
      if (failed(verifyRegion(instruction.getWhile().getBefore(), inputs,
                              outputs + 1, depth + 1))) {
        return failure();
      }
      return verifyRegion(instruction.getWhile().getAfter(), outputs, inputs,
                          depth + 1);
    default:
      return reject(
          "jeff instruction contains an unknown structured-control-flow "
          "operation");
    }
  }

  [[nodiscard]] LogicalResult
  verifyFunctionCall(const ::jeff::Op::Reader operation) const {
    const uint64_t callee = operation.getInstruction().getFunc().getFuncCall();
    if (callee >= functions.size()) {
      return reject("jeff call refers to an unknown function");
    }
    if (callee > currentFunctionIndex) {
      return reject("jeff forward function calls are unsupported");
    }
    const auto body = functions[callee].getDefinition().getBody();
    return verifyArity(operation, body.getSources().size(),
                       body.getTargets().size());
  }

  [[nodiscard]] LogicalResult
  verifyOperation(const ::jeff::Op::Reader operation, const uint64_t depth) {
    if (failed(verifyValueIndices(operation))) {
      return failure();
    }
    switch (operation.getInstruction().which()) {
    case ::jeff::Op::Instruction::QUBIT:
      return verifyQubitOperation(operation);
    case ::jeff::Op::Instruction::QUREG:
      return verifyQuregOperation(operation);
    case ::jeff::Op::Instruction::INT:
      return verifyIntegerOperation(operation);
    case ::jeff::Op::Instruction::INT_ARRAY:
      return verifyIntegerArrayOperation(operation);
    case ::jeff::Op::Instruction::FLOAT:
      return verifyFloatOperation(operation);
    case ::jeff::Op::Instruction::FLOAT_ARRAY:
      return verifyFloatArrayOperation(operation);
    case ::jeff::Op::Instruction::SCF:
      return verifyStructuredControlFlow(operation, depth);
    case ::jeff::Op::Instruction::FUNC:
      return verifyFunctionCall(operation);
    default:
      return reject("jeff instruction has an unknown kind");
    }
  }

  MLIRContext* context;
  ::jeff::Module::Reader module;
  capnp::List<::jeff::Function>::Reader functions;
  uint64_t stringsSize;
  capnp::List<::jeff::Value>::Reader values;
  uint64_t currentFunctionIndex = 0;
  uint64_t totalOperations = 0;
};

} // namespace

[[nodiscard]] static LogicalResult
verifyJeffDeserializerInput(MLIRContext* context,
                            kj::ArrayPtr<const capnp::word> words) {
  capnp::FlatArrayMessageReader message(words);
  const auto module = message.getRoot<::jeff::Module>();
  if (!module.hasFunctions()) {
    return emitError(UnknownLoc::get(context),
                     "jeff module must contain a functions list");
  }
  const auto functions = module.getFunctions();
  if (functions.size() == 0) {
    return emitError(UnknownLoc::get(context),
                     "jeff module must contain at least one function");
  }
  constexpr uint64_t maxFunctions =
      static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 1;
  if (functions.size() > maxFunctions) {
    return emitError(UnknownLoc::get(context))
           << "jeff module contains more than " << maxFunctions << " functions";
  }
  if (module.getEntrypoint() >= functions.size()) {
    return emitError(UnknownLoc::get(context),
                     "jeff entry point does not refer to a function");
  }

  return JeffDeserializerInputVerifier(context, module).verify();
}

[[nodiscard]] static FailureOr<OwningOpRef<ModuleOp>>
deserializeJeffBytes(MLIRContext* context,
                     const std::span<const std::byte> bytes) {
  if (bytes.empty()) {
    return emitError(UnknownLoc::get(context), "jeff data must not be empty");
  }
  if (bytes.size() % sizeof(capnp::word) != 0U) {
    return emitError(
        UnknownLoc::get(context),
        "jeff data size must be a multiple of the Cap'n Proto word size");
  }

  auto words = kj::heapArray<capnp::word>(bytes.size() / sizeof(capnp::word));
  std::memcpy(words.begin(), bytes.data(), bytes.size());

  try {
    if (failed(verifyJeffDeserializerInput(context, words.asPtr()))) {
      return failure();
    }
    auto mod = deserialize(context, words.asPtr());
    if (!mod) {
      return emitError(UnknownLoc::get(context),
                       "failed to deserialize jeff bytes");
    }
    return mod;
  } catch (const kj::Exception& exception) {
    return emitError(UnknownLoc::get(context))
           << "failed to parse jeff data: "
           << exception.getDescription().cStr();
  } catch (const detail::JeffDeserializerError& exception) {
    return emitError(UnknownLoc::get(context))
           << "failed to deserialize jeff data: " << exception.what();
  }
}

FailureOr<OwningOpRef<ModuleOp>>
detail::deserializeJeffFile(MLIRContext* context,
                            const std::filesystem::path& path) {
  std::string errorMessage;
  auto file = openInputFile(path.string(), &errorMessage);
  if (!file) {
    return emitError(UnknownLoc::get(context))
           << "failed to load jeff file '" << path.string()
           << "': " << errorMessage;
  }

  const auto buffer = file->getBuffer();
  return deserializeJeffBytes(
      context, std::as_bytes(std::span(buffer.data(), buffer.size())));
}

std::optional<JeffProgram>
JeffProgram::fromBytes(const std::span<const std::byte> bytes) {
  auto context = createCompilerContext();
  auto mod = deserializeJeffBytes(context.get(), bytes);
  if (failed(mod)) {
    return std::nullopt;
  }
  return JeffProgram({.context = std::move(context), .mod = std::move(*mod)});
}

std::optional<JeffProgram>
JeffProgram::fromFile(const std::filesystem::path& path) {
  auto context = createCompilerContext();
  auto mod = detail::deserializeJeffFile(context.get(), path);
  if (failed(mod)) {
    return std::nullopt;
  }
  return JeffProgram({.context = std::move(context), .mod = std::move(*mod)});
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

QIRProgram::QIRProgram(Storage storage, const QIRProfile profile)
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

std::optional<CompilerProgram>
runDefaultPipeline(CompilerInput&& program, const ProgramFormat output,
                   const CompilerTarget* const target,
                   const std::string_view qcoPipeline, const bool enableTiming,
                   const bool enableStatistics) {
  const bool hasValidInput = std::visit(
      []<typename T>(T& value) {
        using ProgramType = std::remove_cvref_t<T>;
        if constexpr (std::is_same_v<ProgramType, OpenQASMProgram>) {
          return true;
        } else {
          return succeeded(verify(value.module())) &&
                 succeeded(mqt::verifyProgramMetadata(value.module()));
        }
      },
      program);
  if (!hasValidInput) {
    return std::nullopt;
  }
  if (target != nullptr &&
      (output == ProgramFormat::QCImport || output == ProgramFormat::QCO ||
       output == ProgramFormat::Jeff)) {
    llvm::errs()
        << "a compiler target requires QCOOptimized, QC, OpenQASM3, or QIR "
           "output.\n";
    return std::nullopt;
  }
  if (target != nullptr && qcoPipeline != "mqt-qco-default") {
    llvm::errs() << "a custom QCO pass pipeline cannot be combined with a "
                    "compiler target.\n";
    return std::nullopt;
  }
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

  if (target != nullptr) {
    if (!qco->compileForTarget(*target, enableTiming, enableStatistics)) {
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
    auto openQASM = qc->toOpenQASM3();
    if (!openQASM) {
      return std::nullopt;
    }
    return CompilerProgram(std::move(*openQASM));
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
