/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/QuantumComputation.hpp"
#include "mlir/Compiler/CompilerPipeline.h"
#include "mlir/Conversion/JeffToQCO/JeffToQCO.h"
#include "mlir/Conversion/QCOToQC/QCOToQC.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/Translation/TranslateQASM3ToQC.h"
#include "mlir/Dialect/QC/Translation/TranslateQuantumComputationToQC.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Support/Passes.h"

#include <jeff/IR/JeffDialect.h>
#include <jeff/Translation/Deserialize.hpp>
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
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/ModuleTranslation.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h> // NOLINT(misc-include-cleaner)

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

namespace {

/**
 * @brief Construct and initialize the MLIR context used by the compiler.
 */
[[nodiscard]] std::unique_ptr<mlir::MLIRContext> createCompilerContext() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::qc::QCDialect, mlir::qco::QCODialect,
                  mlir::qtensor::QTensorDialect, mlir::arith::ArithDialect,
                  mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                  mlir::scf::SCFDialect, mlir::LLVM::LLVMDialect,
                  mlir::memref::MemRefDialect, mlir::jeff::JeffDialect>();
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);

  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

/**
 * @brief Deserialize a `.jeff` file and convert the program to QC.
 */
[[nodiscard]] mlir::OwningOpRef<mlir::ModuleOp>
moduleFromJeffFile(mlir::MLIRContext* context, const std::string& path) {
  auto mod = deserializeFromFile(context, path);
  if (!mod) {
    throw std::runtime_error(std::string("Failed to deserialize jeff file '") +
                             path + "'.");
  }

  mlir::PassManager pm(mod->getContext());
  pm.addPass(mlir::createJeffToQCO());
  populateQCOCleanupPipeline(pm);
  pm.addPass(mlir::createQCOToQC());
  populateQCCleanupPipeline(pm);

  if (mlir::failed(pm.run(*mod))) {
    throw std::runtime_error("Failed to convert from jeff to QC.");
  }

  return mod;
}

/**
 * @brief Open a file and wrap it in a `SourceMgr` ready for parsing.
 */
[[nodiscard]] llvm::SourceMgr openSourceMgr(const std::string& path) {
  std::string errorMessage;
  auto file = mlir::openInputFile(path, &errorMessage);
  if (!file) {
    throw std::runtime_error(std::string("Failed to load file '") + path +
                             "': '" + errorMessage + "'");
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  return sourceMgr;
}

/**
 * @brief Parse an MLIR source string into a module.
 */
[[nodiscard]] mlir::OwningOpRef<mlir::ModuleOp>
moduleFromMlirString(mlir::MLIRContext* context, const std::string& text) {
  return mlir::parseSourceString<mlir::ModuleOp>(llvm::StringRef(text),
                                                 context);
}

/**
 * @brief Parse a `.mlir` file into a module.
 */
[[nodiscard]] mlir::OwningOpRef<mlir::ModuleOp>
moduleFromMlirFile(mlir::MLIRContext* context, const std::string& path) {
  const auto sourceMgr = openSourceMgr(path);
  return parseSourceFile<mlir::ModuleOp>(sourceMgr, context);
}

/**
 * @brief Parse an OpenQASM source string and translate it to a QC program.
 */
[[nodiscard]] mlir::OwningOpRef<mlir::ModuleOp>
moduleFromQasmString(mlir::MLIRContext* context,
                     const std::string& qasmSource) {
  return mlir::qc::translateQASM3ToQC(qasmSource, context);
}

/**
 * @brief Parse a `.qasm` file and translate it to a QC program.
 */
[[nodiscard]] mlir::OwningOpRef<mlir::ModuleOp>
moduleFromQasmFile(mlir::MLIRContext* context, const std::string& path) {
  auto sourceMgr = openSourceMgr(path);
  return mlir::qc::translateQASM3ToQC(sourceMgr, context);
}

/**
 * @brief Parse a source string into an MLIR module.
 */
[[nodiscard]] mlir::OwningOpRef<mlir::ModuleOp>
moduleFromSourceString(mlir::MLIRContext* context, const std::string& input) {
  if (input.find("OPENQASM") != std::string::npos) {
    return moduleFromQasmString(context, input);
  }

  if (auto mod = moduleFromMlirString(context, input)) {
    return mod;
  }

  throw std::runtime_error("Failed to parse source string.");
}

/**
 * @brief Resolve a string to an MLIR module.
 *
 * @details The string can be a source string or a path to a file.
 */
[[nodiscard]] mlir::OwningOpRef<mlir::ModuleOp>
moduleFromString(mlir::MLIRContext* context, const std::string& input) {
  if (input.find('\n') != std::string::npos) {
    return moduleFromSourceString(context, input);
  }

  const auto path = std::filesystem::path(input);
  if (path.empty()) {
    return moduleFromSourceString(context, input);
  }

  std::error_code ec;
  const auto status = std::filesystem::status(path, ec);
  if (ec) {
    throw std::runtime_error(std::string("Failed to inspect path '") + input +
                             "': " + ec.message());
  }
  if (!std::filesystem::exists(status)) {
    throw std::runtime_error(std::string("Input file '") + input +
                             "' does not exist.");
  }
  if (!std::filesystem::is_regular_file(status)) {
    throw std::runtime_error(std::string("Input path '") + input +
                             "' is not a file.");
  }

  using ModuleLoader = mlir::OwningOpRef<mlir::ModuleOp> (*)(
      mlir::MLIRContext*, const std::string&);
  static const std::unordered_map<std::string, ModuleLoader> loaders{
      {".jeff", &moduleFromJeffFile},
      {".mlir", &moduleFromMlirFile},
      {".qasm", &moduleFromQasmFile},
  };

  const auto extension = path.extension().string();
  const auto it = loaders.find(extension);
  if (it == loaders.end()) {
    throw std::runtime_error(std::string("Input file '") + input +
                             "' has unsupported extension '" + extension +
                             "'.");
  }
  return it->second(context, input);
}

/**
 * @brief Translate a `QuantumComputation` to a QC program.
 */
[[nodiscard]] mlir::OwningOpRef<mlir::ModuleOp>
moduleFromQuantumComputation(mlir::MLIRContext* context,
                             const qc::QuantumComputation& computation) {
  auto mod = mlir::translateQuantumComputationToQC(context, computation);
  if (!mod) {
    throw std::runtime_error("Failed to translate QuantumComputation to MLIR.");
  }
  return mod;
}

/**
 * @brief Convert a generic Python object to an MLIR module.
 */
[[nodiscard]] mlir::OwningOpRef<mlir::ModuleOp>
moduleFromInputProgram(mlir::MLIRContext* context, const nb::object& program) {
  if (nb::isinstance<nb::str>(program)) {
    return moduleFromString(context, nb::cast<std::string>(program));
  }

  if (nb::hasattr(program, "__fspath__")) {
    const auto path = nb::cast<std::string>(
        nb::module_::import_("os").attr("fspath")(program));
    return moduleFromString(context, path);
  }

  if (nb::isinstance<qc::QuantumComputation>(program)) {
    const auto& qc = nb::cast<const qc::QuantumComputation&>(program);
    return moduleFromQuantumComputation(context, qc);
  }

  const auto programType =
      nb::cast<std::string>(program.type().attr("__name__"));
  if (programType == "QuantumCircuit") {
    const auto& qc = nb::cast<qc::QuantumComputation>(
        nb::module_::import_("mqt.core.load").attr("load")(program));
    return moduleFromQuantumComputation(context, qc);
  }

  throw std::runtime_error(std::string("Program type ") + programType +
                           " is not supported.");
}

/**
 * @brief Compile a program and return the final MLIR module as a string.
 */
[[nodiscard]] std::string
compileProgram(const nb::object& program, const bool convertToQIRBase,
               const bool convertToQIRAdaptive,
               const bool disableMergeSingleQubitRotationGates,
               const bool enableHadamardLifting, const bool enableTiming,
               const bool enableStatistics) {
  auto context = createCompilerContext();

  auto mod = moduleFromInputProgram(context.get(), program);
  if (!mod) {
    throw std::runtime_error("Failed to create MLIR module.");
  }

  mlir::QuantumCompilerConfig config;
  config.convertToQIRBase = convertToQIRBase;
  config.convertToQIRAdaptive = convertToQIRAdaptive;
  config.disableMergeSingleQubitRotationGates =
      disableMergeSingleQubitRotationGates;
  config.enableHadamardLifting = enableHadamardLifting;
  config.enableTiming = enableTiming;
  config.enableStatistics = enableStatistics;

  if (const mlir::QuantumCompilerPipeline pipeline(config);
      mlir::failed(pipeline.runPipeline(mod.get()))) {
    throw std::runtime_error("Failed to run compilation pipeline.");
  }

  if (convertToQIRBase || convertToQIRAdaptive) {
    llvm::LLVMContext llvmContext;
    std::unique_ptr<llvm::Module> llvmMod =
        mlir::translateModuleToLLVMIR(*mod, llvmContext);
    if (!llvmMod) {
      throw std::runtime_error("Failed to translate MLIR module to LLVM IR.");
    }
    std::string result;
    llvm::raw_string_ostream os(result);
    llvmMod->print(os, nullptr);
    return result;
  }

  return mlir::captureIR(mod.get());
}

} // namespace

NB_MODULE(MQT_CORE_MODULE_NAME, m) {
  m.doc() = R"pb(
MQT Core MLIR compiler bindings.
)pb";

  nb::module_::import_("mqt.core.ir");

  m.def("compile_program", &compileProgram, "program"_a, nb::kw_only(),
        "convert_to_qir_base"_a = false, "convert_to_qir_adaptive"_a = false,
        "disable_merge_single_qubit_rotation_gates"_a = false,
        "enable_hadamard_lifting"_a = false, "enable_timing"_a = false,
        "enable_statistics"_a = false,
        R"pb(
Compile an input quantum program with the MQT MLIR compiler pipeline.

Args:
    program: Input program in one of the supported forms:
        - Path to a `.jeff`, `.mlir`, or `.qasm` file
        - MLIR or OpenQASM source string
        - :class:`mqt.core.ir.QuantumComputation`
        - :class:`~qiskit.circuit.QuantumCircuit`
    convert_to_qir_base: Whether to lower the result to a QIR program compliant with the Base Profile.
    convert_to_qir_adaptive: Whether to lower the result to QIR program compliant with the Adaptive Profile.
    disable_merge_single_qubit_rotation_gates: Disable quaternion-based rotation merging.
    enable_hadamard_lifting: Enable Hadamard lifting optimization.
    enable_timing: Enable MLIR pass timing.
    enable_statistics: Enable MLIR pass statistics.

Returns:
    The final MLIR module as text.
)pb");
}

} // namespace mqt
