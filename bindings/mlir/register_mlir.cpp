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
#include "mlir/Dialect/QC/Translation/TranslateQuantumComputationToQC.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Support/Passes.h"
#include "qasm3/Exception.hpp"
#include "qasm3/Importer.hpp"

#include <jeff/IR/JeffDialect.h>
#include <jeff/Translation/Deserialize.hpp>
#include <llvm/ADT/StringRef.h>
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
#include <mlir/Support/LogicalResult.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h> // NOLINT(misc-include-cleaner)

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

namespace {

/**
 * @brief Construct and initialize the MLIR context used by the compiler.
 */
[[nodiscard]] auto createCompilerContext()
    -> std::unique_ptr<mlir::MLIRContext> {
  mlir::DialectRegistry registry;
  registry.insert<mlir::qc::QCDialect, mlir::qco::QCODialect,
                  mlir::qtensor::QTensorDialect, mlir::arith::ArithDialect,
                  mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
                  mlir::scf::SCFDialect, mlir::LLVM::LLVMDialect,
                  mlir::memref::MemRefDialect, mlir::jeff::JeffDialect>();

  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

/**
 * @brief Convert a `QuantumComputation` to a QC MLIR module.
 */
[[nodiscard]] auto
moduleFromQuantumComputation(mlir::MLIRContext* context,
                             const qc::QuantumComputation& computation)
    -> mlir::OwningOpRef<mlir::ModuleOp> {
  auto module = mlir::translateQuantumComputationToQC(context, computation);
  if (!module) {
    throw std::runtime_error(
        "Failed to translate quantum computation to MLIR.");
  }
  return module;
}

/**
 * @brief Parse MLIR source text into a module.
 */
[[nodiscard]] auto parseMlirText(mlir::MLIRContext* context,
                                 const std::string& text)
    -> mlir::OwningOpRef<mlir::ModuleOp> {
  return mlir::parseSourceString<mlir::ModuleOp>(llvm::StringRef(text),
                                                 context);
}

/**
 * @brief Parse OpenQASM source text into a QC MLIR module.
 */
[[nodiscard]] auto moduleFromQasmString(mlir::MLIRContext* context,
                                        const std::string& qasmSource)
    -> mlir::OwningOpRef<mlir::ModuleOp> {
  try {
    const auto computation = qasm3::Importer::imports(qasmSource);
    return moduleFromQuantumComputation(context, computation);
  } catch (const qasm3::CompilerError& exception) {
    throw std::runtime_error(std::string("Failed to parse OpenQASM input: ") +
                             exception.what());
  } catch (const std::exception& exception) {
    throw std::runtime_error(std::string("Failed to import OpenQASM input: ") +
                             exception.what());
  }
}

/**
 * @brief Parse an OpenQASM file into a QC MLIR module.
 */
[[nodiscard]] auto moduleFromQasmFile(mlir::MLIRContext* context,
                                      const std::string& path)
    -> mlir::OwningOpRef<mlir::ModuleOp> {
  try {
    const auto computation = qasm3::Importer::importf(path);
    return moduleFromQuantumComputation(context, computation);
  } catch (const qasm3::CompilerError& exception) {
    throw std::runtime_error(std::string("Failed to parse OpenQASM file '") +
                             path + "': " + exception.what());
  } catch (const std::exception& exception) {
    throw std::runtime_error(std::string("Failed to import OpenQASM file '") +
                             path + "': " + exception.what());
  }
}

/**
 * @brief Convert a deserialized jeff module to QC dialect.
 */
void convertJeffModuleToQC(mlir::ModuleOp module) {
  mlir::PassManager passManager(module.getContext());
  passManager.addPass(mlir::createJeffToQCO());
  ::populateQCOCleanupPipeline(passManager);
  passManager.addPass(mlir::createQCOToQC());
  ::populateQCCleanupPipeline(passManager);

  if (mlir::failed(passManager.run(module))) {
    throw std::runtime_error("Failed to convert jeff input to QC MLIR.");
  }
}

/**
 * @brief Parse a `.jeff` file and convert it to a QC MLIR module.
 */
[[nodiscard]] auto moduleFromJeffFile(mlir::MLIRContext* context,
                                      const std::string& path)
    -> mlir::OwningOpRef<mlir::ModuleOp> {
  auto module = deserializeFromFile(context, path);
  if (!module) {
    throw std::runtime_error(std::string("Failed to deserialize jeff file '") +
                             path + "'.");
  }
  convertJeffModuleToQC(module.get());
  return module;
}

/**
 * @brief Read a text file into a string.
 */
[[nodiscard]] auto readTextFile(const std::string& path) -> std::string {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error(std::string("Failed to open file '") + path +
                             "'.");
  }
  return {std::istreambuf_iterator<char>(file),
          std::istreambuf_iterator<char>()};
}

/**
 * @brief Lowercase a file extension for robust comparisons.
 */
[[nodiscard]] auto toLower(std::string value) -> std::string {
  std::ranges::transform(value, value.begin(), [](const unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

/**
 * @brief Try to parse in-memory source either as OpenQASM or MLIR.
 */
[[nodiscard]] auto moduleFromTextInput(mlir::MLIRContext* context,
                                       const std::string& input)
    -> mlir::OwningOpRef<mlir::ModuleOp> {
  if (input.find("OPENQASM") != std::string::npos) {
    return moduleFromQasmString(context, input);
  }

  if (auto module = parseMlirText(context, input)) {
    return module;
  }

  try {
    return moduleFromQasmString(context, input);
  } catch (const std::exception&) {
    throw std::runtime_error("Failed to parse input as MLIR or OpenQASM.");
  }
}

/**
 * @brief Resolve string/path-like input to an MLIR module.
 */
[[nodiscard]] auto moduleFromStringOrPathLike(mlir::MLIRContext* context,
                                              const std::string& input,
                                              const bool pathLikeObject)
    -> mlir::OwningOpRef<mlir::ModuleOp> {
  const auto path = std::filesystem::path(input);
  std::error_code errorCode;
  const auto inputExists = std::filesystem::exists(path, errorCode);

  if (errorCode) {
    if (pathLikeObject) {
      throw std::runtime_error(std::string("Failed to inspect input path '") +
                               input + "': " + errorCode.message());
    }
    return moduleFromTextInput(context, input);
  }

  if (inputExists) {
    if (!std::filesystem::is_regular_file(path, errorCode)) {
      if (errorCode) {
        throw std::runtime_error(std::string("Failed to inspect input path '") +
                                 input + "': " + errorCode.message());
      }
      throw std::runtime_error(std::string("Input path '") + input +
                               "' is not a file.");
    }

    const auto extension = toLower(path.extension().string());
    if (extension == ".qasm") {
      return moduleFromQasmFile(context, input);
    }
    if (extension == ".jeff") {
      return moduleFromJeffFile(context, input);
    }

    return moduleFromTextInput(context, readTextFile(input));
  }

  if (pathLikeObject || path.extension() == ".qasm" ||
      path.extension() == ".mlir" || path.extension() == ".jeff") {
    throw std::runtime_error(std::string("Input file '") + input +
                             "' does not exist.");
  }

  return moduleFromTextInput(context, input);
}

/**
 * @brief Convert a generic Python object to an MLIR module.
 */
[[nodiscard]] auto moduleFromInputProgram(mlir::MLIRContext* context,
                                          const nb::object& program)
    -> mlir::OwningOpRef<mlir::ModuleOp> {
  if (nb::isinstance<qc::QuantumComputation>(program)) {
    const auto& computation = nb::cast<const qc::QuantumComputation&>(program);
    return moduleFromQuantumComputation(context, computation);
  }

  if (nb::isinstance<nb::str>(program)) {
    return moduleFromStringOrPathLike(context, nb::cast<std::string>(program),
                                      false);
  }

  if (nb::hasattr(program, "__fspath__")) {
    const auto pathAsString = nb::cast<std::string>(
        nb::module_::import_("os").attr("fspath")(program));
    return moduleFromStringOrPathLike(context, pathAsString, true);
  }

  // Fallback route for optional Qiskit circuits and other Python-side
  // converters handled by `mqt.core.load`.
  const auto loaded =
      nb::module_::import_("mqt.core.load").attr("load")(program);
  const auto& computation = nb::cast<const qc::QuantumComputation&>(loaded);
  return moduleFromQuantumComputation(context, computation);
}

/**
 * @brief Compile a Python-provided quantum program and return MLIR text.
 */
[[nodiscard]] auto
compileProgram(const nb::object& program, const bool convertToQIR,
               const bool disableMergeSingleQubitRotationGates,
               const bool enableHadamardLifting, const bool enableTiming,
               const bool enableStatistics) -> std::string {
  auto context = createCompilerContext();
  auto module = moduleFromInputProgram(context.get(), program);

  mlir::QuantumCompilerConfig config;
  config.convertToQIR = convertToQIR;
  config.disableMergeSingleQubitRotationGates =
      disableMergeSingleQubitRotationGates;
  config.enableHadamardLifting = enableHadamardLifting;
  config.enableTiming = enableTiming;
  config.enableStatistics = enableStatistics;

  const mlir::QuantumCompilerPipeline pipeline(config);
  if (mlir::failed(pipeline.runPipeline(module.get()))) {
    throw std::runtime_error("Compilation pipeline failed.");
  }

  return mlir::captureIR(module.get());
}

} // namespace

NB_MODULE(MQT_CORE_MODULE_NAME, m) {
  m.doc() = R"pb(
MQT Core MLIR compiler bindings.
)pb";

  nb::module_::import_("mqt.core.ir");

  m.def("compile_program", &compileProgram, "program"_a, nb::kw_only(),
        "convert_to_qir"_a = false,
        "disable_merge_single_qubit_rotation_gates"_a = false,
        "enable_hadamard_lifting"_a = false, "enable_timing"_a = false,
        "enable_statistics"_a = false,
        R"pb(
Compile an input quantum program with the MQT MLIR compiler pipeline.

Args:
    program: Input program in one of the supported forms:
        - :class:`mqt.core.ir.QuantumComputation`
        - OpenQASM source text
        - Path to `.qasm`, `.mlir`, or `.jeff` files
        - Qiskit :class:`~qiskit.circuit.QuantumCircuit`
        - MLIR source text
    convert_to_qir: Whether to lower the result to QIR.
    disable_merge_single_qubit_rotation_gates: Disable quaternion-based
        rotation merging.
    enable_hadamard_lifting: Enable Hadamard lifting optimization.
    enable_timing: Enable MLIR pass timing.
    enable_statistics: Enable MLIR pass statistics.

Returns:
    The final MLIR module as text.
)pb");
}

} // namespace mqt
