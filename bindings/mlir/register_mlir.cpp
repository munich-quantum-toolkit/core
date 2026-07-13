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
#include "mlir/Compiler/Programs.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h> // NOLINT(misc-include-cleaner)

#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

namespace {

/**
 * @brief Resolve a Python path-like object to a filesystem path string.
 */
[[nodiscard]] std::string fspath(const nb::object& pathLike) {
  return nb::cast<std::string>(
      nb::module_::import_("os").attr("fspath")(pathLike));
}

/**
 * @brief Check whether @p input unambiguously looks like source text.
 */
[[nodiscard]] bool isSourceString(const std::string_view input) {
  return input.find('\n') != std::string_view::npos ||
         input.find("OPENQASM") != std::string_view::npos ||
         input.starts_with("module");
}

/**
 * @brief Construct a frontend program from a string containing source or path.
 */
[[nodiscard]] mlir::CompilerProgram
programFromString(const std::string& input) {
  if (isSourceString(input)) {
    if (input.find("OPENQASM") != std::string::npos) {
      return mlir::QCProgram::fromQASMString(input);
    }
    return mlir::QCProgram::fromMLIRString(input);
  }

  const std::filesystem::path path(input);
  if (path.empty()) {
    return mlir::QCProgram::fromMLIRString(input);
  }

  std::error_code error;
  const auto exists = std::filesystem::exists(path, error);
  if (error) {
    throw std::runtime_error("Failed to inspect path '" + input +
                             "': " + error.message());
  }

  const auto extension = path.extension().string();
  if (!exists) {
    if (extension == ".jeff" || extension == ".mlir" || extension == ".qasm") {
      throw std::runtime_error("Input file '" + input + "' does not exist.");
    }
    return mlir::QCProgram::fromMLIRString(input);
  }
  if (!std::filesystem::is_regular_file(path, error) || error) {
    throw std::runtime_error("Input path '" + input + "' is not a file.");
  }

  if (extension == ".jeff") {
    return mlir::JeffProgram::fromFile(path);
  }
  if (extension == ".mlir") {
    return mlir::QCProgram::fromMLIRFile(path);
  }
  if (extension == ".qasm") {
    return mlir::QCProgram::fromQASMFile(path);
  }
  throw std::runtime_error("Input file '" + input +
                           "' has unsupported extension '" + extension + "'.");
}

/**
 * @brief Convert a Python object to a compiler program.
 *
 * @details Program objects are copied by default so the high-level entry point
 * behaves like a conventional compiler function. Set @p inplace to transfer
 * ownership from a program object instead.
 */
[[nodiscard]] mlir::CompilerProgram programFromInput(const nb::object& program,
                                                     const bool inplace) {
  if (nb::isinstance<nb::str>(program)) {
    return programFromString(nb::cast<std::string>(program));
  }
  if (nb::hasattr(program, "__fspath__")) {
    return programFromString(fspath(program));
  }
  if (nb::isinstance<qc::QuantumComputation>(program)) {
    return mlir::QCProgram::fromQuantumComputation(
        nb::cast<const qc::QuantumComputation&>(program));
  }

  if (nb::isinstance<mlir::QCProgram>(program)) {
    auto& value = nb::cast<mlir::QCProgram&>(program);
    return inplace ? mlir::CompilerProgram(std::move(value))
                   : mlir::CompilerProgram(value.copy());
  }
  if (nb::isinstance<mlir::QCOProgram>(program)) {
    auto& value = nb::cast<mlir::QCOProgram&>(program);
    return inplace ? mlir::CompilerProgram(std::move(value))
                   : mlir::CompilerProgram(value.copy());
  }
  if (nb::isinstance<mlir::JeffProgram>(program)) {
    auto& value = nb::cast<mlir::JeffProgram&>(program);
    return inplace ? mlir::CompilerProgram(std::move(value))
                   : mlir::CompilerProgram(value.copy());
  }
  const auto programType =
      nb::cast<std::string>(program.type().attr("__name__"));
  const auto programModule =
      nb::cast<std::string>(program.type().attr("__module__"));
  if (programType == "QuantumCircuit" && programModule.starts_with("qiskit.")) {
    const auto computation = nb::cast<qc::QuantumComputation>(
        nb::module_::import_("mqt.core.load").attr("load")(program));
    return mlir::QCProgram::fromQuantumComputation(computation);
  }

  throw std::runtime_error("Program type " + programType +
                           " is not supported.");
}

/**
 * @brief Convert a C++ compiler program variant into its Python object.
 */
[[nodiscard]] nb::object toPython(mlir::CompilerProgram&& program) {
  return std::visit(
      []<typename T>(T&& value) -> nb::object {
        return nb::cast(std::forward<T>(value));
      },
      std::move(program));
}

/**
 * @brief Run the coordinated default pipeline and return a typed program.
 */
[[nodiscard]] nb::object
compileProgram(const nb::object& program, const mlir::ProgramFormat output,
               const bool inplace,
               const bool disableMergeSingleQubitRotationGates,
               const bool enableHadamardLifting, const bool enableTiming,
               const bool enableStatistics) {
  mlir::QuantumCompilerConfig config;
  config.disableMergeSingleQubitRotationGates =
      disableMergeSingleQubitRotationGates;
  config.enableHadamardLifting = enableHadamardLifting;
  config.enableTiming = enableTiming;
  config.enableStatistics = enableStatistics;

  return toPython(mlir::runDefaultPipeline(programFromInput(program, inplace),
                                           output, config));
}

template <class ProgramType>
[[nodiscard]] ProgramType copiedOrConsumed(ProgramType& program,
                                           const bool copy) {
  if (copy) {
    return program.copy();
  }
  return std::move(program);
}

} // namespace

NB_MODULE(MQT_CORE_MODULE_NAME, m) {
  m.doc() = R"pb(
MQT Core MLIR compiler bindings.
)pb";

  nb::module_::import_("typing");
  nb::module_::import_("mqt.core.ir");

  nb::enum_<mlir::QIRProfile>(m, "QIRProfile", "QIR target profiles.")
      .value("BASE", mlir::QIRProfile::Base, "The QIR Base Profile.")
      .value("ADAPTIVE", mlir::QIRProfile::Adaptive,
             "The QIR Adaptive Profile.");
  nb::enum_<mlir::ProgramFormat>(m, "OutputFormat",
                                 "Default compiler output formats.")
      .value("QC_IMPORT", mlir::ProgramFormat::QCImport,
             "QC directly after frontend import.")
      .value("QC", mlir::ProgramFormat::QC,
             "QC after the optimized QCO round trip.")
      .value("QCO", mlir::ProgramFormat::QCO, "Optimized QCO.")
      .value("JEFF", mlir::ProgramFormat::Jeff, "Serializable Jeff MLIR.")
      .value("QIR_BASE", mlir::ProgramFormat::QIRBase,
             "QIR for the Base Profile.")
      .value("QIR_ADAPTIVE", mlir::ProgramFormat::QIRAdaptive,
             "QIR for the Adaptive Profile.");

  auto program = nb::class_<mlir::Program>(m, "Program");
  program.def_prop_ro("is_valid", &mlir::Program::isValid)
      .def_prop_ro("ir", &mlir::Program::str)
      .def("__str__", &mlir::Program::str);

  auto qcProgram = nb::class_<mlir::QCProgram, mlir::Program>(m, "QCProgram");
  qcProgram
      .def_static("from_mlir_str", &mlir::QCProgram::fromMLIRString, "source"_a)
      .def_static(
          "from_mlir_file",
          [](const nb::object& path) {
            return mlir::QCProgram::fromMLIRFile(fspath(path));
          },
          "path"_a,
          nb::sig("def from_mlir_file(path: str | os.PathLike[str]) "
                  "-> QCProgram"))
      .def_static("from_qasm_str", &mlir::QCProgram::fromQASMString, "source"_a)
      .def_static(
          "from_qasm_file",
          [](const nb::object& path) {
            return mlir::QCProgram::fromQASMFile(fspath(path));
          },
          "path"_a,
          nb::sig("def from_qasm_file(path: str | os.PathLike[str]) "
                  "-> QCProgram"))
      .def_static("from_quantum_computation",
                  &mlir::QCProgram::fromQuantumComputation, "computation"_a,
                  nb::sig("def from_quantum_computation(computation: "
                          "mqt.core.ir.QuantumComputation) -> QCProgram"))
      .def_static(
          "from_qiskit",
          [](const nb::object& circuit) {
            const auto computation = nb::cast<qc::QuantumComputation>(
                nb::module_::import_("mqt.core.load").attr("load")(circuit));
            return mlir::QCProgram::fromQuantumComputation(computation);
          },
          "circuit"_a,
          nb::sig("def from_qiskit(circuit: qiskit.circuit.QuantumCircuit) "
                  "-> QCProgram"))
      .def("copy", &mlir::QCProgram::copy)
      .def("cleanup", &mlir::QCProgram::cleanup)
      .def(
          "to_qco",
          [](mlir::QCProgram& value, const bool copy) {
            auto source = copiedOrConsumed(value, copy);
            return std::move(source).intoQCO();
          },
          nb::kw_only(), "copy"_a = false)
      .def(
          "to_qir",
          [](mlir::QCProgram& value, const mlir::QIRProfile profile,
             const bool copy) {
            auto source = copiedOrConsumed(value, copy);
            return std::move(source).intoQIR(profile);
          },
          "profile"_a, nb::kw_only(), "copy"_a = false);

  auto qcoProgram =
      nb::class_<mlir::QCOProgram, mlir::Program>(m, "QCOProgram");
  qcoProgram
      .def_static("from_mlir_str", &mlir::QCOProgram::fromMLIRString,
                  "source"_a)
      .def_static(
          "from_mlir_file",
          [](const nb::object& path) {
            return mlir::QCOProgram::fromMLIRFile(fspath(path));
          },
          "path"_a,
          nb::sig("def from_mlir_file(path: str | os.PathLike[str]) "
                  "-> QCOProgram"))
      .def("copy", &mlir::QCOProgram::copy)
      .def("cleanup", &mlir::QCOProgram::cleanup)
      .def("optimize", &mlir::QCOProgram::optimize, nb::kw_only(),
           "merge_single_qubit_rotations"_a = true,
           "enable_hadamard_lifting"_a = false)
      .def(
          "to_qc",
          [](mlir::QCOProgram& value, const bool copy) {
            auto source = copiedOrConsumed(value, copy);
            return std::move(source).intoQC();
          },
          nb::kw_only(), "copy"_a = false)
      .def(
          "to_jeff",
          [](mlir::QCOProgram& value, const bool copy) {
            auto source = copiedOrConsumed(value, copy);
            return std::move(source).intoJeff();
          },
          nb::kw_only(), "copy"_a = false);

  auto jeffProgram =
      nb::class_<mlir::JeffProgram, mlir::Program>(m, "JeffProgram");
  jeffProgram
      .def_static(
          "from_file",
          [](const nb::object& path) {
            return mlir::JeffProgram::fromFile(fspath(path));
          },
          "path"_a,
          nb::sig("def from_file(path: str | os.PathLike[str]) "
                  "-> JeffProgram"))
      .def_static(
          "from_bytes",
          [](const nb::bytes& bytes) {
            return mlir::JeffProgram::fromBytes(nb::cast<std::string>(bytes));
          },
          "data"_a)
      .def("copy", &mlir::JeffProgram::copy)
      .def("cleanup", &mlir::JeffProgram::cleanup)
      .def("to_bytes",
           [](const mlir::JeffProgram& value) {
             const auto bytes = value.toBytes();
             return nb::bytes(bytes.data(), bytes.size());
           })
      .def(
          "write",
          [](const mlir::JeffProgram& value, const nb::object& path) {
            value.write(fspath(path));
          },
          "path"_a,
          nb::sig("def write(self, path: str | os.PathLike[str]) -> None"))
      .def(
          "to_qco",
          [](mlir::JeffProgram& value, const bool copy) {
            auto source = copiedOrConsumed(value, copy);
            return std::move(source).intoQCO();
          },
          nb::kw_only(), "copy"_a = false);

  auto qirProgram =
      nb::class_<mlir::QIRProgram, mlir::Program>(m, "QIRProgram");
  qirProgram.def("copy", &mlir::QIRProgram::copy)
      .def("cleanup", &mlir::QIRProgram::cleanup)
      .def_prop_ro("profile", &mlir::QIRProgram::profile)
      .def_prop_ro("llvm_ir", &mlir::QIRProgram::llvmIR);

  m.def("compile_program", &compileProgram, "program"_a, nb::kw_only(),
        "output"_a = mlir::ProgramFormat::QC, "inplace"_a = false,
        "disable_merge_single_qubit_rotation_gates"_a = false,
        "enable_hadamard_lifting"_a = false, "enable_timing"_a = false,
        "enable_statistics"_a = false,
        R"pb(
Run the coordinated default MQT compiler pipeline.

Input source strings, files, MQT `QuantumComputation` objects, Qiskit circuits,
and typed compiler programs can be combined with any supported output format.
Typed program inputs are copied by default; set `inplace=True` to consume them.
Use the typed programs directly to construct a custom pipeline stage by stage.
)pb");
}

} // namespace mqt
