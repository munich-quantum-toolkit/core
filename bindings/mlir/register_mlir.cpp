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
#include "mlir/Compiler/Programs.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>  // NOLINT(misc-include-cleaner)
#include <nanobind/stl/pair.h>        // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string.h>      // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string_view.h> // NOLINT(misc-include-cleaner)
#include <nanobind/stl/variant.h>     // NOLINT(misc-include-cleaner)
#include <nanobind/stl/vector.h>      // NOLINT(misc-include-cleaner)

#include <cctype>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

namespace {

template <class T> [[nodiscard]] T takeResult(std::optional<T>&& result) {
  if (!result) {
    throw std::runtime_error(
        "MLIR operation failed; see diagnostics for details.");
  }
  return *std::move(result);
}

void requireSuccess(const bool succeeded) {
  if (!succeeded) {
    throw std::runtime_error(
        "MLIR operation failed; see diagnostics for details.");
  }
}

template <auto Function> struct OptionalFunctionAdapter;

template <class T, class... Args, std::optional<T> (*Function)(Args...)>
struct OptionalFunctionAdapter<Function> {
  static T call(Args... args) {
    return takeResult(Function(std::forward<Args>(args)...));
  }
};

template <auto Method> struct OptionalMemberAdapter;

template <class Class, class T, class... Args,
          std::optional<T> (Class::*Method)(Args...) const>
struct OptionalMemberAdapter<Method> {
  static T call(const Class& self, Args... args) {
    return takeResult((self.*Method)(std::forward<Args>(args)...));
  }
};

template <auto Method> struct BooleanMemberAdapter;

template <class Class, class... Args, bool (Class::*Method)(Args...)>
struct BooleanMemberAdapter<Method> {
  static void call(Class& self, Args... args) {
    requireSuccess((self.*Method)(std::forward<Args>(args)...));
  }
};

template <class Class, class... Args, bool (Class::*Method)(Args...) const>
struct BooleanMemberAdapter<Method> {
  static void call(const Class& self, Args... args) {
    requireSuccess((self.*Method)(std::forward<Args>(args)...));
  }
};

void requireValid(const mlir::Program& program) {
  if (!program.isValid()) {
    throw std::runtime_error("This compiler program was consumed.");
  }
}

/**
 * @brief Check whether @p input unambiguously looks like source text.
 */
[[nodiscard]] bool isSourceString(const std::string_view input) {
  auto source = input;
  while (!source.empty() &&
         std::isspace(static_cast<unsigned char>(source.front())) != 0) {
    source.remove_prefix(1);
  }
  return input.find('\n') != std::string_view::npos ||
         input.find("OPENQASM") != std::string_view::npos ||
         (source.starts_with("module") && source.size() > 6U &&
          std::isspace(static_cast<unsigned char>(source[6])) != 0);
}

/**
 * @brief Construct a frontend program from a file path.
 */
[[nodiscard]] mlir::CompilerInput
programFromPath(const std::filesystem::path& path) {
  if (path.empty()) {
    throw std::runtime_error("Input path must not be empty.");
  }

  std::error_code error;
  const auto exists = std::filesystem::exists(path, error);
  if (error) {
    throw std::runtime_error("Failed to inspect path '" + path.string() +
                             "': " + error.message());
  }

  const auto extension = path.extension().string();
  if (!exists) {
    if (extension == ".jeff" || extension == ".mlir" || extension == ".qasm") {
      throw std::runtime_error("Input file '" + path.string() +
                               "' does not exist.");
    }
    throw std::runtime_error("Input file '" + path.string() +
                             "' does not exist.");
  }
  if (!std::filesystem::is_regular_file(path, error) || error) {
    throw std::runtime_error("Input path '" + path.string() +
                             "' is not a file.");
  }

  if (extension == ".jeff") {
    return takeResult(mlir::JeffProgram::fromFile(path));
  }
  if (extension == ".mlir") {
    return takeResult(mlir::QCProgram::fromMLIRFile(path));
  }
  if (extension == ".qasm") {
    return takeResult(mlir::QCProgram::fromQASMFile(path));
  }
  throw std::runtime_error("Input file '" + path.string() +
                           "' has unsupported extension '" + extension + "'.");
}

/**
 * @brief Construct a frontend program from a string containing source or path.
 */
[[nodiscard]] mlir::CompilerInput programFromString(const std::string& input) {
  if (isSourceString(input)) {
    if (input.find("OPENQASM") != std::string::npos) {
      return takeResult(mlir::QCProgram::fromQASMString(input));
    }
    return takeResult(mlir::QCProgram::fromMLIRString(input));
  }
  return programFromPath(std::filesystem::path(input));
}

/**
 * @brief Convert a Python object to a compiler program.
 *
 * @details Program objects are copied by default so the high-level entry point
 * behaves like a conventional compiler function. Set @p inplace to transfer
 * ownership from a program object instead.
 */
[[nodiscard]] mlir::CompilerInput programFromInput(const nb::object& program,
                                                   const bool inplace) {
  if (nb::isinstance<nb::str>(program)) {
    return programFromString(nb::cast<std::string>(program));
  }
  if (nb::hasattr(program, "__fspath__")) {
    return programFromPath(nb::cast<std::filesystem::path>(program));
  }
  if (nb::isinstance<qc::QuantumComputation>(program)) {
    return takeResult(mlir::QCProgram::fromQuantumComputation(
        nb::cast<const qc::QuantumComputation&>(program)));
  }

  if (nb::isinstance<mlir::QCProgram>(program)) {
    auto& value = nb::cast<mlir::QCProgram&>(program);
    return inplace ? mlir::CompilerInput(std::move(value))
                   : mlir::CompilerInput(value.copy());
  }
  if (nb::isinstance<mlir::QCOProgram>(program)) {
    auto& value = nb::cast<mlir::QCOProgram&>(program);
    return inplace ? mlir::CompilerInput(std::move(value))
                   : mlir::CompilerInput(value.copy());
  }
  if (nb::isinstance<mlir::JeffProgram>(program)) {
    auto& value = nb::cast<mlir::JeffProgram&>(program);
    return inplace ? mlir::CompilerInput(std::move(value))
                   : mlir::CompilerInput(value.copy());
  }
  const auto programType =
      nb::cast<std::string>(program.type().attr("__name__"));
  const auto sysModules =
      nb::cast<nb::dict>(nb::module_::import_("sys").attr("modules"));
  if (sysModules.contains("qiskit.circuit")) {
    const auto qiskitCircuit =
        nb::module_::import_("qiskit.circuit").attr("QuantumCircuit");
    if (nb::isinstance(program, qiskitCircuit)) {
      const auto computation = nb::cast<qc::QuantumComputation>(
          nb::module_::import_("mqt.core.load").attr("load")(program));
      return takeResult(mlir::QCProgram::fromQuantumComputation(computation));
    }
  }

  throw std::runtime_error("Program type " + programType +
                           " is not supported.");
}

/**
 * @brief Run the coordinated default pipeline and return a typed program.
 */
[[nodiscard]] mlir::CompilerProgram
compileProgram(const nb::object& program, const mlir::ProgramFormat output,
               const bool inplace, const std::string& qcoPipeline,
               const bool enableTiming, const bool enableStatistics) {
  return takeResult(mlir::runDefaultPipeline(programFromInput(program, inplace),
                                             output, qcoPipeline, enableTiming,
                                             enableStatistics));
}

template <class ProgramType>
[[nodiscard]] ProgramType copiedOrConsumed(ProgramType& program,
                                           const bool copy) {
  requireValid(program);
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
      .value("QCO", mlir::ProgramFormat::QCO,
             "QCO immediately after conversion, before optimization.")
      .value("QCO_OPTIMIZED", mlir::ProgramFormat::QCOOptimized,
             "QCO after the configured optimization pipeline.")
      .value("QC", mlir::ProgramFormat::QC,
             "QC after the optimized QCO round trip.")
      .value("JEFF", mlir::ProgramFormat::Jeff, "Serializable Jeff MLIR.")
      .value("QIR_BASE", mlir::ProgramFormat::QIRBase,
             "QIR for the Base Profile.")
      .value("QIR_ADAPTIVE", mlir::ProgramFormat::QIRAdaptive,
             "QIR for the Adaptive Profile.");

  auto program = nb::class_<mlir::Program>(
      m, "Program", R"pb(Base class for a typed MLIR compiler program.

Programs own their MLIR module. Conversions can consume a program; use
``is_valid`` to check whether it can still be used.)pb");
  program
      .def_prop_ro("is_valid", &mlir::Program::isValid,
                   "Whether this program still owns its module.")
      .def_prop_ro(
          "ir",
          [](const mlir::Program& value) {
            requireValid(value);
            return value.str();
          },
          "The textual MLIR representation of this program.")
      .def(
          "__str__",
          [](const mlir::Program& value) {
            requireValid(value);
            return value.str();
          },
          "Return the textual MLIR representation of this program.");

  auto qcProgram = nb::class_<mlir::QCProgram, mlir::Program>(
      m, "QCProgram", R"pb(A compiler program in the QC dialect.

QC programs use reference semantics and represent frontend quantum programs
before conversion to QCO.)pb");
  qcProgram
      .def_static(
          "from_mlir_str",
          &OptionalFunctionAdapter<&mlir::QCProgram::fromMLIRString>::call,
          "source"_a, "Parse a QC-dialect MLIR source string.")
      .def_static(
          "from_mlir_file",
          &OptionalFunctionAdapter<&mlir::QCProgram::fromMLIRFile>::call,
          "path"_a, "Parse QC-dialect MLIR from a file.")
      .def_static(
          "from_qasm_str",
          &OptionalFunctionAdapter<&mlir::QCProgram::fromQASMString>::call,
          "source"_a, "Translate an OpenQASM 3 source string to QC MLIR.")
      .def_static(
          "from_qasm_file",
          &OptionalFunctionAdapter<&mlir::QCProgram::fromQASMFile>::call,
          "path"_a, "Translate an OpenQASM 3 file to QC MLIR.")
      .def_static("from_quantum_computation",
                  &OptionalFunctionAdapter<
                      &mlir::QCProgram::fromQuantumComputation>::call,
                  "computation"_a,
                  "Translate an MQT QuantumComputation to QC MLIR.")
      .def_static(
          "from_qiskit",
          [](const nb::object& circuit) {
            const auto computation = nb::cast<qc::QuantumComputation>(
                nb::module_::import_("mqt.core.load").attr("load")(circuit));
            return takeResult(
                mlir::QCProgram::fromQuantumComputation(computation));
          },
          "circuit"_a,
          nb::sig("def from_qiskit(circuit: qiskit.circuit.QuantumCircuit) "
                  "-> QCProgram"),
          "Translate a Qiskit QuantumCircuit to QC MLIR.")
      .def("copy", &mlir::QCProgram::copy,
           "Return an independent copy of this program.")
      .def("cleanup", &BooleanMemberAdapter<&mlir::QCProgram::cleanup>::call,
           "Run the standard QC cleanup pipeline in place.")
      .def(
          "to_qco",
          [](mlir::QCProgram& value, const bool copy) {
            auto source = copiedOrConsumed(value, copy);
            return takeResult(std::move(source).intoQCO());
          },
          nb::kw_only(), "copy"_a = false,
          "Convert this program to QCO. Set ``copy=True`` to preserve it.")
      .def(
          "to_qir",
          [](mlir::QCProgram& value, const mlir::QIRProfile profile,
             const bool copy) {
            auto source = copiedOrConsumed(value, copy);
            return takeResult(std::move(source).intoQIR(profile));
          },
          "profile"_a, nb::kw_only(), "copy"_a = false,
          "Lower this program to QIR for the requested profile.");

  auto qcoProgram = nb::class_<mlir::QCOProgram, mlir::Program>(
      m, "QCOProgram", R"pb(A compiler program in the QCO dialect.

QCO programs use value semantics and expose optimization and transformation
operations.)pb");
  qcoProgram
      .def_static(
          "from_mlir_str",
          &OptionalFunctionAdapter<&mlir::QCOProgram::fromMLIRString>::call,
          "source"_a, "Parse a QCO-dialect MLIR source string.")
      .def_static(
          "from_mlir_file",
          &OptionalFunctionAdapter<&mlir::QCOProgram::fromMLIRFile>::call,
          "path"_a, "Parse QCO-dialect MLIR from a file.")
      .def("copy", &mlir::QCOProgram::copy,
           "Return an independent copy of this program.")
      .def("cleanup", &BooleanMemberAdapter<&mlir::QCOProgram::cleanup>::call,
           "Run the standard QCO cleanup pipeline in place.")
      .def("run_pass_pipeline",
           &BooleanMemberAdapter<&mlir::QCOProgram::runPassPipeline>::call,
           "pipeline"_a, nb::kw_only(), "enable_timing"_a = false,
           "enable_statistics"_a = false,
           "Run a textual MLIR pass pipeline in place.")
      .def("merge_single_qubit_rotation_gates",
           &BooleanMemberAdapter<
               &mlir::QCOProgram::mergeSingleQubitRotationGates>::call,
           "Merge compatible consecutive single-qubit rotation gates.")
      .def(
          "fuse_single_qubit_unitary_runs",
          &BooleanMemberAdapter<
              &mlir::QCOProgram::fuseSingleQubitUnitaryRuns>::call,
          nb::kw_only(), "basis"_a = "zyz",
          "Fuse single-qubit unitary runs into the chosen decomposition basis.")
      .def("fuse_two_qubit_unitary_runs",
           &BooleanMemberAdapter<
               &mlir::QCOProgram::fuseTwoQubitUnitaryRuns>::call,
           nb::kw_only(), "native_gates"_a = "",
           "Lower unitaries to a comma-separated native gate menu.")
      .def("unroll_quantum_loops",
           &BooleanMemberAdapter<&mlir::QCOProgram::unrollQuantumLoops>::call,
           nb::kw_only(), "unroll_factor"_a = -1,
           "Unroll quantum loops, optionally using a maximum unroll factor.")
      .def("lift_hadamards",
           &BooleanMemberAdapter<&mlir::QCOProgram::liftHadamards>::call,
           "Move Hadamard gates through compatible operations.")
      .def(
          "place_and_route",
          [](mlir::QCOProgram& value,
             const std::vector<std::pair<std::size_t, std::size_t>>& coupling,
             const std::size_t nlookahead, const float alpha,
             const float lambda, const std::size_t niterations,
             const std::size_t ntrials, const std::size_t seed) {
            requireSuccess(value.placeAndRoute(std::span(coupling), nlookahead,
                                               alpha, lambda, niterations,
                                               ntrials, seed));
          },
          "coupling"_a, nb::kw_only(), "nlookahead"_a = 1, "alpha"_a = 1.F,
          "lambda_"_a = 0.5F, "niterations"_a = 1, "ntrials"_a = 4,
          "seed"_a = 42, "Place and route the program for a coupling graph.")
      .def(
          "to_qc",
          [](mlir::QCOProgram& value, const bool copy) {
            auto source = copiedOrConsumed(value, copy);
            return takeResult(std::move(source).intoQC());
          },
          nb::kw_only(), "copy"_a = false,
          "Convert this program to QC. Set ``copy=True`` to preserve it.")
      .def(
          "to_jeff",
          [](mlir::QCOProgram& value, const bool copy) {
            auto source = copiedOrConsumed(value, copy);
            return takeResult(std::move(source).intoJeff());
          },
          nb::kw_only(), "copy"_a = false,
          "Serialize this program as Jeff. Set ``copy=True`` to preserve it.");

  auto jeffProgram = nb::class_<mlir::JeffProgram, mlir::Program>(
      m, "JeffProgram", R"pb(A serialized Jeff compiler program.

Jeff programs can be stored as bytes or files and converted back to QCO for
further compilation.)pb");
  jeffProgram
      .def_static("from_file",
                  &OptionalFunctionAdapter<&mlir::JeffProgram::fromFile>::call,
                  "path"_a, "Read a Jeff program from a file.")
      .def_static(
          "from_bytes",
          [](const nb::bytes& bytes) {
            const auto view =
                std::span(reinterpret_cast<const std::byte*>(bytes.c_str()),
                          bytes.size());
            return takeResult(mlir::JeffProgram::fromBytes(view));
          },
          "data"_a, "Deserialize a Jeff program from bytes.")
      .def("copy", &mlir::JeffProgram::copy,
           "Return an independent copy of this program.")
      .def("cleanup", &BooleanMemberAdapter<&mlir::JeffProgram::cleanup>::call,
           "Run the standard Jeff cleanup pipeline in place.")
      .def(
          "to_bytes",
          [](const mlir::JeffProgram& value) {
            requireValid(value);
            const auto bytes = value.toBytes();
            return nb::bytes(bytes.data(), bytes.size());
          },
          "Serialize this program to its Jeff byte representation.")
      .def("write", &BooleanMemberAdapter<&mlir::JeffProgram::write>::call,
           "path"_a, "Write this program to a Jeff file.")
      .def(
          "to_qco",
          [](mlir::JeffProgram& value, const bool copy) {
            auto source = copiedOrConsumed(value, copy);
            return takeResult(std::move(source).intoQCO());
          },
          nb::kw_only(), "copy"_a = false,
          "Deserialize this program to QCO. Set ``copy=True`` to preserve it.");

  auto qirProgram = nb::class_<mlir::QIRProgram, mlir::Program>(
      m, "QIRProgram", R"pb(A compiler program lowered to Quantum IR.

QIR programs retain their target profile and can be emitted as LLVM IR or
LLVM bitcode.)pb");
  qirProgram
      .def("copy", &mlir::QIRProgram::copy,
           "Return an independent copy of this program.")
      .def("cleanup", &BooleanMemberAdapter<&mlir::QIRProgram::cleanup>::call,
           "Run the standard QIR cleanup pipeline in place.")
      .def_prop_ro("profile", &mlir::QIRProgram::profile,
                   "The QIR target profile used to produce this program.")
      .def_prop_ro("llvm_ir",
                   &OptionalMemberAdapter<&mlir::QIRProgram::llvmIR>::call,
                   "The program as textual LLVM IR.")
      .def(
          "to_bitcode",
          [](const mlir::QIRProgram& value) {
            requireValid(value);
            const auto bytes = takeResult(value.toBitcode());
            return nb::bytes(reinterpret_cast<const char*>(bytes.data()),
                             bytes.size());
          },
          "Serialize this program as LLVM bitcode.")
      .def("write_bitcode",
           &BooleanMemberAdapter<&mlir::QIRProgram::writeBitcode>::call,
           "path"_a, "Write this program as LLVM bitcode.");

  m.def("compile_program", &compileProgram, "program"_a, nb::kw_only(),
        "output"_a = mlir::ProgramFormat::QC, "inplace"_a = false,
        "qco_pipeline"_a = "mqt-qco-default", "enable_timing"_a = false,
        "enable_statistics"_a = false,
        R"pb(
Run the coordinated default MQT compiler pipeline.

Input source strings, files, MQT `QuantumComputation` objects, Qiskit circuits,
and typed compiler programs can be combined with any supported output format.
Typed program inputs are copied by default; set `inplace=True` to consume them.
Use the typed programs directly to construct a custom pipeline stage by stage.

Args:
    program: Source text, a file path, a circuit, or a typed compiler program.
    output: The requested output stage of the compiler pipeline.
    inplace: Whether a typed input program may be consumed.
    qco_pipeline: The QCO optimization pipeline to run.
    enable_timing: Whether to collect pass timing information.
    enable_statistics: Whether to collect pass statistics.

Returns:
    A typed compiler program for the requested output format.
)pb");
}

} // namespace mqt
