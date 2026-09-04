/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/DDDefinitions.hpp"
#include "dd/Edge.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "mlir/Compiler/Programs.h"
#include "mlir/Compiler/QDMIAdapter.h"
#include "mlir/Compiler/Target.h"
#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/QCO/Utils/DDFunctionality.h"
#include "mlir/bench/Generate.h"
#include "qdmi/Client.hpp"
#include "qdmi/driver/SessionConfig.hpp"
#include "qiskit/Qiskit.h"

#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LogicalResult.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <cctype>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

using DenseVector = nb::ndarray<nb::numpy, std::complex<dd::fp>, nb::ndim<1>,
                                nb::c_contig, nb::device::cpu>;
using DenseMatrix = nb::ndarray<nb::numpy, std::complex<dd::fp>, nb::ndim<2>,
                                nb::c_contig, nb::device::cpu>;

template <class T>
[[nodiscard]] static T takeResult(std::optional<T>&& result) {
  if (!result) {
    throw std::runtime_error(
        "MLIR operation failed; see diagnostics for details.");
  }
  return *std::move(result);
}

template <class T>
[[nodiscard]] static T takeResult(llvm::Expected<T>&& result) {
  if (!result) {
    const auto message = llvm::toString(result.takeError());
    throw nb::value_error(message.c_str());
  }
  return *std::move(result);
}

template <class T>
static void constructFromExpected(T& self, llvm::Expected<T>&& result) {
  std::construct_at(&self, takeResult(std::move(result)));
}

static void requireSuccess(const bool succeeded) {
  if (!succeeded) {
    throw std::runtime_error(
        "MLIR operation failed; see diagnostics for details.");
  }
}

namespace {
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
} // namespace

static void requireValid(const mlir::Program& program) {
  if (!program.isValid()) {
    throw std::runtime_error(
        "This compiler program has already been consumed.");
  }
}

[[nodiscard]] static mlir::func::FuncOp
entryFunc(const mlir::QCOProgram& program) {
  requireValid(program);
  auto func = mlir::mqt::getEntryPoint(program.module());
  if (!func) {
    throw nb::value_error("QCO program has no func.func entry point");
  }
  return func;
}

[[nodiscard]] static std::mt19937_64 makeRng(const uint64_t seed) {
  if (seed == 0) {
    return std::mt19937_64(std::random_device{}());
  }
  return std::mt19937_64(seed);
}

/// Run @p fn under a diagnostic handler and raise `ValueError` on failure,
/// appending any emitted MLIR diagnostics to @p message.
template <typename Fn>
[[nodiscard]] static auto takeFailureOr(mlir::MLIRContext* context,
                                        const char* message, Fn&& fn) {
  std::string diagnostics;
  const mlir::ScopedDiagnosticHandler handler(
      context, [&](mlir::Diagnostic& diag) {
        if (!diagnostics.empty()) {
          diagnostics.push_back('\n');
        }
        llvm::raw_string_ostream os(diagnostics);
        if (!llvm::isa<mlir::UnknownLoc>(diag.getLocation())) {
          os << diag.getLocation() << ": ";
        }
        os << diag;
        return mlir::success();
      });
  auto result = std::forward<Fn>(fn)();
  if (mlir::failed(result)) {
    std::string full = message;
    if (!diagnostics.empty()) {
      full.append(": ").append(diagnostics);
    }
    throw nb::value_error(full.c_str());
  }
  return *std::move(result);
}

template <class ProgramType>
[[nodiscard]] static ProgramType copiedOrConsumed(ProgramType& program,
                                                  const bool copy) {
  requireValid(program);
  if (copy) {
    return program.copy();
  }
  return std::move(program);
}

/// Check whether @p input unambiguously looks like source text.
[[nodiscard]] static bool isSourceString(const std::string_view input) {
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

/// Construct a frontend program from a file path.
[[nodiscard]] static mlir::CompilerInput
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
  if (!exists) {
    throw std::runtime_error("Input file '" + path.string() +
                             "' does not exist.");
  }
  if (!std::filesystem::is_regular_file(path, error) || error) {
    throw std::runtime_error("Input path '" + path.string() +
                             "' is not a file.");
  }

  const auto extension = path.extension().string();
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

/// Construct a frontend program from a string containing source or path.
[[nodiscard]] static mlir::CompilerInput
programFromString(const std::string& input) {
  if (isSourceString(input)) {
    if (input.find("OPENQASM") != std::string::npos) {
      return takeResult(mlir::QCProgram::fromQASMString(input));
    }
    return takeResult(mlir::QCProgram::fromMLIRString(input));
  }
  return programFromPath(std::filesystem::path(input));
}

/// Convert a Python object to a compiler program.
///
/// Program objects are copied by default so the high-level entry point
/// behaves like a conventional compiler function. Set @p inplace to transfer
/// ownership from a program object instead.
[[nodiscard]] static mlir::CompilerInput
programFromInput(const nb::object& program, const bool inplace) {
  if (nb::isinstance<nb::str>(program)) {
    return programFromString(nb::cast<std::string>(program));
  }
  if (nb::hasattr(program, "__fspath__")) {
    return programFromPath(nb::cast<std::filesystem::path>(program));
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
  if (nb::isinstance<mlir::OpenQASMProgram>(program)) {
    return {nb::cast<const mlir::OpenQASMProgram&>(program)};
  }

  const auto programType =
      nb::cast<std::string>(program.type().attr("__name__"));
  const auto sysModules =
      nb::cast<nb::dict>(nb::module_::import_("sys").attr("modules"));
  if (sysModules.contains("qiskit.circuit")) {
    const auto qiskitCircuit =
        nb::module_::import_("qiskit.circuit").attr("QuantumCircuit");
    if (nb::isinstance(program, qiskitCircuit)) {
      return bindings::qiskit::importCircuit(program);
    }
  }
  throw std::runtime_error("Program type " + programType +
                           " is not supported.");
}

/// Run the coordinated default pipeline and return a typed program.
[[nodiscard]] static mlir::CompilerProgram
compileProgram(const nb::object& program, const mlir::ProgramFormat output,
               const bool inplace, const mlir::CompilerTarget* const target,
               const std::string& qcoPipeline, const bool enableTiming,
               const bool enableStatistics) {
  return takeResult(mlir::runDefaultPipeline(programFromInput(program, inplace),
                                             output, target, qcoPipeline,
                                             enableTiming, enableStatistics));
}

template <class Function>
[[nodiscard]] static auto withQCOProgram(const nb::object& program,
                                         Function&& function) {
  if (nb::isinstance<mlir::QCOProgram>(program)) {
    return std::forward<Function>(function)(
        nb::cast<const mlir::QCOProgram&>(program));
  }
  auto compiled = takeResult(mlir::runDefaultPipeline(
      programFromInput(program, false), mlir::ProgramFormat::QCO));
  return std::forward<Function>(function)(std::get<mlir::QCOProgram>(compiled));
}

[[nodiscard]] static dd::MatrixDD
buildQCOFunctionality(const mlir::QCOProgram& program, dd::Package& ddPackage) {
  auto func = entryFunc(program);
  return takeFailureOr(
      func.getContext(), "cannot build DD functionality for this QCO program",
      [&] { return mlir::qco::buildFunctionality(func, ddPackage); });
}

[[nodiscard]] static dd::VectorDD simulateQCO(const mlir::QCOProgram& program,
                                              const dd::VectorDD& initialState,
                                              dd::Package& ddPackage,
                                              uint64_t seed) {
  if (dd::VectorDD::trackingRequired(initialState) &&
      !ddPackage.getRootSet<dd::vNode>().contains(initialState)) {
    throw nb::value_error(
        "initial_state must have a live reference in dd_package");
  }
  auto func = entryFunc(program);
  auto rng = makeRng(seed);
  return takeFailureOr(
      func.getContext(), "cannot simulate this QCO program",
      [&] { return mlir::qco::simulate(func, initialState, ddPackage, rng); });
}

[[nodiscard]] static std::map<std::string, size_t>
sampleQCO(const mlir::QCOProgram& program, size_t shots, uint64_t seed) {
  auto func = entryFunc(program);
  return takeFailureOr(func.getContext(), "cannot sample this QCO program",
                       [&] { return mlir::qco::sample(func, shots, seed); });
}

[[nodiscard]] static DenseVector toDenseVector(const dd::VectorDD& state) {
  if (!state.isTerminal()) {
    const auto numQubits = static_cast<size_t>(state.p->v) + 1U;
    if (numQubits >= std::numeric_limits<size_t>::digits ||
        (size_t{1} << numQubits) >
            std::numeric_limits<size_t>::max() / sizeof(std::complex<dd::fp>)) {
      throw nb::value_error(
          "dense statevector dimensions exceed addressable memory");
    }
  }
  auto dataPtr = std::make_unique<dd::CVec>(state.getVector());
  auto* const data = dataPtr->data();
  const auto size = dataPtr->size();
  const nb::capsule owner(dataPtr.get(), [](void* ptr) noexcept {
    delete static_cast<dd::CVec*>(ptr);
  });
  [[maybe_unused]] const auto* const releasedDataPtr = dataPtr.release();
  return DenseVector(data, {size}, owner);
}

[[nodiscard]] static DenseMatrix toDenseMatrix(const dd::MatrixDD& matrix,
                                               size_t numQubits) {
  if (numQubits >= std::numeric_limits<size_t>::digits) {
    throw nb::value_error("dense unitary dimensions exceed addressable memory");
  }

  const size_t dim = size_t{1} << numQubits;
  if (dim >
      std::numeric_limits<size_t>::max() / sizeof(std::complex<dd::fp>) / dim) {
    throw nb::value_error("dense unitary dimensions exceed addressable memory");
  }
  auto dataPtr = std::make_unique<std::complex<dd::fp>[]>(dim * dim);
  matrix.traverseMatrix(
      std::complex<dd::fp>{1., 0.}, 0ULL, 0ULL,
      [&dataPtr, dim](size_t i, size_t j, const std::complex<dd::fp>& value) {
        dataPtr[(i * dim) + j] = value;
      },
      numQubits);
  auto* const data = dataPtr.get();
  const nb::capsule owner(data, [](void* ptr) noexcept {
    delete[] static_cast<std::complex<dd::fp>*>(ptr);
  });
  [[maybe_unused]] const auto* const releasedDataPtr = dataPtr.release();
  return DenseMatrix(data, {dim, dim}, owner);
}

[[nodiscard]] static DenseMatrix
buildDenseFunctionality(const nb::object& program) {
  return withQCOProgram(program, [](const mlir::QCOProgram& qco) {
    dd::Package ddPackage(0);
    const auto matrix = buildQCOFunctionality(qco, ddPackage);
    return toDenseMatrix(matrix, ddPackage.qubits());
  });
}

[[nodiscard]] static DenseVector simulateDense(const nb::object& program) {
  return withQCOProgram(program, [](const mlir::QCOProgram& qco) {
    dd::Package ddPackage(0);
    auto func = entryFunc(qco);
    const auto state = takeFailureOr(
        func.getContext(), "cannot simulate this QCO program",
        [&] { return mlir::qco::simulateStatevector(func, ddPackage); });
    return toDenseVector(state);
  });
}

[[nodiscard]] static std::map<std::string, size_t>
sample(const nb::object& program, size_t shots, uint64_t seed) {
  return withQCOProgram(program, [&](const mlir::QCOProgram& qco) {
    return sampleQCO(qco, shots, seed);
  });
}

[[nodiscard]] static mlir::QCProgram
generateBenchmark(const std::string_view instanceSpecificationJSON) {
  auto generated = bench::generate(instanceSpecificationJSON);
  if (!generated) {
    throw std::runtime_error("failed to generate benchmark");
  }
  return std::move(generated->program);
}

NB_MODULE(MQT_CORE_MODULE_NAME, m) {
  m.doc() = "MQT Core MLIR compiler bindings.";

  nb::module_::import_("typing");
  nb::module_::import_("mqt.core.qdmi");

  m.def("_generate_benchmark", &generateBenchmark,
        "instance_specification_json"_a,
        "Generate the QC program described by an instance specification.");

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
      .value("OPENQASM3", mlir::ProgramFormat::OpenQASM3,
             "OpenQASM 3 after the optimized QCO round trip.")
      .value("JEFF", mlir::ProgramFormat::Jeff, "Serializable ``jeff`` MLIR.")
      .value("QIR_BASE", mlir::ProgramFormat::QIRBase,
             "QIR for the Base Profile.")
      .value("QIR_ADAPTIVE", mlir::ProgramFormat::QIRAdaptive,
             "QIR for the Adaptive Profile.");

  auto compilerTarget = nb::class_<mlir::CompilerTarget>(
      m, "CompilerTarget", R"pb(Immutable MLIR compiler target.

Every target has either all-to-all or explicitly enumerated connectivity and
either unrestricted or explicitly enumerated native-operation support.)pb");

  auto durationUnit = nb::class_<mlir::CompilerTarget::DurationUnit>(
      compilerTarget, "DurationUnit", "Unit for raw target timing metadata.");
  durationUnit
      .def(
          "__init__",
          [](mlir::CompilerTarget::DurationUnit& self, std::string unit,
             const double scaleFactor) {
            constructFromExpected(self,
                                  mlir::CompilerTarget::DurationUnit::create(
                                      std::move(unit), scaleFactor));
          },
          "unit"_a, "scale_factor"_a)
      .def_prop_ro(
          "unit",
          [](const mlir::CompilerTarget::DurationUnit& value) {
            return value.unit().str();
          },
          "The reported duration unit.")
      .def_prop_ro("scale_factor",
                   &mlir::CompilerTarget::DurationUnit::scaleFactor,
                   "The multiplier applied to raw timing values.");

  auto targetSite = nb::class_<mlir::CompilerTarget::Site>(
      compilerTarget, "Site", "A hardware site and its optional metadata.");
  targetSite
      .def(
          "__init__",
          [](mlir::CompilerTarget::Site& self,
             const mlir::CompilerTarget::SiteId siteId,
             std::optional<std::string> name, const std::optional<uint64_t> t1,
             const std::optional<uint64_t> t2) {
            constructFromExpected(self, mlir::CompilerTarget::Site::create(
                                            siteId, std::move(name), t1, t2));
          },
          "site_id"_a, "name"_a = nb::none(), "t1"_a = nb::none(),
          "t2"_a = nb::none())
      .def_prop_ro("id", &mlir::CompilerTarget::Site::id,
                   "The target-defined nonnegative site identifier.")
      .def_prop_ro(
          "name",
          [](const mlir::CompilerTarget::Site& site) {
            const auto name = site.name();
            return name ? std::optional<std::string>(name->str())
                        : std::nullopt;
          },
          "The reported site name, if available.")
      .def_prop_ro("t1", &mlir::CompilerTarget::Site::t1,
                   "The raw T1 coherence time, if available.")
      .def_prop_ro("t2", &mlir::CompilerTarget::Site::t2,
                   "The raw T2 coherence time, if available.");

  auto siteTuple = nb::class_<mlir::CompilerTarget::SiteTuple>(
      compilerTarget, "SiteTuple",
      "A supported ordered placement with optional calibration.");
  siteTuple
      .def(
          "__init__",
          [](mlir::CompilerTarget::SiteTuple& self,
             std::vector<mlir::CompilerTarget::SiteId> sites,
             const std::optional<uint64_t> duration,
             const std::optional<double> fidelity) {
            constructFromExpected(self,
                                  mlir::CompilerTarget::SiteTuple::create(
                                      std::move(sites), duration, fidelity));
          },
          "sites"_a, "duration"_a = nb::none(), "fidelity"_a = nb::none())
      .def_prop_ro(
          "sites",
          [](const mlir::CompilerTarget::SiteTuple& tuple) {
            return std::vector<mlir::CompilerTarget::SiteId>(
                tuple.sites().begin(), tuple.sites().end());
          },
          "The ordered target site identifiers.")
      .def_prop_ro("duration", &mlir::CompilerTarget::SiteTuple::duration,
                   "The raw operation duration, if available.")
      .def_prop_ro("fidelity", &mlir::CompilerTarget::SiteTuple::fidelity,
                   "The operation fidelity, if available.");

  nb::implicitly_convertible<std::vector<mlir::CompilerTarget::SiteId>,
                             mlir::CompilerTarget::SiteTuple>();

  nb::enum_<mlir::CompilerTarget::Operation::Arity::Kind>(
      compilerTarget, "OperationArityKind",
      "How an operation capability accepts qubit widths.")
      .value("FIXED", mlir::CompilerTarget::Operation::Arity::Kind::Fixed)
      .value("VARIADIC",
             mlir::CompilerTarget::Operation::Arity::Kind::Variadic);

  auto operationArity = nb::class_<mlir::CompilerTarget::Operation::Arity>(
      compilerTarget, "OperationArity", "Accepted operation qubit widths.");
  operationArity
      .def_static("fixed", &mlir::CompilerTarget::Operation::Arity::fixed,
                  "value"_a, "Create an exact operation arity.")
      .def_static("variadic", &mlir::CompilerTarget::Operation::Arity::variadic,
                  "minimum"_a,
                  "Create an operation arity with an inclusive minimum. "
                  "Operation construction requires a positive minimum.")
      .def_prop_ro("kind", &mlir::CompilerTarget::Operation::Arity::kind,
                   "The arity kind.")
      .def_prop_ro("value", &mlir::CompilerTarget::Operation::Arity::value,
                   "The exact arity or inclusive variadic minimum.")
      .def("accepts", &mlir::CompilerTarget::Operation::Arity::accepts,
           "width"_a, "Whether this arity accepts a concrete width.");

  auto targetOperation = nb::class_<mlir::CompilerTarget::Operation>(
      compilerTarget, "Operation",
      "A target operation capability, calibration, and ordered "
      "applicability.");
  targetOperation
      .def(
          "__init__",
          [](mlir::CompilerTarget::Operation& self, std::string name,
             const mlir::CompilerTarget::Operation::Arity arity,
             const size_t numParameters,
             std::optional<std::vector<mlir::CompilerTarget::SiteTuple>>
                 siteTuples,
             const std::optional<uint64_t> duration,
             const std::optional<double> fidelity) {
            constructFromExpected(
                self,
                mlir::CompilerTarget::Operation::create(
                    std::move(name), arity, numParameters,
                    std::move(siteTuples)
                        .value_or(
                            std::vector<mlir::CompilerTarget::SiteTuple>{}),
                    duration, fidelity));
          },
          "name"_a, "arity"_a, "num_parameters"_a, "site_tuples"_a = nb::none(),
          "duration"_a = nb::none(), "fidelity"_a = nb::none())
      .def(
          "__init__",
          [](mlir::CompilerTarget::Operation& self, std::string name,
             const size_t arity, const size_t numParameters,
             std::optional<std::vector<mlir::CompilerTarget::SiteTuple>>
                 siteTuples,
             const std::optional<uint64_t> duration,
             const std::optional<double> fidelity) {
            constructFromExpected(
                self,
                mlir::CompilerTarget::Operation::create(
                    std::move(name), arity, numParameters,
                    std::move(siteTuples)
                        .value_or(
                            std::vector<mlir::CompilerTarget::SiteTuple>{}),
                    duration, fidelity));
          },
          "name"_a, "arity"_a, "num_parameters"_a, "site_tuples"_a = nb::none(),
          "duration"_a = nb::none(), "fidelity"_a = nb::none())
      .def_prop_ro(
          "name",
          [](const mlir::CompilerTarget::Operation& operation) {
            return operation.name().str();
          },
          "The exact reported operation name.")
      .def_prop_ro(
          "canonical_name",
          [](const mlir::CompilerTarget::Operation& operation) {
            return operation.canonicalName().str();
          },
          "The normalized compiler operation name.")
      .def_prop_ro("arity", &mlir::CompilerTarget::Operation::arity,
                   "The accepted operation arity.")
      .def_prop_ro("num_parameters",
                   &mlir::CompilerTarget::Operation::numParameters,
                   "The number of real-valued parameters.")
      .def_prop_ro(
          "site_tuples",
          [](const mlir::CompilerTarget::Operation& operation) {
            return std::vector<mlir::CompilerTarget::SiteTuple>(
                operation.siteTuples().begin(), operation.siteTuples().end());
          },
          "Supported ordered placements with optional calibration; empty means "
          "general applicability.")
      .def_prop_ro("duration", &mlir::CompilerTarget::Operation::duration,
                   "The raw default duration, if available.")
      .def_prop_ro("fidelity", &mlir::CompilerTarget::Operation::fidelity,
                   "The default fidelity, if available.");

  nb::enum_<mlir::CompilerTarget::GateKind>(
      compilerTarget, "GateKind", "Recognized native gate capability.")
      .value("U", mlir::CompilerTarget::GateKind::U)
      .value("X", mlir::CompilerTarget::GateKind::X)
      .value("SX", mlir::CompilerTarget::GateKind::SX)
      .value("RZ", mlir::CompilerTarget::GateKind::RZ)
      .value("RX", mlir::CompilerTarget::GateKind::RX)
      .value("RY", mlir::CompilerTarget::GateKind::RY)
      .value("R", mlir::CompilerTarget::GateKind::R)
      .value("RXX", mlir::CompilerTarget::GateKind::RXX)
      .value("RYY", mlir::CompilerTarget::GateKind::RYY)
      .value("RZX", mlir::CompilerTarget::GateKind::RZX)
      .value("RZZ", mlir::CompilerTarget::GateKind::RZZ)
      .value("ISWAP", mlir::CompilerTarget::GateKind::ISWAP)
      .value("CZ", mlir::CompilerTarget::GateKind::CZ)
      .value("CX", mlir::CompilerTarget::GateKind::CX)
      .value("ECR", mlir::CompilerTarget::GateKind::ECR);

  nb::enum_<mlir::CompilerTarget::SingleQubitBasis>(
      compilerTarget, "SingleQubitBasis",
      "Recognized target-wide single-qubit synthesis basis.")
      .value("U", mlir::CompilerTarget::SingleQubitBasis::U)
      .value("ZSXX", mlir::CompilerTarget::SingleQubitBasis::ZSXX)
      .value("R", mlir::CompilerTarget::SingleQubitBasis::R)
      .value("XZX", mlir::CompilerTarget::SingleQubitBasis::XZX)
      .value("XYX", mlir::CompilerTarget::SingleQubitBasis::XYX)
      .value("ZYZ", mlir::CompilerTarget::SingleQubitBasis::ZYZ)
      .value("ZXZ", mlir::CompilerTarget::SingleQubitBasis::ZXZ);

  auto synthesisBasis = nb::class_<mlir::CompilerTarget::SynthesisBasis>(
      compilerTarget, "SynthesisBasis",
      "One synthesis basis usable across the complete target.");
  synthesisBasis
      .def_ro("single_qubit",
              &mlir::CompilerTarget::SynthesisBasis::singleQubit,
              "The single-qubit synthesis basis.")
      .def_ro("entangler", &mlir::CompilerTarget::SynthesisBasis::entangler,
              "The two-qubit entangler.");

  nb::enum_<mlir::CompilerTarget::Connectivity::Kind>(
      compilerTarget, "ConnectivityKind", "The target connectivity model.")
      .value("ALL_TO_ALL", mlir::CompilerTarget::Connectivity::Kind::AllToAll)
      .value("EXPLICIT", mlir::CompilerTarget::Connectivity::Kind::Explicit);

  auto connectivity = nb::class_<mlir::CompilerTarget::Connectivity>(
      compilerTarget, "Connectivity", "A target connectivity model.");
  connectivity
      .def(
          "__init__",
          [](mlir::CompilerTarget::Connectivity& self,
             const std::vector<mlir::CompilerTarget::Coupling>& couplings) {
            new (&self) mlir::CompilerTarget::Connectivity(
                mlir::CompilerTarget::Connectivity::fromCouplings(couplings));
          },
          "couplings"_a, "Create an explicit connectivity model.")
      .def_static("all_to_all", &mlir::CompilerTarget::Connectivity::allToAll,
                  "Create an all-to-all connectivity model.")
      .def_prop_ro("kind", &mlir::CompilerTarget::Connectivity::kind,
                   "The connectivity model.")
      .def_prop_ro(
          "couplings",
          [](const mlir::CompilerTarget::Connectivity& value) {
            return std::vector<mlir::CompilerTarget::Coupling>(
                value.couplings().begin(), value.couplings().end());
          },
          "The explicit couplings, if present.");

  nb::enum_<mlir::CompilerTarget::NativeOperations::Kind>(
      compilerTarget, "NativeOperationsKind",
      "The native-operation support model.")
      .value("UNRESTRICTED",
             mlir::CompilerTarget::NativeOperations::Kind::Unrestricted)
      .value("EXPLICIT",
             mlir::CompilerTarget::NativeOperations::Kind::Explicit);

  auto nativeOperations = nb::class_<mlir::CompilerTarget::NativeOperations>(
      compilerTarget, "NativeOperations", "Native-operation support.");
  nativeOperations
      .def(
          "__init__",
          [](mlir::CompilerTarget::NativeOperations& self,
             const std::vector<mlir::CompilerTarget::Operation>& operations) {
            new (&self) mlir::CompilerTarget::NativeOperations(
                mlir::CompilerTarget::NativeOperations::fromOperations(
                    operations));
          },
          "operations"_a, "Create explicit native-operation support.")
      .def_static("unrestricted",
                  &mlir::CompilerTarget::NativeOperations::unrestricted,
                  "Create unrestricted native-operation support.")
      .def_prop_ro("kind", &mlir::CompilerTarget::NativeOperations::kind,
                   "The native-operation support model.")
      .def_prop_ro(
          "operations",
          [](const mlir::CompilerTarget::NativeOperations& value) {
            return std::vector<mlir::CompilerTarget::Operation>(
                value.operations().begin(), value.operations().end());
          },
          "The explicit operations, if present.");

  compilerTarget
      .def(
          "__init__",
          [](mlir::CompilerTarget& self, const size_t numSites,
             mlir::CompilerTarget::Connectivity connectivity,
             mlir::CompilerTarget::NativeOperations nativeOperations,
             std::optional<mlir::CompilerTarget::DurationUnit> durationUnit) {
            constructFromExpected(self, mlir::CompilerTarget::create(
                                            numSites, std::move(connectivity),
                                            std::move(nativeOperations),
                                            std::move(durationUnit)));
          },
          "num_sites"_a, nb::kw_only(), "connectivity"_a, "native_operations"_a,
          "duration_unit"_a = nb::none())
      .def(
          "__init__",
          [](mlir::CompilerTarget& self, std::string name,
             const size_t numSites,
             mlir::CompilerTarget::Connectivity connectivity,
             mlir::CompilerTarget::NativeOperations nativeOperations,
             std::optional<mlir::CompilerTarget::DurationUnit> durationUnit) {
            constructFromExpected(
                self, mlir::CompilerTarget::create(std::move(name), numSites,
                                                   std::move(connectivity),
                                                   std::move(nativeOperations),
                                                   std::move(durationUnit)));
          },
          "name"_a, "num_sites"_a, nb::kw_only(), "connectivity"_a,
          "native_operations"_a, "duration_unit"_a = nb::none())
      .def(
          "__init__",
          [](mlir::CompilerTarget& self,
             std::vector<mlir::CompilerTarget::Site> sites,
             mlir::CompilerTarget::Connectivity connectivity,
             mlir::CompilerTarget::NativeOperations nativeOperations,
             std::optional<mlir::CompilerTarget::DurationUnit> durationUnit) {
            constructFromExpected(
                self, mlir::CompilerTarget::create(std::move(sites),
                                                   std::move(connectivity),
                                                   std::move(nativeOperations),
                                                   std::move(durationUnit)));
          },
          "sites"_a, nb::kw_only(), "connectivity"_a, "native_operations"_a,
          "duration_unit"_a = nb::none())
      .def(
          "__init__",
          [](mlir::CompilerTarget& self, std::string name,
             std::vector<mlir::CompilerTarget::Site> sites,
             mlir::CompilerTarget::Connectivity connectivity,
             mlir::CompilerTarget::NativeOperations nativeOperations,
             std::optional<mlir::CompilerTarget::DurationUnit> durationUnit) {
            constructFromExpected(self, mlir::CompilerTarget::create(
                                            std::move(name), std::move(sites),
                                            std::move(connectivity),
                                            std::move(nativeOperations),
                                            std::move(durationUnit)));
          },
          "name"_a, "sites"_a, nb::kw_only(), "connectivity"_a,
          "native_operations"_a, "duration_unit"_a = nb::none())
      .def_static(
          "from_device",
          [](const qdmi::Device& device) {
            return takeResult(mlir::compilerTargetFromDevice(device));
          },
          "device"_a, "Snapshot a circuit-model QDMI device.")
      .def_static(
          "from_device_id",
          [](const std::string& deviceId, std::optional<std::string> baseUrl,
             std::optional<std::string> token,
             std::optional<std::filesystem::path> authFile,
             std::optional<std::string> authUrl,
             std::optional<std::string> username,
             std::optional<std::string> password,
             std::optional<std::string> deviceConfig,
             std::optional<std::filesystem::path> deviceConfigFile,
             std::optional<std::string> custom1,
             std::optional<std::string> custom2,
             std::optional<std::string> custom3,
             std::optional<std::string> custom4,
             std::optional<std::string> custom5) {
            // Keep this preflight at the Python boundary so the public
            // ValueError does not depend on cross-extension exception
            // translation.
            if (deviceConfig && deviceConfigFile) {
              throw nb::value_error(
                  "device_config and device_config_file are mutually "
                  "exclusive");
            }
            const auto overrides = qdmi::makeDeviceSessionConfig(
                std::move(baseUrl), std::move(token), std::move(authFile),
                std::move(authUrl), std::move(username), std::move(password),
                std::move(deviceConfig), std::move(deviceConfigFile),
                std::move(custom1), std::move(custom2), std::move(custom3),
                std::move(custom4), std::move(custom5));
            auto device = qdmi::Session::openDevice(deviceId, overrides);
            return takeResult(mlir::compilerTargetFromDevice(device));
          },
          "device_id"_a, nb::kw_only(), "base_url"_a = std::nullopt,
          "token"_a = std::nullopt, "auth_file"_a = std::nullopt,
          "auth_url"_a = std::nullopt, "username"_a = std::nullopt,
          "password"_a = std::nullopt, "device_config"_a = std::nullopt,
          "device_config_file"_a = std::nullopt, "custom1"_a = std::nullopt,
          "custom2"_a = std::nullopt, "custom3"_a = std::nullopt,
          "custom4"_a = std::nullopt, "custom5"_a = std::nullopt,
          "Open a registered device and snapshot its compiler target.")
      .def_prop_ro(
          "name",
          [](const mlir::CompilerTarget& target) {
            const auto name = target.name();
            return name ? std::optional<std::string>(name->str())
                        : std::nullopt;
          },
          "The target name, if available.")
      .def_prop_ro("duration_unit", &mlir::CompilerTarget::durationUnit,
                   "The target timing unit, if available.")
      .def_prop_ro("num_sites", &mlir::CompilerTarget::numSites,
                   "The number of target sites.")
      .def_prop_ro(
          "sites",
          [](const mlir::CompilerTarget& target) {
            return std::vector<mlir::CompilerTarget::Site>(
                target.sites().begin(), target.sites().end());
          },
          "Detailed sites in compiler-vertex order.")
      .def_prop_ro("connectivity_kind", &mlir::CompilerTarget::connectivityKind,
                   "The target connectivity model.")
      .def_prop_ro(
          "couplings",
          [](const mlir::CompilerTarget& target) {
            return std::vector<mlir::CompilerTarget::Coupling>(
                target.couplings().begin(), target.couplings().end());
          },
          "Canonical undirected couplings in target site IDs.")
      .def_prop_ro("native_operations_kind",
                   &mlir::CompilerTarget::nativeOperationsKind,
                   "The target native-operation support model.")
      .def_prop_ro(
          "operations",
          [](const mlir::CompilerTarget& target) {
            return std::vector<mlir::CompilerTarget::Operation>(
                target.operations().begin(), target.operations().end());
          },
          "Operation capabilities in reported order.")
      .def_prop_ro(
          "supported_gates",
          [](const mlir::CompilerTarget& target) {
            return std::vector<mlir::CompilerTarget::GateKind>(
                target.supportedGates().begin(), target.supportedGates().end());
          },
          "Recognized native gates supported by the target.")
      .def_prop_ro("synthesis_basis", &mlir::CompilerTarget::synthesisBasis,
                   "A complete target-wide synthesis basis, if available.")
      .def(
          "supports_operation",
          [](const mlir::CompilerTarget& target, const std::string_view name,
             const size_t arity, const std::optional<size_t> numParameters,
             const std::optional<std::vector<mlir::CompilerTarget::SiteId>>&
                 sites) {
            if (sites) {
              return target.supportsOperation(name, arity, numParameters,
                                              *sites);
            }
            return target.supportsOperation(name, arity, numParameters);
          },
          "name"_a, "arity"_a, "num_parameters"_a = nb::none(),
          "sites"_a = nb::none(), "Whether the target supports an operation.");

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
          "source"_a, "Parse a QC MLIR source string.")
      .def_static(
          "from_mlir_file",
          &OptionalFunctionAdapter<&mlir::QCProgram::fromMLIRFile>::call,
          "path"_a, "Parse QC MLIR from a file.")
      .def_static(
          "from_qasm_str",
          &OptionalFunctionAdapter<&mlir::QCProgram::fromQASMString>::call,
          "source"_a, "Translate an OpenQASM 3 source string to QC MLIR.")
      .def_static(
          "from_qasm_file",
          &OptionalFunctionAdapter<&mlir::QCProgram::fromQASMFile>::call,
          "path"_a, "Translate an OpenQASM 3 file to QC MLIR.")
      .def_static(
          "from_qiskit",
          [](const nb::object& circuit) {
            return bindings::qiskit::importCircuit(circuit);
          },
          "circuit"_a,
          nb::sig("def from_qiskit(circuit: qiskit.circuit.QuantumCircuit) "
                  "-> QCProgram"),
          R"pb(Translate a Qiskit {py:class}`~qiskit.circuit.QuantumCircuit` to QC MLIR.)pb")
      .def("copy", &mlir::QCProgram::copy,
           "Return an independent copy of this program.")
      .def("cleanup", &BooleanMemberAdapter<&mlir::QCProgram::cleanup>::call,
           "Run the standard QC cleanup pipeline in place.")
      .def("normalize_global_phases",
           &BooleanMemberAdapter<&mlir::QCProgram::normalizeGlobalPhases>::call,
           "Normalize scoped global phases in place.")
      .def(
          "to_openqasm3",
          [](const mlir::QCProgram& program) {
            requireValid(program);
            try {
              return takeFailureOr(
                  program.module().getContext(),
                  "cannot export QC program to OpenQASM 3",
                  [&]() -> mlir::FailureOr<mlir::OpenQASMProgram> {
                    auto result = program.toOpenQASM3();
                    if (!result) {
                      return mlir::failure();
                    }
                    return std::move(*result);
                  });
            } catch (const nb::builtin_exception& error) {
              throw std::runtime_error(error.what());
            }
          },
          "Clean up and emit this QC program as OpenQASM 3 without QCO "
          "optimization.")
      .def(
          "to_qiskit",
          [](const mlir::QCProgram& program,
             const mlir::CompilerTarget* const target) {
            requireValid(program);
            return bindings::qiskit::exportCircuit(program, target);
          },
          nb::kw_only(), "target"_a = nb::none(),
          nb::sig("def to_qiskit(self, *, target: CompilerTarget | None = "
                  "None) -> qiskit.circuit.QuantumCircuit"),
          R"pb(Translate this QC program to a Qiskit {py:class}`~qiskit.circuit.QuantumCircuit` without consuming it.

Args:
    target: The optional compiler target used for mapping. When provided, emit
        a canonical physical circuit. All qubits must be static, and their site
        IDs must belong to the target.)pb")
      .def(
          "to_qco",
          [](mlir::QCProgram& value, const bool copy) {
            auto source = copiedOrConsumed(value, copy);
            return takeResult(std::move(source).intoQCO());
          },
          nb::kw_only(), "copy"_a = false,
          R"pb(Convert this program to QCO.

Set ``copy=True`` to preserve it.)pb")
      .def(
          "to_qir",
          [](mlir::QCProgram& value, const mlir::QIRProfile profile,
             const bool copy) {
            auto source = copiedOrConsumed(value, copy);
            return takeResult(std::move(source).intoQIR(profile));
          },
          "profile"_a, nb::kw_only(), "copy"_a = false,
          R"pb(Lower this program to QIR for the requested profile.

Set ``copy=True`` to preserve it.)pb")
      .def(
          "num_gates",
          [](const mlir::QCProgram& program) {
            requireValid(program);
            return program.numGates();
          },
          R"pb(Count the gates in the program.

Any operation that implements the ``UnitaryOpInterface`` is counted. Operations
in every structured control-flow region are counted once, regardless of how
often the region executes. Operations within modifiers are not counted
recursively, and barriers are skipped.)pb")
      .def(
          "num_single_qubit_gates",
          [](const mlir::QCProgram& program) {
            requireValid(program);
            return program.numSingleQubitGates();
          },
          R"pb(Count the single-qubit gates in the program.

Any operation that implements the ``UnitaryOpInterface`` and acts on one qubit
is counted. Operations in every structured control-flow region are counted
once, regardless of how often the region executes. Operations within modifiers
are not counted recursively, and barriers are skipped.)pb")
      .def(
          "num_two_qubit_gates",
          [](const mlir::QCProgram& program) {
            requireValid(program);
            return program.numTwoQubitGates();
          },
          R"pb(Count the two-qubit gates in the program.

Any operation that implements the ``UnitaryOpInterface`` and acts on two qubits
is counted. Operations in every structured control-flow region are counted
once, regardless of how often the region executes. Operations within modifiers
are not counted recursively, and barriers are skipped.)pb");

  auto qcoProgram = nb::class_<mlir::QCOProgram, mlir::Program>(
      m, "QCOProgram", R"pb(A compiler program in the QCO dialect.

QCO programs use value semantics and expose optimization and transformation
operations.)pb");
  qcoProgram
      .def_static(
          "from_mlir_str",
          &OptionalFunctionAdapter<&mlir::QCOProgram::fromMLIRString>::call,
          "source"_a, "Parse a QCO MLIR source string.")
      .def_static(
          "from_mlir_file",
          &OptionalFunctionAdapter<&mlir::QCOProgram::fromMLIRFile>::call,
          "path"_a, "Parse QCO MLIR from a file.")
      .def("copy", &mlir::QCOProgram::copy,
           "Return an independent copy of this program.")
      .def("cleanup", &BooleanMemberAdapter<&mlir::QCOProgram::cleanup>::call,
           "Run the standard QCO cleanup pipeline in place.")
      .def(
          "normalize_global_phases",
          &BooleanMemberAdapter<&mlir::QCOProgram::normalizeGlobalPhases>::call,
          "Normalize scoped global phases in place.")
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
      .def("unroll_quantum_loops",
           &BooleanMemberAdapter<&mlir::QCOProgram::unrollQuantumLoops>::call,
           nb::kw_only(), "unroll_factor"_a = -1,
           "Unroll quantum loops, optionally using a maximum unroll factor.")
      .def("lift_hadamards",
           &BooleanMemberAdapter<&mlir::QCOProgram::liftHadamards>::call,
           "Move Hadamard gates through compatible operations.")
      .def("reuse_qubits",
           &BooleanMemberAdapter<&mlir::QCOProgram::reuseQubits>::call,
           "Reuse independent single-qubit allocations.")
      .def(
          "run_qubit_reuse_pipeline",
          &BooleanMemberAdapter<&mlir::QCOProgram::runQubitReusePipeline>::call,
          "Prepare the program for qubit reuse and reuse eligible qubits.")
      .def("decompose_multi_controlled",
           &BooleanMemberAdapter<
               &mlir::QCOProgram::decomposeMultiControlled>::call,
           nb::kw_only(), "min_qubits"_a = 3,
           "Decompose controlled X/Z/SWAP gates, qco.rccx, and constant-angle "
           "phase gates that act on at least min_qubits qubits (min_qubits "
           "must be at least 3; default 3 means wider than two-qubit).")
      .def("compile_for_target",
           &BooleanMemberAdapter<&mlir::QCOProgram::compileForTarget>::call,
           "target"_a, nb::kw_only(), "enable_timing"_a = false,
           "enable_statistics"_a = false,
           "Compile this QCO program for the target in place. Do not rely on "
           "its contents if compilation fails.")
      .def(
          "to_qc",
          [](mlir::QCOProgram& value, const bool copy) {
            auto source = copiedOrConsumed(value, copy);
            return takeResult(std::move(source).intoQC());
          },
          nb::kw_only(), "copy"_a = false,
          R"pb(Convert this program to QC.

Set ``copy=True`` to preserve it.)pb")
      .def(
          "to_jeff",
          [](mlir::QCOProgram& value, const bool copy) {
            auto source = copiedOrConsumed(value, copy);
            return takeResult(std::move(source).intoJeff());
          },
          nb::kw_only(), "copy"_a = false,
          R"pb(Serialize this program as ``jeff``.

Set ``copy=True`` to preserve it.)pb");

  auto jeffProgram = nb::class_<mlir::JeffProgram, mlir::Program>(
      m, "JeffProgram", R"pb(A serialized ``jeff`` compiler program.

``jeff`` programs can be stored as bytes or files and converted back to QCO for
further compilation.)pb");
  jeffProgram
      .def_static("from_file",
                  &OptionalFunctionAdapter<&mlir::JeffProgram::fromFile>::call,
                  "path"_a, "Read a ``jeff`` program from a file.")
      .def_static(
          "from_bytes",
          [](const nb::bytes& bytes) {
            const auto view =
                std::span(reinterpret_cast<const std::byte*>(bytes.c_str()),
                          bytes.size());
            return takeResult(mlir::JeffProgram::fromBytes(view));
          },
          "data"_a, "Deserialize a ``jeff`` program from bytes.")
      .def("copy", &mlir::JeffProgram::copy,
           "Return an independent copy of this program.")
      .def("cleanup", &BooleanMemberAdapter<&mlir::JeffProgram::cleanup>::call,
           "Run the standard ``jeff`` cleanup pipeline in place.")
      .def(
          "to_bytes",
          [](const mlir::JeffProgram& value) {
            requireValid(value);
            const auto bytes = value.toBytes();
            return nb::bytes(bytes.data(), bytes.size());
          },
          "Serialize this program to its ``jeff`` byte representation.")
      .def("write", &BooleanMemberAdapter<&mlir::JeffProgram::write>::call,
           "path"_a, "Write this program to a ``jeff`` file.")
      .def(
          "to_qco",
          [](mlir::JeffProgram& value, const bool copy) {
            auto source = copiedOrConsumed(value, copy);
            return takeResult(std::move(source).intoQCO());
          },
          nb::kw_only(), "copy"_a = false,
          R"pb(Deserialize this program to QCO.

Set ``copy=True`` to preserve it.)pb");

  nb::class_<mlir::OpenQASMProgram>(
      m, "OpenQASMProgram",
      "An immutable compiler program containing OpenQASM 3 source.")
      .def_prop_ro("source", &mlir::OpenQASMProgram::source,
                   "The emitted OpenQASM 3 source.")
      .def("write", &BooleanMemberAdapter<&mlir::OpenQASMProgram::write>::call,
           "path"_a, "Write the emitted source to a file.")
      .def("__str__", &mlir::OpenQASMProgram::str,
           "Return the emitted OpenQASM 3 source.");

  auto qirProgram = nb::class_<mlir::QIRProgram, mlir::Program>(
      m, "QIRProgram", R"pb(A compiler program lowered to QIR.

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

  nb::module_::import_("mqt.core.dd");

  qcoProgram.def(
      "build_functionality", &buildQCOFunctionality, "dd_package"_a,
      // Keep the DD package alive while the returned matrix DD is alive.
      nb::keep_alive<0, 2>(),
      R"pb(Build a matrix DD for a static unitary QCO program.

Args:
    dd_package: DD package with enough qubits for the program.

Returns:
    Matrix DD of the program functionality.

Raises:
    ValueError: When the program is unsupported for functionality construction.)pb");

  qcoProgram.def(
      "simulate", &simulateQCO, "initial_state"_a, "dd_package"_a,
      "seed"_a = 0U,
      // Keep the DD package alive while the returned vector DD is alive.
      nb::keep_alive<0, 3>(),
      R"pb(Simulate a QCO program on a DD state.

Args:
    initial_state: Input state DD that spans at least the program's qubits and
        has a live reference in ``dd_package``. Higher wires are preserved. A
        valid input reference is consumed.
    dd_package: DD package with enough qubits for the program.
    seed: RNG seed. ``0`` (default) selects nondeterministic seeding. Any other
        value produces reproducible measurement and reset results.

Returns:
    Output state DD.

Raises:
    ValueError: When ``initial_state`` has no live reference in ``dd_package``,
        has too few qubits, or the program is unsupported for simulation.)pb");

  qcoProgram.def("sample", &sampleQCO, "shots"_a = 1024U, "seed"_a = 0U,
                 R"pb(Sample the declared outputs of a QCO program.

Args:
    shots: Number of shots (default 1024).
    seed: RNG seed. ``0`` (default) selects nondeterministic seeding. Any other
        value produces reproducible results.

Returns:
    Histogram keys use conventional count-string order. The last returned
    register comes first, and each register is MSB-first. If no CBit result
    exists, final ``measureAll`` bitstrings are used instead.

Raises:
    ValueError: When the program is unsupported for sampling.)pb");

  m.def("build_functionality", &buildDenseFunctionality, "program"_a,
        nb::sig("def build_functionality(program: str | os.PathLike[str] | "
                "qiskit.circuit.QuantumCircuit | QCProgram | QCOProgram | "
                "JeffProgram | OpenQASMProgram) -> "
                "typing.Annotated[numpy.typing.NDArray[numpy.complex128], "
                "{'shape': (None, None)}]"),
        R"pb(Build the full unitary matrix of a supported compiler input.

The DD package is managed internally. The matrix is materialized directly into
the returned NumPy array without an additional copy. The full matrix grows
exponentially, and the caller is responsible for requesting a result that fits
in memory.

Raises:
    MemoryError: When the dense matrix does not fit in memory.
    ValueError: When the program is unsupported or the matrix dimensions exceed
        addressable memory.)pb");

  m.def("simulate", &simulateDense, "program"_a,
        nb::sig("def simulate(program: str | os.PathLike[str] | "
                "qiskit.circuit.QuantumCircuit | QCProgram | QCOProgram | "
                "JeffProgram | OpenQASMProgram) -> "
                "typing.Annotated[numpy.typing.NDArray[numpy.complex128], "
                "{'shape': (None,)}]"),
        R"pb(Simulate a closed compiler input from the all-zero state.

The DD package is managed internally. Terminal measurements that only assemble
returned classical registers do not collapse the state. Mid-circuit measurement
feedback and resets are unsupported; use {py:meth}`QCOProgram.simulate` with an
explicit DD package for those workflows or for a custom initial state.

Args:
    program: Compiler input to lower directly to QCO.

Returns:
    Full statevector, materialized directly into the returned NumPy array.

Raises:
    MemoryError: When the dense statevector does not fit in memory.
    ValueError: When the program is not closed, is unsupported for statevector
        simulation, or the statevector dimensions exceed addressable memory.)pb");

  m.def("sample", &sample, "program"_a, "shots"_a = 1024U, "seed"_a = 0U,
        nb::sig("def sample(program: str | os.PathLike[str] | "
                "qiskit.circuit.QuantumCircuit | QCProgram | QCOProgram | "
                "JeffProgram | OpenQASMProgram, shots: int = 1024, seed: int = "
                "0) -> dict[str, int]"),
        R"pb(Sample a supported input after lowering it directly to QCO.

An existing QCO program is used without copying. See
{py:meth}`QCOProgram.sample` for the shot, seed, histogram, and error
contracts.)pb");

  m.def("compile_program", &compileProgram, "program"_a, nb::kw_only(),
        "output"_a = mlir::ProgramFormat::QC, "inplace"_a = false,
        "target"_a = nb::none(), "qco_pipeline"_a = "mqt-qco-default",
        "enable_timing"_a = false, "enable_statistics"_a = false,
        R"pb(
Run the coordinated default MQT compiler pipeline.

Input source strings, files, Qiskit
{py:class}`~qiskit.circuit.QuantumCircuit` objects, and typed compiler programs
can be combined with any supported output format. Typed program inputs are
copied by default; set ``inplace=True`` to consume them. Use the typed programs
directly to construct a custom pipeline stage by stage.

Args:
    program: Source text, a file path, a Qiskit circuit, or a typed compiler program.
    output: The requested output stage of the compiler pipeline.
    inplace: Whether a typed input program may be consumed.
    target: An optional compiler target for decomposition, mapping, and native
        synthesis. A target requires optimized QCO, QC, or QIR output.
    qco_pipeline: The QCO optimization pipeline to run. A custom pipeline
        cannot be combined with a target.
    enable_timing: Whether to collect pass timing information.
    enable_statistics: Whether to collect pass statistics.

Returns:
    A typed compiler program for the requested output format.
)pb");
}

} // namespace mqt
